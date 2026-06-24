import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension=384, index_path="data/index"):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.chunks = []
        self.documents = []  # Track documents
        
        os.makedirs(index_path, exist_ok=True)
        self.load_documents_metadata()
        
    def load_documents_metadata(self):
        """Load document list from metadata file"""
        meta_file = f"{self.index_path}/documents.json"
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                self.documents = json.load(f)
        else:
            self.documents = []
        
    def save_documents_metadata(self):
        """Save document list to metadata file"""
        meta_file = f"{self.index_path}/documents.json"
        with open(meta_file, "w") as f:
            json.dump(self.documents, f, indent=2)
        
    def create_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created FAISS index with dimension {self.dimension}")
    
    def clear(self):
        self.index = None
        self.chunks = []
        self.documents = []
        self.save_documents_metadata()
        logger.info("Index cleared")
        
    def add_chunks(self, chunks: List[Dict[str, Any]], doc_name: str = "Unknown"):
        if self.index is None:
            self.create_index()
        
        # Store document metadata
        doc_id = f"doc_{len(self.documents) + 1}"
        self.documents.append({
            "id": doc_id,
            "name": doc_name,
            "chunks": len(chunks),
            "pages": chunks[-1]["page_number"] if chunks else 0
        })
        self.save_documents_metadata()
            
        vectors = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        self.index.add(vectors)
        
        for chunk in chunks:
            chunk_copy = chunk.copy()
            chunk_copy.pop("embedding", None)
            chunk_copy["doc_id"] = doc_id
            self.chunks.append(chunk_copy)
        
        logger.info(f"Added {len(chunks)} chunks from {doc_name}")
        
    def add_multiple_files(self, all_chunks: List[Dict[str, Any]], doc_name: str = "Unknown"):
        """Add chunks from multiple files with document tracking"""
        if self.index is None:
            self.create_index()
        
        doc_id = f"doc_{len(self.documents) + 1}"
        
        # Count unique documents
        unique_docs = set()
        for chunk in all_chunks:
            unique_docs.add(chunk.get("doc_name", "Unknown"))
        
        self.documents.append({
            "id": doc_id,
            "name": doc_name,
            "chunks": len(all_chunks),
            "pages": all_chunks[-1]["page_number"] if all_chunks else 0
        })
        self.save_documents_metadata()
        
        vectors = np.array([chunk["embedding"] for chunk in all_chunks], dtype=np.float32)
        self.index.add(vectors)
        
        for chunk in all_chunks:
            chunk_copy = chunk.copy()
            chunk_copy.pop("embedding", None)
            chunk_copy["doc_id"] = doc_id
            self.chunks.append(chunk_copy)
        
        logger.info(f"Added {len(all_chunks)} chunks from {doc_name}")
        
    def search(self, query_vector: np.ndarray, k: int = 3):
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": float(distances[0][i])
                })
                
        return results
    
    def delete_document(self, doc_id: str):
        """Delete a document and its chunks"""
        # Remove from documents list
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        self.save_documents_metadata()
        
        # Remove chunks from that document
        self.chunks = [c for c in self.chunks if c.get("doc_id") != doc_id]
        
        # Rebuild index
        if self.chunks:
            self.create_index()
            vectors = np.array([c["embedding"] for c in self.chunks if "embedding" in c], dtype=np.float32)
            self.index.add(vectors)
        else:
            self.index = None
            
        logger.info(f"Deleted document {doc_id}")
        return True
    
    def get_documents(self):
        return self.documents
        
    def save(self):
        if self.index is None:
            return
            
        faiss.write_index(self.index, f"{self.index_path}/index.bin")
        
        with open(f"{self.index_path}/chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)
            
        self.save_documents_metadata()
        logger.info(f"Saved index with {self.index.ntotal} vectors")
        
    def load(self):
        index_file = f"{self.index_path}/index.bin"
        chunks_file = f"{self.index_path}/chunks.json"
        
        if not os.path.exists(index_file):
            logger.warning("No saved index found")
            return False
            
        self.index = faiss.read_index(index_file)
        
        with open(chunks_file, "r") as f:
            self.chunks = json.load(f)
            
        self.load_documents_metadata()
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        return True