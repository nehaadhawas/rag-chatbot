from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles text embedding using sentence-transformers.
    
    This class:
    1. Loads the all-MiniLM-L6-v2 model
    2. Converts text to 384-dimensional vectors
    3. Provides batch processing for efficiency
    
    Why all-MiniLM-L6-v2?
    - Fast and lightweight
    - Good quality embeddings
    - 384 dimensions (balanced for FAISS)
    - Works well for semantic search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence-transformers model.
        
        Args:
            model_name: Name of the pre-trained model to use
                       (default: "all-MiniLM-L6-v2")
        
        Notes:
            - Model will be downloaded on first use
            - Download size: ~80MB
            - First load may take 5-10 seconds
        """
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_embedding_dimension()
            logger.info(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
            logger.info(f"   Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text to an embedding vector.
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of shape (embedding_dimension,)
            
        Example:
            embedder = Embedder()
            vector = embedder.embed_text("Hello world")
            # vector shape: (384,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension)
        
        try:
            # Single text embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Convert multiple texts to embedding vectors in batch.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show a progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dimension)
            
        Example:
            embedder = Embedder()
            texts = ["Hello", "World", "RAG Chatbot"]
            vectors = embedder.embed_batch(texts)
            # vectors shape: (3, 384)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts in batch")
            return np.array([])
        
        try:
            logger.info(f"🔄 Embedding {len(valid_texts)} texts...")
            start_time = time.time()
            
            # Batch embedding
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=show_progress,
                batch_size=32  # Process 32 at a time
            )
            
            logger.info(f"✅ Embedding complete in {time.time() - start_time:.2f} seconds")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed batch: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], 
                    show_progress: bool = False) -> List[Dict[str, Any]]:
        """
        Embed a list of chunks and add the vectors to each chunk.
        
        Args:
            chunks: List of chunk dictionaries (from PDFChunker)
            show_progress: Whether to show progress bar
            
        Returns:
            Same chunks but with 'embedding' key added
            
        Example:
            chunks = chunker.chunk_text(pages)
            chunks_with_embeddings = embedder.embed_chunks(chunks)
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return []
        
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Get embeddings
        embeddings = self.embed_batch(texts, show_progress)
        
        # Add embeddings back to chunks
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                chunk["embedding"] = embeddings[i]
            else:
                # This shouldn't happen, but just in case
                chunk["embedding"] = np.zeros(self.embedding_dimension)
        
        logger.info(f"✅ Added embeddings to {len(chunks)} chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Integer dimension (384 for all-MiniLM-L6-v2)
        """
        return self.embedding_dimension