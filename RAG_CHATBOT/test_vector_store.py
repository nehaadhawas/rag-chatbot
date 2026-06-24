from backend.chunker import PDFChunker
from backend.embedder import Embedder
from backend.vector_store import VectorStore

def test_vector_store():
    chunker = PDFChunker()
    pages = chunker.extract_text_from_pdf("sample.pdf")
    chunks = chunker.chunk_text(pages)
    
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    
    store = VectorStore()
    store.add_chunks(chunks)
    store.save()
    
    print(f"Saved {len(chunks)} chunks to FAISS")
    
    store2 = VectorStore()
    store2.load()
    print(f"Loaded {len(store2.chunks)} chunks from FAISS")

if __name__ == "__main__":
    test_vector_store()