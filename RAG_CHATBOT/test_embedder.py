from backend.chunker import PDFChunker
from backend.embedder import Embedder

def test_embedder():
    chunker = PDFChunker()
    pages = chunker.extract_text_from_pdf("sample.pdf")
    print(f"Extracted {len(pages)} pages")
    
    chunks = chunker.chunk_text(pages)
    print(f"Created {len(chunks)} chunks")
    
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    
    print(f"Embedded {len(chunks)} chunks")
    print(f"Vector size: {chunks[0]['embedding'].shape}")

if __name__ == "__main__":
    test_embedder()