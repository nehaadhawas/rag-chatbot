from backend.chunker import PDFChunker

def test_chunker():
    # Initialize the chunker
    chunker = PDFChunker()
    
    # IMPORTANT: Change this to your PDF file path!
    pdf_path = "sample.pdf"  # Change this to your actual PDF
    
    try:
        print("[PDF] Extracting text from PDF...")
        pages = chunker.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(pages)} pages")
        
        print("\n[Chunker] Chunking text...")
        chunks = chunker.chunk_text(pages)
        print(f"Created {len(chunks)} chunks")
        
        # Display sample chunks
        print("\n--- First 3 chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  Page: {chunk['page_number']}")
            print(f"  Tokens: {chunk['token_count']}")
            print(f"  Preview: {chunk['text'][:200]}...")
            
    except FileNotFoundError:
        print(f"[Error] PDF file not found at '{pdf_path}'")
        print("   Please make sure the file exists and the path is correct")
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    test_chunker()
