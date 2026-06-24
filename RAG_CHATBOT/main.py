import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.chunker import PDFChunker
from backend.embedder import Embedder
from backend.vector_store import VectorStore
from backend.llm import LLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
logger.info("Initializing RAG chatbot components...")
store = VectorStore()
# Load existing index if it exists, otherwise initialize empty
if not store.load():
    logger.info("No pre-existing vector store found. Initializing empty index.")
    store.create_index()

embedder = Embedder()
llm = LLM()

app = FastAPI(
    title="RAG Chatbot API",
    description="A minimalist backend API for the PDF-based RAG Chatbot.",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat(request: QueryRequest):
    """
    Handle query request by embedding the question, searching the vector store
    for the most relevant chunks, and generating an answer using the LLM.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # 1. Embed query
        query_vector = embedder.embed_text(request.question)
        
        # 2. Search vector store
        results = store.search(query_vector, k=3)
        
        # 3. Generate answer
        answer = llm.generate_answer(request.question, results)
        
        # 4. Formulate source metadata for frontend
        sources = []
        for res in results:
            sources.append({
                "text": res["chunk"]["text"],
                "page_number": res["chunk"].get("page_number", "Unknown"),
                "distance": res["distance"]
            })
            
        return {
            "question": request.question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF file, chunk it, generate embeddings,
    add to vector store, and save the updated index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_file_path = None
    try:
        # Create a temporary file to save the uploaded PDF bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            temp_file_path = tmp.name
        
        # 1. Extract text and chunk
        chunker = PDFChunker()
        pages = chunker.extract_text_from_pdf(temp_file_path)
        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")
            
        chunks = chunker.chunk_text(pages)
        
        # 2. Generate embeddings
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        
        # 3. Add to vector store and save
        store.add_chunks(chunks_with_embeddings)
        store.save()
        
        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}'. Added {len(chunks)} text chunks to the database.",
            "chunks_added": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

# Mount static files folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def get_index():
    """Serve the main index.html file."""
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8005
    uvicorn.run(app, host="127.0.0.1", port=8005)
