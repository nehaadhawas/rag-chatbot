from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from typing import List, Optional
import tempfile

from backend.chunker import PDFChunker
from backend.embedder import Embedder
from backend.vector_store import VectorStore
from backend.llm import LLM

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chunker = PDFChunker()
embedder = Embedder()
vector_store = VectorStore()
llm = LLM()

vector_store.load()

TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple files and process them"""
    
    if not files:
        raise HTTPException(400, "No files uploaded")
    
    # Clear existing index
    vector_store.clear()
    
    all_chunks = []
    doc_names = []
    
    for file in files:
        if not file.filename.endswith(('.pdf', '.txt')):
            continue
            
        file_path = os.path.join(TEMP_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            if file.filename.endswith('.pdf'):
                pages = chunker.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                pages = [{"page_number": 1, "text": content}]
            
            chunks = chunker.chunk_text(pages)
            for chunk in chunks:
                chunk["doc_name"] = file.filename
            all_chunks.extend(chunks)
            doc_names.append(file.filename)
            
            os.remove(file_path)
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(500, f"Error processing {file.filename}: {str(e)}")
    
    if not all_chunks:
        raise HTTPException(400, "No valid files processed")
    
    # Embed and store all chunks
    all_chunks = embedder.embed_chunks(all_chunks)
    vector_store.add_multiple_files(all_chunks, ", ".join(doc_names[:3]) + ("..." if len(doc_names) > 3 else ""))
    vector_store.save()
    
    return {
        "message": f"Processed {len(files)} files successfully!",
        "total_chunks": len(all_chunks),
        "files": doc_names
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer based on the indexed documents."""
    
    if vector_store.index is None or vector_store.index.ntotal == 0:
        raise HTTPException(400, "No documents indexed. Please upload files first.")
    
    query_vector = embedder.embed_text(request.question)
    results = vector_store.search(query_vector, k=request.top_k)
    
    if not results:
        return AnswerResponse(
            answer="No relevant information found in the documents.",
            sources=[]
        )
    
    answer = llm.generate_answer(request.question, results)
    
    sources = []
    for r in results:
        sources.append({
            "text": r["chunk"]["text"][:300] + "...",
            "page": r["chunk"]["page_number"],
            "doc_id": r["chunk"].get("doc_id", "Unknown"),
            "distance": r["distance"]
        })
    
    return AnswerResponse(answer=answer, sources=sources)


@app.get("/documents")
def get_documents():
    """Get list of uploaded documents"""
    return {"documents": vector_store.get_documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a specific document"""
    success = vector_store.delete_document(doc_id)
    vector_store.save()
    if success:
        return {"message": f"Document {doc_id} deleted successfully"}
    return {"message": "Document not found"}


@app.get("/status")
def get_status():
    return {
        "documents_indexed": vector_store.index is not None and vector_store.index.ntotal > 0,
        "total_chunks": vector_store.index.ntotal if vector_store.index else 0,
        "total_documents": len(vector_store.get_documents())
    }


@app.post("/clear")
def clear_index():
    vector_store.clear()
    vector_store.save()
    return {"message": "Index cleared"}


app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")