import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini model loaded")
    
    def generate_answer(self, question: str, chunks: list) -> str:
        context = "\n\n".join([chunk["chunk"]["text"] for chunk in chunks])
        
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"Error: {str(e)}" 
