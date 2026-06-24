from backend.vector_store import VectorStore
from backend.embedder import Embedder
from backend.llm import LLM

def test_llm():
    store = VectorStore()
    store.load()
    
    question = "What is RAG?"
    print(f"Question: {question}")
    
    embedder = Embedder()
    query_vector = embedder.embed_text(question)
    
    results = store.search(query_vector, k=3)
    
    llm = LLM()
    answer = llm.generate_answer(question, results)
    
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    test_llm()