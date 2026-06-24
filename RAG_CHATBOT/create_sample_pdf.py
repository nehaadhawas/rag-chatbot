# First install fpdf if you don't have it
# Run: pip install fpdf

from fpdf import FPDF

def create_sample_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add 5 pages of content
    for page_num in range(1, 6):
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt=f"Page {page_num}", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        
        # Sample content about RAG
        content = [
            "Retrieval Augmented Generation (RAG) is an AI framework that combines",
            "information retrieval with large language models. It enhances the",
            "accuracy and reliability of generative AI models by grounding them",
            "in external knowledge sources.",
            "",
            "The RAG process works in three main steps:",
            "1. Retrieval: Finding relevant information from a knowledge base",
            "2. Augmentation: Adding the retrieved information to the prompt",
            "3. Generation: Producing a response based on the augmented context",
            "",
            "This approach reduces hallucinations and provides verifiable sources.",
            "RAG is widely used in question answering, chatbots, and research.",
            "It allows AI models to access up-to-date information without retraining.",
        ]
        
        for line in content:
            pdf.multi_cell(0, 10, line)
            pdf.ln(3)
    
    pdf.output("sample.pdf")
    print("Created sample.pdf successfully!")
    print("   You can now run: python test_chunker.py")

if __name__ == "__main__":
    create_sample_pdf()
