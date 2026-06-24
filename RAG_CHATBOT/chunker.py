import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
import tiktoken


class PDFChunker:
    """
    Handles PDF reading and text chunking with overlap.
    
    This class is responsible for:
    1. Extracting text from PDF files
    2. Cleaning the extracted text
    3. Splitting text into overlapping chunks for better retrieval
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker with token limits.
        
        Args:
            chunk_size: Number of tokens per chunk (default: 500)
            overlap: Number of overlapping tokens between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Initialize tokenizer for accurate token counting
        # cl100k_base is used by GPT-4 and Gemini models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from all pages of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries with page number and text content
            
        Raises:
            Exception: If PDF cannot be read
        """
        pages = []
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            
            # Iterate through each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean the extracted text
                text = self._clean_text(text)
                
                # Only add non-empty pages
                if text.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text
                    })
            
            doc.close()
            return pages
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove multiple newlines (replace 2+ newlines with 2 newlines)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive spaces (replace 2+ spaces with 1 space)
        text = re.sub(r' +', ' ', text)
        
        # Remove special characters but keep basic punctuation
        # This helps with embedding quality
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        return text.strip()
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split text from pages into overlapping chunks.
        
        Args:
            pages: List of page dictionaries with 'page_number' and 'text'
            
        Returns:
            List of chunk dictionaries with:
                - text: The chunk text
                - page_number: Which page it came from
                - token_count: Number of tokens
                - chunk_id: Unique identifier for the chunk
        """
        chunks = []
        
        # Process each page
        for page in pages:
            page_text = page["text"]
            page_num = page["page_number"]
            
            # First split by paragraphs (better semantic boundaries)
            paragraphs = page_text.split('\n\n')
            
            current_chunk = ""
            current_tokens = 0
            
            # Build chunks by adding paragraphs
            for paragraph in paragraphs:
                # Skip empty paragraphs
                if not paragraph.strip():
                    continue
                    
                paragraph_tokens = self._count_tokens(paragraph)
                
                # If a single paragraph exceeds chunk size, split it
                if paragraph_tokens > self.chunk_size:
                    # Save current chunk if it exists
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, page_num))
                        current_chunk = ""
                        current_tokens = 0
                    
                    # Split the large paragraph into smaller chunks
                    sub_chunks = self._split_long_paragraph(paragraph, page_num)
                    chunks.extend(sub_chunks)
                    continue
                
                # Check if adding this paragraph would exceed chunk size
                if current_tokens + paragraph_tokens > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, page_num))
                    
                    # Start new chunk with this paragraph
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens
            
            # Don't forget the last chunk from this page
            if current_chunk:
                chunks.append(self._create_chunk(current_chunk, page_num))
        
        # Add overlap between consecutive chunks for context continuity
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Split a long paragraph into smaller chunks at sentence boundaries.
        
        Args:
            paragraph: Long paragraph text
            page_num: Page number
            
        Returns:
            List of chunk dictionaries
        """
        # Split by sentences (period, exclamation, question mark followed by space)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_tokens = self._count_tokens(sentence)
            
            # If sentence is too long, split by commas
            if sentence_tokens > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_num))
                    current_chunk = ""
                    current_tokens = 0
                
                # Split long sentence by commas
                parts = sentence.split(', ')
                for part in parts:
                    part_tokens = self._count_tokens(part)
                    if current_tokens + part_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk, page_num))
                        current_chunk = part
                        current_tokens = part_tokens
                    else:
                        if current_chunk:
                            current_chunk += ", " + part
                        else:
                            current_chunk = part
                        current_tokens += part_tokens
                
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_num))
                    current_chunk = ""
                    current_tokens = 0
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_num))
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_num))
        
        return chunks
    
    def _create_chunk(self, text: str, page_num: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.
        
        Args:
            text: Chunk text
            page_num: Page number
            
        Returns:
            Chunk dictionary with metadata
        """
        token_count = self._count_tokens(text)
        return {
            "text": text,
            "page_number": page_num,
            "token_count": token_count,
            "chunk_id": f"page_{page_num}_{hash(text) % 1000000}"  # Simple unique ID
        }
    
    def _add_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add overlap between consecutive chunks for better context continuity.
        
        Args:
            chunks: Original chunks
            
        Returns:
            Chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks
        
        # Calculate how many tokens to overlap (use smaller of overlap and 10% of chunk size)
        overlap_tokens = min(self.overlap, int(self.chunk_size * 0.1))
        
        for i in range(len(chunks) - 1):
            current_text = chunks[i]["text"]
            next_text = chunks[i + 1]["text"]
            
            # Get the last few tokens from current chunk
            tokens = self.tokenizer.encode(current_text)
            if len(tokens) > overlap_tokens:
                # Decode the last overlap_tokens tokens
                overlap_text = self.tokenizer.decode(tokens[-overlap_tokens:])
                # Prepend overlap to next chunk
                chunks[i + 1]["text"] = overlap_text + "\n\n" + next_text
                chunks[i + 1]["overlap_from_previous"] = True
        
        return chunks