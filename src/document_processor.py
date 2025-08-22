import re
from typing import List
from config.config import Config

class DocumentProcessor:
    """Handles document processing and text chunking"""
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        # Take last part of current chunk as overlap
                        overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    if len(sentence) > self.chunk_size:
                        # Split long sentence into smaller parts
                        words = sentence.split()
                        temp_chunk = ""
                        
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                        
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def process_document(self, text: str) -> List[str]:
        """
        Process a document by cleaning and chunking
        
        Args:
            text: Raw document text
            
        Returns:
            List of processed text chunks
        """
        chunks = self.chunk_text(text)
        print(f"Document processed into {len(chunks)} chunks")
        
        # Print chunk statistics
        if chunks:
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            max_length = max(len(chunk) for chunk in chunks)
            min_length = min(len(chunk) for chunk in chunks)
            
            print(f"Chunk statistics:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Max length: {max_length} characters")
            print(f"  Min length: {min_length} characters")
        
        return chunks