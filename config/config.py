import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for RAG pipeline"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model configurations
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"
    
    # RAG configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # FAISS configurations
    VECTOR_DIMENSION = 3072  # Gemini embedding dimension (actual dimension from Gemini model)
    FAISS_INDEX_PATH = "faiss_index"
    
    @classmethod
    def validate(cls):
        """Validate that all required configurations are set"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return True