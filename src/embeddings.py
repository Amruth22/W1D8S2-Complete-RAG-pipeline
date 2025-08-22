from google import genai
from config.config import Config
import numpy as np
from typing import List, Union

class EmbeddingGenerator:
    """Handles text embedding generation using Gemini embedding model"""
    
    def __init__(self):
        """Initialize the embedding generator with Gemini client"""
        Config.validate()
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model = Config.EMBEDDING_MODEL
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for given text(s)
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts
            )
            
            # Convert embeddings to numpy array
            embeddings = []
            for embedding in result.embeddings:
                embeddings.append(embedding.values)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of single embedding
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]