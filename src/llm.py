from google import genai
from google.genai import types
from config.config import Config
from typing import List

class GeminiLLM:
    """Gemini 2.5 Flash LLM for generating responses"""
    
    def __init__(self):
        """Initialize Gemini LLM client"""
        Config.validate()
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model = Config.LLM_MODEL
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate response using query and retrieved context
        
        Args:
            query: User's question
            context: List of relevant text chunks from vector search
            
        Returns:
            Generated response string
        """
        # Prepare context
        context_text = "\n\n".join([f"Context {i+1}: {text}" for i, text in enumerate(context)])
        
        # Create prompt
        prompt = f"""Based on the following context information, please answer the question accurately and comprehensively.

Context Information:
{context_text}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to answer the question, please mention that."""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation with thinking disabled
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            final_response = response.candidates[0].content.parts[0].text
            return final_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    def generate_simple_response(self, text: str) -> str:
        """
        Generate a simple response without context (for testing)
        
        Args:
            text: Input text
            
        Returns:
            Generated response string
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=text),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            final_response = response.candidates[0].content.parts[0].text
            return final_response
            
        except Exception as e:
            print(f"Error generating simple response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"