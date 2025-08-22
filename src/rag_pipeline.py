from typing import List, Dict, Any
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from src.llm import GeminiLLM
from src.document_processor import DocumentProcessor
from config.config import Config

class RAGPipeline:
    """Main RAG Pipeline orchestrator"""
    
    def __init__(self):
        """Initialize all components of the RAG pipeline"""
        print("Initializing RAG Pipeline...")
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        self.llm = GeminiLLM()
        self.document_processor = DocumentProcessor()
        
        # Pipeline state
        self.is_indexed = False
        
        print("RAG Pipeline initialized successfully!")
    
    def ingest_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the RAG pipeline
        
        Args:
            documents: List of document texts to ingest
            
        Returns:
            Dictionary with ingestion statistics
        """
        print(f"Starting document ingestion for {len(documents)} documents...")
        
        all_chunks = []
        total_chunks = 0
        
        # Process each document
        for i, document in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}...")
            chunks = self.document_processor.process_document(document)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
        
        print(f"Total chunks created: {total_chunks}")
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        
        # Add to vector store
        print("Adding embeddings to vector store...")
        self.vector_store.add_embeddings(embeddings, all_chunks)
        
        # Save the index
        self.vector_store.save_index()
        
        self.is_indexed = True
        
        stats = {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "total_embeddings": len(embeddings),
            "vector_store_stats": self.vector_store.get_stats()
        }
        
        print("Document ingestion completed!")
        return stats
    
    def load_existing_index(self) -> bool:
        """
        Load existing vector index from disk
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        print("Attempting to load existing index...")
        success = self.vector_store.load_index()
        if success:
            self.is_indexed = True
            print("Existing index loaded successfully!")
        else:
            print("No existing index found.")
        return success
    
    def query(self, question: str, top_k: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        """
        Query the RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of top similar chunks to retrieve
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not self.is_indexed:
            return {
                "response": "Error: No documents have been indexed yet. Please ingest documents first.",
                "context": [],
                "similarity_scores": [],
                "error": "No index available"
            }
        
        print(f"Processing query: {question}")
        
        # Generate embedding for the query
        print("Generating query embedding...")
        query_embedding = self.embedding_generator.generate_single_embedding(question)
        
        # Search for similar chunks
        print("Searching for relevant context...")
        similar_texts, similarity_scores = self.vector_store.search(query_embedding, top_k)
        
        if not similar_texts:
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "context": [],
                "similarity_scores": [],
                "error": "No relevant context found"
            }
        
        print(f"Found {len(similar_texts)} relevant chunks")
        
        # Generate response using LLM
        print("Generating response...")
        response = self.llm.generate_response(question, similar_texts)
        
        result = {
            "response": response,
            "context": similar_texts,
            "similarity_scores": similarity_scores,
            "num_context_chunks": len(similar_texts)
        }
        
        print("Query processed successfully!")
        return result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current pipeline state
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "is_indexed": self.is_indexed,
            "vector_store_stats": self.vector_store.get_stats(),
            "config": {
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "top_k_results": Config.TOP_K_RESULTS,
                "llm_model": Config.LLM_MODEL,
                "embedding_model": Config.EMBEDDING_MODEL
            }
        }
        return stats
    
    def reset_pipeline(self):
        """Reset the pipeline by clearing the vector store"""
        print("Resetting pipeline...")
        self.vector_store = FAISSVectorStore()
        self.is_indexed = False
        print("Pipeline reset completed!")
    
    def test_components(self) -> Dict[str, bool]:
        """
        Test all pipeline components
        
        Returns:
            Dictionary with test results for each component
        """
        print("Testing pipeline components...")
        
        results = {}
        
        # Test embedding generator
        try:
            test_embedding = self.embedding_generator.generate_single_embedding("Test text")
            results["embedding_generator"] = len(test_embedding) == Config.VECTOR_DIMENSION
            print("[PASS] Embedding generator test passed")
        except Exception as e:
            results["embedding_generator"] = False
            print(f"[FAIL] Embedding generator test failed: {e}")
        
        # Test LLM
        try:
            test_response = self.llm.generate_simple_response("Hello, how are you?")
            results["llm"] = len(test_response) > 0
            print("[PASS] LLM test passed")
        except Exception as e:
            results["llm"] = False
            print(f"[FAIL] LLM test failed: {e}")
        
        # Test document processor
        try:
            test_chunks = self.document_processor.process_document("This is a test document. It has multiple sentences. Each sentence should be processed correctly.")
            results["document_processor"] = len(test_chunks) > 0
            print("[PASS] Document processor test passed")
        except Exception as e:
            results["document_processor"] = False
            print(f"[FAIL] Document processor test failed: {e}")
        
        # Test vector store
        try:
            self.vector_store.create_index()
            results["vector_store"] = self.vector_store.index is not None
            print("[PASS] Vector store test passed")
        except Exception as e:
            results["vector_store"] = False
            print(f"[FAIL] Vector store test failed: {e}")
        
        print("Component testing completed!")
        return results
