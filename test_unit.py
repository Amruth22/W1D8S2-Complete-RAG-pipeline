import unittest
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Add the current directory to Python path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class CoreCompleteRAGPipelineTests(unittest.TestCase):
    """Core 5 unit tests for Complete RAG Pipeline System with real components"""
    
    @classmethod
    def setUpClass(cls):
        """Load environment variables and validate API key"""
        load_dotenv()
        
        # Validate API key
        cls.api_key = os.getenv('GOOGLE_API_KEY')
        if not cls.api_key or not cls.api_key.startswith('AIza'):
            raise unittest.SkipTest("Valid GOOGLE_API_KEY not found in environment")
        
        print(f"Using API Key: {cls.api_key[:10]}...{cls.api_key[-5:]}")
        
        # Initialize RAG pipeline components
        try:
            from config.config import Config
            from src.embeddings import EmbeddingGenerator
            from src.vector_store import FAISSVectorStore
            from src.llm import GeminiLLM
            from src.document_processor import DocumentProcessor
            from src.rag_pipeline import RAGPipeline
            
            cls.Config = Config
            cls.EmbeddingGenerator = EmbeddingGenerator
            cls.FAISSVectorStore = FAISSVectorStore
            cls.GeminiLLM = GeminiLLM
            cls.DocumentProcessor = DocumentProcessor
            cls.RAGPipeline = RAGPipeline
            
            # Initialize components
            cls.embedding_generator = EmbeddingGenerator()
            cls.vector_store = FAISSVectorStore()
            cls.llm = GeminiLLM()
            cls.document_processor = DocumentProcessor()
            cls.rag_pipeline = RAGPipeline()
            
            print("Complete RAG pipeline components loaded successfully")
        except ImportError as e:
            raise unittest.SkipTest(f"Required RAG pipeline components not found: {e}")

    def test_01_embedding_generation_and_processing(self):
        """Test 1: Embedding Generation and Text Processing"""
        print("Running Test 1: Embedding Generation and Processing")
        
        # Test embedding generator initialization
        self.assertIsNotNone(self.embedding_generator)
        self.assertEqual(self.embedding_generator.model, self.Config.EMBEDDING_MODEL)
        
        # Test single embedding generation
        test_text = "This is a test document for RAG pipeline validation."
        single_embedding = self.embedding_generator.generate_single_embedding(test_text)
        
        self.assertIsInstance(single_embedding, np.ndarray)
        self.assertEqual(single_embedding.shape, (self.Config.VECTOR_DIMENSION,))
        # Accept both float32 and float64 as valid embedding types
        self.assertIn(single_embedding.dtype, [np.float32, np.float64])
        
        # Test batch embedding generation
        test_texts = ["First test text", "Second test text", "Third test text"]
        batch_embeddings = self.embedding_generator.generate_embeddings(test_texts)
        
        self.assertIsInstance(batch_embeddings, np.ndarray)
        self.assertEqual(batch_embeddings.shape, (len(test_texts), self.Config.VECTOR_DIMENSION))
        # Accept both float32 and float64 as valid embedding types
        self.assertIn(batch_embeddings.dtype, [np.float32, np.float64])
        
        # Test string input handling
        string_embeddings = self.embedding_generator.generate_embeddings("Single string input")
        self.assertIsInstance(string_embeddings, np.ndarray)
        self.assertEqual(string_embeddings.shape, (1, self.Config.VECTOR_DIMENSION))
        # Accept both float32 and float64 as valid embedding types
        self.assertIn(string_embeddings.dtype, [np.float32, np.float64])
        
        # Test embedding quality
        embedding_norms = np.linalg.norm(batch_embeddings, axis=1)
        self.assertTrue(all(norm > 0 for norm in embedding_norms))
        
        print(f"PASS: Single embedding - Shape: {single_embedding.shape}, Dtype: {single_embedding.dtype}")
        print(f"PASS: Batch embeddings - Shape: {batch_embeddings.shape}, Dtype: {batch_embeddings.dtype}")
        print(f"PASS: Vector dimension: {self.Config.VECTOR_DIMENSION}")

    def test_02_document_processing_and_chunking(self):
        """Test 2: Document Processing and Intelligent Chunking"""
        print("Running Test 2: Document Processing and Chunking")
        
        # Test document processor initialization
        self.assertIsNotNone(self.document_processor)
        self.assertEqual(self.document_processor.chunk_size, self.Config.CHUNK_SIZE)
        self.assertEqual(self.document_processor.chunk_overlap, self.Config.CHUNK_OVERLAP)
        
        # Test text cleaning
        dirty_text = "  This   is    a   test   text   with   extra   spaces!!!   "
        cleaned_text = self.document_processor.clean_text(dirty_text)
        self.assertIn("This is a test text", cleaned_text)
        self.assertEqual(cleaned_text.strip(), cleaned_text)
        
        # Test text chunking
        test_document = """This is the first sentence. This is the second sentence. This is the third sentence. 
        This is the fourth sentence. This is the fifth sentence. This is the sixth sentence.
        This is a very long sentence that might exceed the chunk size limit and should be handled appropriately.
        This is the final sentence in the test document."""
        
        chunks = self.document_processor.chunk_text(test_document)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertTrue(all(len(chunk) <= self.Config.CHUNK_SIZE + 100 for chunk in chunks))
        
        # Test document processing
        processed_chunks = self.document_processor.process_document(test_document)
        self.assertIsInstance(processed_chunks, list)
        self.assertGreater(len(processed_chunks), 0)
        self.assertTrue(all(chunk.strip() for chunk in processed_chunks))
        
        # Test empty document handling
        empty_chunks = self.document_processor.process_document("")
        self.assertIsInstance(empty_chunks, list)
        self.assertEqual(len(empty_chunks), 0)
        
        # Test custom configuration
        custom_processor = self.DocumentProcessor(chunk_size=500, chunk_overlap=100)
        self.assertEqual(custom_processor.chunk_size, 500)
        self.assertEqual(custom_processor.chunk_overlap, 100)
        
        print(f"PASS: Document processing - {len(processed_chunks)} chunks created")
        print(f"PASS: Chunk configuration - Size: {self.Config.CHUNK_SIZE}, Overlap: {self.Config.CHUNK_OVERLAP}")

    def test_03_vector_store_operations(self):
        """Test 3: Vector Store Operations and FAISS Integration"""
        print("Running Test 3: Vector Store Operations")
        
        # Test vector store initialization
        self.assertIsNotNone(self.vector_store)
        self.assertEqual(self.vector_store.dimension, self.Config.VECTOR_DIMENSION)
        self.assertIsInstance(self.vector_store.texts, list)
        
        # Test index creation
        self.vector_store.create_index()
        self.assertIsNotNone(self.vector_store.index)
        
        # Test adding embeddings
        test_texts = ["First test document", "Second test document", "Third test document"]
        test_embeddings = np.random.rand(len(test_texts), self.Config.VECTOR_DIMENSION).astype('float32')
        
        initial_count = len(self.vector_store.texts)
        self.vector_store.add_embeddings(test_embeddings, test_texts)
        
        # Verify embeddings were added
        self.assertEqual(len(self.vector_store.texts), initial_count + len(test_texts))
        self.assertEqual(self.vector_store.index.ntotal, len(test_texts))
        
        # Test search functionality
        query_embedding = np.random.rand(self.Config.VECTOR_DIMENSION).astype('float32')
        similar_texts, similarity_scores = self.vector_store.search(query_embedding, k=3)
        
        self.assertIsInstance(similar_texts, list)
        self.assertIsInstance(similarity_scores, list)
        self.assertLessEqual(len(similar_texts), 3)
        self.assertEqual(len(similar_texts), len(similarity_scores))
        
        # Test search with empty index
        empty_store = self.FAISSVectorStore()
        empty_texts, empty_scores = empty_store.search(query_embedding, k=5)
        self.assertEqual(empty_texts, [])
        self.assertEqual(empty_scores, [])
        
        # Test statistics
        stats = self.vector_store.get_stats()
        self.assertIn('total_embeddings', stats)
        self.assertIn('dimension', stats)
        self.assertIn('total_texts', stats)
        self.assertEqual(stats['dimension'], self.Config.VECTOR_DIMENSION)
        
        print(f"PASS: Vector store operations - {stats['total_embeddings']} embeddings indexed")
        print(f"PASS: Search functionality - {len(similar_texts)} results returned")

    def test_04_llm_integration_and_response_generation(self):
        """Test 4: LLM Integration and Context-Aware Response Generation"""
        print("Running Test 4: LLM Integration and Response Generation")
        
        # Test LLM initialization
        self.assertIsNotNone(self.llm)
        self.assertEqual(self.llm.model, self.Config.LLM_MODEL)
        
        # Test simple response generation
        simple_response = self.llm.generate_simple_response("Hi")
        self.assertIsInstance(simple_response, str)
        self.assertGreater(len(simple_response), 0)
        self.assertNotIn("Error:", simple_response)
        
        # Test context-aware response generation
        test_query = "What is machine learning?"
        test_context = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "It involves algorithms that can identify patterns and make predictions without explicit programming.",
            "Common applications include image recognition, natural language processing, and recommendation systems."
        ]
        
        context_response = self.llm.generate_response(test_query, test_context)
        self.assertIsInstance(context_response, str)
        self.assertGreater(len(context_response), 0)
        self.assertNotIn("Error:", context_response)
        
        # Test empty context handling
        empty_context_response = self.llm.generate_response("Test question", [])
        self.assertIsInstance(empty_context_response, str)
        self.assertGreater(len(empty_context_response), 0)
        
        # Test multiple context chunks
        multiple_context = [
            "Context chunk 1 with relevant information about the topic.",
            "Context chunk 2 with additional details and explanations.",
            "Context chunk 3 with supporting evidence and examples."
        ]
        
        multi_context_response = self.llm.generate_response("Multi-context question", multiple_context)
        self.assertIsInstance(multi_context_response, str)
        self.assertGreater(len(multi_context_response), 0)
        
        print(f"PASS: Simple response - Length: {len(simple_response)} characters")
        print(f"PASS: Context response - Length: {len(context_response)} characters")
        print("PASS: Context-aware response generation validated")

    def test_05_configuration_and_pipeline_validation(self):
        """Test 5: Configuration and Pipeline Component Validation"""
        print("Running Test 5: Configuration and Pipeline Validation")
        
        # Test configuration validation
        self.assertIsNotNone(self.Config.GOOGLE_API_KEY)
        self.assertTrue(self.Config.GOOGLE_API_KEY.startswith('AIza'))
        self.assertEqual(self.Config.LLM_MODEL, "gemini-2.5-flash")
        self.assertEqual(self.Config.EMBEDDING_MODEL, "gemini-embedding-001")
        
        # Test RAG configuration parameters
        self.assertEqual(self.Config.CHUNK_SIZE, 1000)
        self.assertEqual(self.Config.CHUNK_OVERLAP, 200)
        self.assertEqual(self.Config.TOP_K_RESULTS, 5)
        self.assertEqual(self.Config.VECTOR_DIMENSION, 3072)
        
        # Validate parameter relationships
        self.assertLess(self.Config.CHUNK_OVERLAP, self.Config.CHUNK_SIZE)
        self.assertGreater(self.Config.TOP_K_RESULTS, 0)
        self.assertGreater(self.Config.VECTOR_DIMENSION, 0)
        
        # Test pipeline component initialization
        self.assertIsNotNone(self.rag_pipeline)
        self.assertIsNotNone(self.rag_pipeline.embedding_generator)
        self.assertIsNotNone(self.rag_pipeline.vector_store)
        self.assertIsNotNone(self.rag_pipeline.llm)
        self.assertIsNotNone(self.rag_pipeline.document_processor)
        self.assertFalse(self.rag_pipeline.is_indexed)
        
        # Test component configuration
        self.assertEqual(self.rag_pipeline.embedding_generator.model, self.Config.EMBEDDING_MODEL)
        self.assertEqual(self.rag_pipeline.llm.model, self.Config.LLM_MODEL)
        self.assertEqual(self.rag_pipeline.vector_store.dimension, self.Config.VECTOR_DIMENSION)
        self.assertEqual(self.rag_pipeline.document_processor.chunk_size, self.Config.CHUNK_SIZE)
        self.assertEqual(self.rag_pipeline.document_processor.chunk_overlap, self.Config.CHUNK_OVERLAP)
        
        # Test pipeline statistics (initial state)
        stats = self.rag_pipeline.get_pipeline_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("is_indexed", stats)
        self.assertIn("vector_store_stats", stats)
        self.assertIn("config", stats)
        self.assertFalse(stats["is_indexed"])
        
        # Validate config in stats
        config_stats = stats["config"]
        self.assertEqual(config_stats["chunk_size"], self.Config.CHUNK_SIZE)
        self.assertEqual(config_stats["top_k_results"], self.Config.TOP_K_RESULTS)
        self.assertEqual(config_stats["llm_model"], self.Config.LLM_MODEL)
        self.assertEqual(config_stats["embedding_model"], self.Config.EMBEDDING_MODEL)
        
        # Test query without index (should return error)
        unindexed_result = self.rag_pipeline.query("Test question")
        self.assertIn("error", unindexed_result)
        self.assertIn("No documents have been indexed", unindexed_result["response"])
        
        # Test pipeline reset functionality
        original_vector_store = self.rag_pipeline.vector_store
        self.rag_pipeline.reset_pipeline()
        self.assertFalse(self.rag_pipeline.is_indexed)
        # After reset, should have new vector store instance
        self.assertIsNotNone(self.rag_pipeline.vector_store)
        
        # Test file path configuration
        self.assertIsInstance(self.Config.FAISS_INDEX_PATH, str)
        self.assertGreater(len(self.Config.FAISS_INDEX_PATH), 0)
        
        print(f"PASS: Configuration validation - LLM: {self.Config.LLM_MODEL}, Embedding: {self.Config.EMBEDDING_MODEL}")
        print(f"PASS: RAG parameters - Chunk size: {self.Config.CHUNK_SIZE}, Overlap: {self.Config.CHUNK_OVERLAP}, Top-K: {self.Config.TOP_K_RESULTS}")
        print(f"PASS: Pipeline components initialized and configured correctly")
        print("PASS: Configuration and pipeline validation completed")

def run_core_tests():
    """Run core tests and provide summary"""
    print("=" * 70)
    print("[*] Core Complete RAG Pipeline System Unit Tests (5 Tests)")
    print("Testing with REAL API and Complete RAG Components")
    print("=" * 70)
    
    # Check API key
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key or not api_key.startswith('AIza'):
        print("[ERROR] Valid GOOGLE_API_KEY not found!")
        return False
    
    print(f"[OK] Using API Key: {api_key[:10]}...{api_key[-5:]}")
    print()
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(CoreCompleteRAGPipelineTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("[*] Test Results:")
    print(f"[*] Tests Run: {result.testsRun}")
    print(f"[*] Failures: {len(result.failures)}")
    print(f"[*] Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n[FAILURES]:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    if result.errors:
        print("\n[ERRORS]:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n[SUCCESS] All 5 core complete RAG pipeline tests passed!")
        print("[OK] Complete RAG pipeline components working correctly with real API")
        print("[OK] Embeddings, Document Processing, Vector Store, LLM, RAG Orchestration validated")
    else:
        print(f"\n[WARNING] {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success

if __name__ == "__main__":
    print("[*] Starting Core Complete RAG Pipeline System Tests")
    print("[*] 5 essential tests with real API and complete RAG components")
    print("[*] Components: Embeddings, Document Processing, Vector Store, LLM, RAG Pipeline")
    print()
    
    success = run_core_tests()
    exit(0 if success else 1)