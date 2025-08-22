#!/usr/bin/env python3
"""
Simple Unit Test for Complete RAG Pipeline
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from src.rag_pipeline import RAGPipeline
from data.sample_story import get_sample_story


def test_complete_pipeline():
    """Test the complete RAG pipeline workflow"""
    try:
        # Initialize pipeline
        print("Initializing RAG Pipeline...")
        rag = RAGPipeline()
        
        # Test 1: Environment and Configuration
        print("1. Testing environment configuration...")
        Config.validate()
        if not Config.GOOGLE_API_KEY:
            print("FAIL: API key not found")
            return False
        print("PASS: Environment configuration")
        
        # Test 2: Component functionality
        print("2. Testing individual components...")
        test_results = rag.test_components()
        if not all(test_results.values()):
            print("FAIL: Some components failed")
            return False
        print("PASS: All components working")
        
        # Test 3: Document ingestion
        print("3. Testing document ingestion...")
        sample_story = get_sample_story()
        stats = rag.ingest_documents([sample_story])
        if stats["total_documents"] == 0 or stats["total_chunks"] == 0:
            print("FAIL: Document ingestion failed")
            return False
        print(f"PASS: Ingested {stats['total_chunks']} chunks")
        
        # Test 4: Query processing
        print("4. Testing query processing...")
        result = rag.query("Who is Lyra Moonwhisper?")
        if "response" not in result or not result["response"]:
            print("FAIL: Query processing failed")
            return False
        print("PASS: Query processing successful")
        
        # Test 5: Pipeline statistics
        print("5. Testing pipeline statistics...")
        stats = rag.get_pipeline_stats()
        if not stats["is_indexed"] or stats["vector_store_stats"]["total_embeddings"] == 0:
            print("FAIL: Pipeline statistics check failed")
            return False
        print("PASS: Pipeline statistics valid")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Exception occurred - {e}")
        return False


def main():
    """Main test function"""
    print("RAG Pipeline Complete Test")
    print("=" * 30)
    
    success = test_complete_pipeline()
    
    print("\n" + "=" * 30)
    if success:
        print("RESULT: ALL TESTS PASSED")
        print("RAG Pipeline is working correctly!")
        return 0
    else:
        print("RESULT: TESTS FAILED")
        print("RAG Pipeline has issues!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
