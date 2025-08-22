#!/usr/bin/env python3
"""
Demo script for the RAG Pipeline
This script demonstrates how to use the complete RAG pipeline
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from data.sample_story import get_sample_story

def main():
    """Main demo function"""
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)
    
    # Initialize the RAG pipeline
    rag = RAGPipeline()
    
    # Test all components first
    print("\n1. Testing Pipeline Components...")
    test_results = rag.test_components()
    
    all_tests_passed = all(test_results.values())
    if not all_tests_passed:
        print("❌ Some component tests failed:")
        for component, passed in test_results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"   {status} {component}")
        print("\nPlease check your configuration and try again.")
        return
    
    print("[PASS] All component tests passed!")
    
    # Try to load existing index
    print("\n2. Loading Existing Index...")
    if not rag.load_existing_index():
        print("No existing index found. Creating new index...")
        
        # Get sample story
        print("\n3. Ingesting Sample Document...")
        sample_story = get_sample_story()
        
        # Ingest the document
        ingestion_stats = rag.ingest_documents([sample_story])
        
        print("\nIngestion Statistics:")
        print(f"  Documents processed: {ingestion_stats['total_documents']}")
        print(f"  Text chunks created: {ingestion_stats['total_chunks']}")
        print(f"  Embeddings generated: {ingestion_stats['total_embeddings']}")
    else:
        print("[PASS] Existing index loaded successfully!")
    
    # Show pipeline statistics
    print("\n4. Pipeline Statistics:")
    stats = rag.get_pipeline_stats()
    print(f"  Indexed: {stats['is_indexed']}")
    print(f"  Total embeddings: {stats['vector_store_stats']['total_embeddings']}")
    print(f"  Vector dimension: {stats['vector_store_stats']['dimension']}")
    print(f"  Chunk size: {stats['config']['chunk_size']}")
    print(f"  Top-K results: {stats['config']['top_k_results']}")
    
    # Demo queries
    print("\n5. Demo Queries:")
    print("=" * 40)
    
    demo_questions = [
        "Who is Lyra Moonwhisper?",
        "What are the five Sacred Temples?",
        "What is the Void Blight?",
        "How did Lyra become a Guardian?",
        "What happened in the final battle?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\nQuery {i}: {question}")
        print("-" * 50)
        
        result = rag.query(question)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print("Response:")
        print(result["response"])
        
        print(f"\nContext used ({result['num_context_chunks']} chunks):")
        for j, (context, score) in enumerate(zip(result["context"][:2], result["similarity_scores"][:2])):
            print(f"  {j+1}. (Score: {score:.3f}) {context[:100]}...")
        
        print("\n" + "=" * 40)
    
    # Interactive mode
    print("\n6. Interactive Mode")
    print("You can now ask your own questions about the story!")
    print("Type 'quit' to exit, 'stats' to see pipeline statistics")
    print("-" * 50)
    
    while True:
        try:
            user_question = input("\nYour question: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_question.lower() == 'stats':
                stats = rag.get_pipeline_stats()
                print(f"\nPipeline Statistics:")
                print(f"  Total embeddings: {stats['vector_store_stats']['total_embeddings']}")
                print(f"  Total texts: {stats['vector_store_stats']['total_texts']}")
                continue
            
            if not user_question:
                print("Please enter a question.")
                continue
            
            print("\nProcessing your question...")
            result = rag.query(user_question)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print("\nResponse:")
            print(result["response"])
            
            # Show similarity scores
            if result["similarity_scores"]:
                print(f"\nRelevance scores: {[f'{score:.3f}' for score in result['similarity_scores'][:3]]}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
