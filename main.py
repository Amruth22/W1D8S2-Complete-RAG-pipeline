#!/usr/bin/env python3
"""
Main entry point for the RAG Pipeline - Automatic Startup
"""

import sys
from src.rag_pipeline import RAGPipeline
from data.sample_story import get_sample_story

def main():
    """Main function - automatic startup with complete RAG pipeline"""
    print("🤖 RAG Pipeline - Starting Automatically...")
    print("=" * 50)
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Step 1: Test all components
        print("\n1. 🔧 Testing all components...")
        test_results = rag.test_components()
        
        if not all(test_results.values()):
            print("❌ Component tests failed:")
            for component, passed in test_results.items():
                status = "[PASS]" if passed else "[FAIL]"
                print(f"   {status} {component}")
            print("\n⚠️  Please check your configuration and try again.")
            return 1
        
        print("✅ All components working perfectly!")
        
        # Step 2: Load existing index or ingest documents
        print("\n2. 📄 Preparing document index...")
        if not rag.load_existing_index():
            print("📚 Ingesting sample document 'The Chronicles of Eldoria'...")
            sample_story = get_sample_story()
            stats = rag.ingest_documents([sample_story])
            print(f"✅ Successfully processed {stats['total_chunks']} text chunks")
            print(f"🎯 Generated {stats['total_embeddings']} embeddings")
        else:
            print("✅ Existing document index loaded successfully!")
        
        # Step 3: Show pipeline statistics
        stats = rag.get_pipeline_stats()
        print(f"\n3. 📊 Pipeline Ready!")
        print(f"   Total embeddings: {stats['vector_store_stats']['total_embeddings']}")
        print(f"   Vector dimension: {stats['vector_store_stats']['dimension']}")
        print(f"   Model: {stats['config']['llm_model']}")
        
        # Step 4: Start interactive Q&A
        print(f"\n4. 🎯 Interactive Q&A Mode")
        print("Ask questions about 'The Chronicles of Eldoria: The Last Guardian'!")
        print("Examples: 'Who is Lyra?', 'What are the Sacred Temples?', 'Tell me about the Void Blight'")
        print("\nCommands: 'stats' (show statistics), 'quit' (exit)")
        print("-" * 70)
        
        # Interactive Q&A loop
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                # Handle exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using the RAG Pipeline! Goodbye!")
                    break
                
                # Handle stats command
                if question.lower() == 'stats':
                    current_stats = rag.get_pipeline_stats()
                    print(f"\n📊 Pipeline Statistics:")
                    print(f"   Status: {'✅ Indexed' if current_stats['is_indexed'] else '❌ Not indexed'}")
                    print(f"   Embeddings: {current_stats['vector_store_stats']['total_embeddings']}")
                    print(f"   Texts: {current_stats['vector_store_stats']['total_texts']}")
                    continue
                
                # Skip empty questions
                if not question:
                    print("Please enter a question!")
                    continue
                
                # Process the question
                print("🔍 Searching for relevant information...")
                result = rag.query(question)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                print(f"\n📝 Answer:")
                print(result["response"])
                
                # Show context information
                if result["similarity_scores"]:
                    print(f"\n📈 Found {result['num_context_chunks']} relevant chunks")
                    print(f"   Top similarity scores: {[f'{score:.3f}' for score in result['similarity_scores'][:3]]}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again or type 'quit' to exit.")
        
        return 0
        
    except Exception as e:
        print(f"💥 Failed to start RAG Pipeline: {e}")
        print("Please check your environment configuration (.env file with GOOGLE_API_KEY)")
        return 1

if __name__ == "__main__":
    main()
