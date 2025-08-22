# Complete RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline implementation using Google's Gemini models and FAISS vector database.

## Features

- **🤖 Gemini 2.5 Flash LLM** - Advanced language model for response generation
- **🔍 Gemini Embedding Model** - High-quality text embeddings for semantic search
- **⚡ FAISS Vector Database** - Fast similarity search and retrieval
- **📄 Document Processing** - Intelligent text chunking and preprocessing
- **🔧 Modular Architecture** - Clean, maintainable, and extensible codebase
- **🎯 Interactive Demo** - Easy-to-use demonstration and testing interface

## Project Structure

```
Complete-RAG-pipeline/
├── README.md
├── requirements.txt
├── .env                          # Environment variables (API keys)
├── .gitignore
├── main.py                       # Main entry point
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration management
├── src/
│   ├── __init__.py
│   ├── embeddings.py            # Gemini embedding functions
│   ├── vector_store.py          # FAISS vector database operations
│   ├── llm.py                   # Gemini 2.5 Flash LLM functions
│   ├── document_processor.py    # Document chunking and processing
│   └── rag_pipeline.py          # Main RAG pipeline orchestrator
├── data/
│   ├── __init__.py
│   └── sample_story.py          # Sample story for testing
└── examples/
    ├── __init__.py
    └── demo.py                  # Comprehensive demo script
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amruth22/Complete-RAG-pipeline.git
   cd Complete-RAG-pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy the `.env` file and add your Google API key:
   ```bash
   # .env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Quick Start

### 1. Test the Pipeline
```bash
python main.py --test
```

### 2. Ingest Sample Document
```bash
python main.py --ingest
```

### 3. Ask a Question
```bash
python main.py --query "Who is Lyra Moonwhisper?"
```

### 4. Run Interactive Demo
```bash
python main.py --demo
```

### 5. Interactive Mode (Default)
```bash
python main.py
```

## Usage Examples

### Command Line Interface

```bash
# Test all components
python main.py --test

# Ingest sample document
python main.py --ingest

# Ask a single question
python main.py --query "What are the five Sacred Temples?"

# Show pipeline statistics
python main.py --stats

# Reset the pipeline
python main.py --reset

# Run comprehensive demo
python main.py --demo
```

### Python API

```python
from src.rag_pipeline import RAGPipeline
from data.sample_story import get_sample_story

# Initialize pipeline
rag = RAGPipeline()

# Test components
test_results = rag.test_components()

# Ingest documents
sample_story = get_sample_story()
stats = rag.ingest_documents([sample_story])

# Query the pipeline
result = rag.query("Who is the main character?")
print(result["response"])
```

## Configuration

The pipeline can be configured through `config/config.py`:

```python
class Config:
    # Model configurations
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"
    
    # RAG configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # FAISS configurations
    VECTOR_DIMENSION = 768
```

## Components

### 1. Embedding Generator (`src/embeddings.py`)
- Uses Gemini embedding model for text vectorization
- Supports batch and single text embedding generation
- Handles API communication and error management

### 2. Vector Store (`src/vector_store.py`)
- FAISS-based vector database for similarity search
- Supports saving/loading indexes to/from disk
- Implements cosine similarity search
- Provides statistics and management functions

### 3. LLM Interface (`src/llm.py`)
- Gemini 2.5 Flash integration for response generation
- Context-aware response generation
- Configurable thinking budget (disabled for faster responses)
- Error handling and fallback mechanisms

### 4. Document Processor (`src/document_processor.py`)
- Intelligent text chunking with overlap
- Text cleaning and normalization
- Sentence-aware splitting for better context preservation
- Configurable chunk sizes and overlap

### 5. RAG Pipeline (`src/rag_pipeline.py`)
- Main orchestrator combining all components
- Document ingestion and indexing workflow
- Query processing and response generation
- Pipeline state management and statistics

## Sample Data

The pipeline includes a sample fantasy story "The Chronicles of Eldoria: The Last Guardian" for testing and demonstration purposes. The story contains rich narrative content perfect for testing RAG capabilities.

## Demo Features

The interactive demo (`examples/demo.py`) includes:

- **Component Testing** - Verify all components work correctly
- **Document Ingestion** - Process and index the sample story
- **Predefined Queries** - Test with curated questions
- **Interactive Mode** - Ask your own questions
- **Statistics Display** - View pipeline performance metrics

## API Key Setup

1. Get your Google API key from [Google AI Studio](https://aistudio.google.com/)
2. Add it to the `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Performance

- **Embedding Dimension**: 768 (Gemini embedding model)
- **Default Chunk Size**: 1000 characters
- **Default Overlap**: 200 characters
- **Default Top-K**: 5 results
- **Vector Search**: Cosine similarity with FAISS

## Error Handling

The pipeline includes comprehensive error handling:
- API key validation
- Network error recovery
- Malformed input handling
- Graceful degradation when components fail

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Google API key is correctly set in the `.env` file
2. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
3. **Memory Issues**: Reduce chunk size or batch size for large documents
4. **FAISS Errors**: Ensure you have the correct FAISS version installed

### Getting Help

- Check the demo script for usage examples
- Run `python main.py --test` to verify component functionality
- Use `python main.py --stats` to check pipeline status

## Roadmap

- [ ] Support for multiple document formats (PDF, DOCX, etc.)
- [ ] Advanced chunking strategies
- [ ] Multiple vector store backends
- [ ] Web interface
- [ ] Batch processing capabilities
- [ ] Performance optimization
- [ ] Multi-language support

---

**Built with ❤️ using Google Gemini and FAISS**