# Medical RAG System

A production-ready Retrieval Augmented Generation (RAG) system for answering medical questions using verified documentation sources with citation support.

## Overview

This system combines vector similarity search with Large Language Models to provide accurate, source-cited answers to medical questions. It retrieves relevant medical documentation, builds context, and generates responses grounded in the retrieved information.

## Features

- **Smart Document Chunking**: Respects sentence boundaries for better context preservation
- **Vector Store**: ChromaDB-based persistent storage with cosine similarity search
- **Multi-Provider LLM Support**: Works with Groq API (cloud) or Ollama (local)
- **Citation Support**: All answers include source citations
- **Medical Domain Optimization**: Configurable for biomedical embedding models
- **Confidence Scoring**: Provides confidence scores based on retrieval quality

## Project Structure

```
medical-rag/
├── main.py                     # Entry point - run demo
├── requirements.txt            # Dependencies
├── medical_rag/
│   ├── __init__.py            # Package exports
│   ├── models.py              # Data classes (Document, SearchResult, RAGResponse)
│   ├── embeddings.py          # Medical text embeddings
│   ├── processor.py           # Document chunking
│   ├── vector_store.py        # ChromaDB vector database
│   ├── llm_client.py          # Multi-provider LLM client
│   ├── rag.py                 # Main RAG orchestrator
│   └── data/
│       ├── __init__.py
│       └── sample_docs.py     # Sample medical documents
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Documents  │────▶│ Document Chunker │────▶│  Embeddings │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│    Query    │────▶│ Query Embedding  │────▶│ Vector Store│
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
                    ┌──────────────────┐            │
                    │   LLM Generation │◀───────────┘
                    │  (with context)  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Cited Response   │
                    └──────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/fergim92/medical-rag.git
cd medical-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Using Groq API (Recommended)

1. Get a free API key from [Groq Console](https://console.groq.com)
2. Set the environment variable:
```bash
export GROQ_API_KEY="your-api-key-here"
```

### Using Ollama (Local)

1. Install [Ollama](https://ollama.ai)
2. Pull the model:
```bash
ollama pull llama3.1
```
3. Initialize with Ollama provider:
```python
from medical_rag import MedicalRAG
rag = MedicalRAG(llm_provider="ollama")
```

## Usage

### Running the Demo

```bash
python main.py
```

### Basic Usage

```python
from medical_rag import MedicalRAG

# Initialize the system
rag = MedicalRAG()

# Ingest documents
documents = [
    {
        "title": "Diabetes Overview",
        "source": "Medical Encyclopedia",
        "content": "Diabetes mellitus is a metabolic disease..."
    }
]
rag.ingest_documents(documents)

# Query the system
response = rag.query("What are the symptoms of diabetes?")

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2%}")
print(f"Sources: {[s.title for s in response.sources]}")
```

### Advanced Configuration

```python
from medical_rag import MedicalRAG

rag = MedicalRAG(
    embedding_model="pritamdeka/S-PubMedBert-MS-MARCO",  # Medical-optimized
    llm_provider="groq",
    persist_dir="./my_vector_db"
)
```

## Components

| Module | Class | Description |
|--------|-------|-------------|
| `models.py` | `Document` | Medical document with metadata |
| `models.py` | `SearchResult` | Search result with relevance score |
| `models.py` | `RAGResponse` | Structured response with citations |
| `embeddings.py` | `MedicalEmbeddings` | Vector embeddings generator |
| `processor.py` | `DocumentProcessor` | Smart document chunking |
| `vector_store.py` | `VectorStore` | ChromaDB wrapper |
| `llm_client.py` | `LLMClient` | Multi-provider LLM interface |
| `rag.py` | `MedicalRAG` | Main orchestrator |

## Embedding Models

| Model | Description |
|-------|-------------|
| `all-MiniLM-L6-v2` | Fast, general purpose (default) |
| `pritamdeka/S-PubMedBert-MS-MARCO` | Optimized for biomedical text |

## Response Structure

```python
@dataclass
class RAGResponse:
    answer: str              # Generated answer with citations
    sources: List[Document]  # Retrieved source documents
    confidence: float        # Confidence score (0-1)
    query: str              # Original query
```

## Safety Guidelines

The system follows strict medical information guidelines:

1. Answers are based only on provided documentation
2. All responses include source citations
3. Users are reminded to consult healthcare professionals
4. System explicitly states when information is not available

## Requirements

- Python 3.8+
- ChromaDB 0.4.0+
- sentence-transformers 2.2.0+
- NLTK 3.8.0+
- NumPy 1.24.0+

## License

MIT License

## Author

Fernando Gimenez
