"""
Medical RAG System - Retrieval Augmented Generation for Healthcare
===================================================================
A production-ready RAG system that answers medical questions using
verified documentation sources with citation support.
"""

from .models import Document, SearchResult, RAGResponse
from .embeddings import MedicalEmbeddings
from .processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_client import LLMClient
from .rag import MedicalRAG

__version__ = "1.0.0"
__author__ = "Fernando Gimenez"

__all__ = [
    "Document",
    "SearchResult",
    "RAGResponse",
    "MedicalEmbeddings",
    "DocumentProcessor",
    "VectorStore",
    "LLMClient",
    "MedicalRAG",
]
