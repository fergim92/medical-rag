"""
Data models for the Medical RAG system.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """Represents a medical document with metadata."""
    id: str
    content: str
    source: str
    title: str
    chunk_index: int = 0


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    document: Document
    score: float


@dataclass
class RAGResponse:
    """Structured response from the RAG system."""
    answer: str
    sources: List[Document]
    confidence: float
    query: str
