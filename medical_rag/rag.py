"""
Main RAG orchestrator for medical question answering.
"""

from typing import List, Dict, Optional
import numpy as np

from .models import Document, SearchResult, RAGResponse
from .embeddings import MedicalEmbeddings
from .processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_client import LLMClient


class MedicalRAG:
    """
    Main RAG system for medical question answering.
    Combines retrieval, reranking, and generation with citations.
    """

    SYSTEM_PROMPT = """You are a medical information assistant. Your role is to provide
accurate, helpful information based on the provided medical documentation.

IMPORTANT GUIDELINES:
1. Only answer based on the provided context. If the information isn't in the context, say so.
2. Always cite your sources using [Source: title] format.
3. Be precise and factual. Never make up medical information.
4. Include relevant warnings or disclaimers when appropriate.
5. If asked about diagnosis or treatment, remind users to consult healthcare professionals.

Format your response clearly with:
- A direct answer to the question
- Supporting evidence from the sources
- Relevant citations
"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_provider: str = "groq",
        api_key: Optional[str] = None,
        persist_dir: str = "./chroma_db"
    ):
        """
        Initialize the Medical RAG System.

        Args:
            embedding_model: Sentence transformer model name
            llm_provider: LLM provider ('groq' or 'ollama')
            api_key: API key for cloud LLM providers
            persist_dir: Directory for vector store persistence
        """
        print("Initializing Medical RAG System...")
        self.embeddings = MedicalEmbeddings(embedding_model)
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.llm = LLMClient(provider=llm_provider, api_key=api_key)
        print("Medical RAG System ready!")

    def ingest_documents(self, documents: List[Dict[str, str]]):
        """
        Ingest documents into the RAG system.

        Args:
            documents: List of dicts with 'content', 'source', 'title' keys
        """
        print(f"Ingesting {len(documents)} documents...")

        all_chunks = []
        for i, doc_data in enumerate(documents):
            doc = Document(
                id=f"doc_{i}",
                content=doc_data["content"],
                source=doc_data["source"],
                title=doc_data["title"]
            )
            chunks = self.processor.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks. Generating embeddings...")

        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embeddings.embed(texts)

        # Store in vector database
        self.vector_store.add_documents(all_chunks, embeddings)
        print("Documents ingested successfully!")

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """
        Answer a medical question using RAG.

        Args:
            question: The user's medical question
            top_k: Number of relevant documents to retrieve

        Returns:
            RAGResponse with answer, sources, and confidence
        """
        # Step 1: Embed the query
        query_embedding = self.embeddings.embed_query(question)

        # Step 2: Retrieve relevant documents
        search_results = self.vector_store.search(query_embedding, top_k=top_k)

        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the medical documentation.",
                sources=[],
                confidence=0.0,
                query=question
            )

        # Step 3: Build context from retrieved documents
        context = self._build_context(search_results)

        # Step 4: Generate answer using LLM
        prompt = f"""Based on the following medical documentation, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based only on the provided context. Include citations."""

        answer = self.llm.generate(prompt, self.SYSTEM_PROMPT)

        # Calculate confidence based on retrieval scores
        avg_score = np.mean([r.score for r in search_results])
        confidence = min(avg_score * 1.2, 1.0)

        return RAGResponse(
            answer=answer,
            sources=[r.document for r in search_results],
            confidence=confidence,
            query=question
        )

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[Source {i}: {result.document.title}]\n{result.document.content}\n"
            )
        return "\n".join(context_parts)

    def clear_database(self):
        """Clear all documents from the vector store."""
        self.vector_store.clear()
        print("Database cleared.")

    def document_count(self) -> int:
        """Return the number of documents in the store."""
        return self.vector_store.count()
