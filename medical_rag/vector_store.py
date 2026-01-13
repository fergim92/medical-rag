"""
Vector database for document storage and retrieval.
"""

from typing import List
import numpy as np
import chromadb

from .models import Document, SearchResult


class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.
    Uses ChromaDB for efficient similarity search.
    """

    def __init__(self, collection_name: str = "medical_docs", persist_dir: str = "./chroma_db"):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Vector store initialized. Documents: {self.collection.count()}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of Document objects
            embeddings: Numpy array of embeddings
        """
        self.collection.add(
            ids=[doc.id for doc in documents],
            embeddings=embeddings.tolist(),
            documents=[doc.content for doc in documents],
            metadatas=[{
                "source": doc.source,
                "title": doc.title,
                "chunk_index": doc.chunk_index
            } for doc in documents]
        )
        print(f"Added {len(documents)} documents. Total: {self.collection.count()}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                source=results['metadatas'][0][i]['source'],
                title=results['metadatas'][0][i]['title'],
                chunk_index=results['metadatas'][0][i]['chunk_index']
            )
            # Convert distance to similarity score (cosine)
            score = 1 - results['distances'][0][i]
            search_results.append(SearchResult(document=doc, score=score))

        return search_results

    def clear(self):
        """Clear all documents from the store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.collection.count()
