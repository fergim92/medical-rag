"""
Embedding model for medical text vectorization.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalEmbeddings:
    """
    Embedding model optimized for medical text.
    Uses sentence-transformers with configurable models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: The sentence-transformer model to use.
                Options:
                - 'all-MiniLM-L6-v2': Fast, general purpose (default)
                - 'pritamdeka/S-PubMedBert-MS-MARCO': Medical domain optimized
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query string to embed

        Returns:
            numpy array representing the query embedding
        """
        return self.model.encode([query])[0]
