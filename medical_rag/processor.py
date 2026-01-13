"""
Document processing and chunking utilities.
"""

from typing import List
import nltk
from nltk.tokenize import sent_tokenize

from .models import Document

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class DocumentProcessor:
    """
    Processes medical documents into chunks suitable for RAG.
    Implements smart chunking that respects sentence boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[Document]:
        """
        Split a document into overlapping chunks.
        Respects sentence boundaries for better context.

        Args:
            doc: Document to chunk

        Returns:
            List of Document chunks
        """
        sentences = sent_tokenize(doc.content)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    id=f"{doc.id}_chunk_{chunk_index}",
                    content=chunk_text,
                    source=doc.source,
                    title=doc.title,
                    chunk_index=chunk_index
                ))
                chunk_index += 1

                # Keep overlap
                overlap_text = " ".join(current_chunk[-2:]) if len(current_chunk) > 2 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(
                id=f"{doc.id}_chunk_{chunk_index}",
                content=chunk_text,
                source=doc.source,
                title=doc.title,
                chunk_index=chunk_index
            ))

        return chunks
