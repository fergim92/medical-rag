"""
Medical RAG System - Retrieval Augmented Generation for Healthcare
==================================================================
A production-ready RAG system that answers medical questions using
verified documentation sources with citation support.

Author: Fernando Gimenez
Portfolio: LLM Engineer Position
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Vector database
import chromadb
from chromadb.utils import embedding_functions

# LLM Integration (supports multiple providers)
import requests

# Text processing
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


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


class MedicalEmbeddings:
    """
    Embedding model optimized for medical text.
    Uses sentence-transformers with a biomedical model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        For production, consider using:
        - 'pritamdeka/S-PubMedBert-MS-MARCO' for medical domain
        - 'all-MiniLM-L6-v2' for general purpose (faster)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode([query])[0]


class DocumentProcessor:
    """
    Processes medical documents into chunks suitable for RAG.
    Implements smart chunking that respects sentence boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[Document]:
        """
        Split a document into overlapping chunks.
        Respects sentence boundaries for better context.
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


class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.
    Uses ChromaDB for efficient similarity search.
    """

    def __init__(self, collection_name: str = "medical_docs", persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Vector store initialized. Documents: {self.collection.count()}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with their embeddings to the store."""
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
        """Search for similar documents using cosine similarity."""
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


class LLMClient:
    """
    LLM client supporting multiple providers.
    Default: Groq API (free, fast inference)
    """

    def __init__(self, provider: str = "groq", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        self.endpoints = {
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "ollama": "http://localhost:11434/api/chat"
        }

        self.models = {
            "groq": "llama-3.1-70b-versatile",
            "ollama": "llama3.1"
        }

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
        """Generate a response from the LLM."""

        if self.provider == "groq":
            return self._groq_generate(prompt, system_prompt, temperature)
        elif self.provider == "ollama":
            return self._ollama_generate(prompt, system_prompt, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _groq_generate(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate using Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.models["groq"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        response = requests.post(
            self.endpoints["groq"],
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _ollama_generate(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate using local Ollama."""
        data = {
            "model": self.models["ollama"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": temperature}
        }

        response = requests.post(
            self.endpoints["ollama"],
            json=data,
            timeout=60
        )
        response.raise_for_status()

        return response.json()["message"]["content"]


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
        api_key: Optional[str] = None
    ):
        print("Initializing Medical RAG System...")
        self.embeddings = MedicalEmbeddings(embedding_model)
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
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
        confidence = min(avg_score * 1.2, 1.0)  # Scale and cap at 1.0

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


# ============================================================================
# DEMO: Sample Medical Data
# ============================================================================

SAMPLE_MEDICAL_DOCS = [
    {
        "title": "Diabetes Mellitus Overview",
        "source": "Medical Encyclopedia 2024",
        "content": """
        Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar
        levels over a prolonged period. Symptoms often include frequent urination, increased
        thirst, and increased hunger. If left untreated, diabetes can cause many complications.

        Type 1 diabetes results from the pancreas's failure to produce enough insulin due to
        loss of beta cells. Type 2 diabetes begins with insulin resistance, a condition in
        which cells fail to respond to insulin properly. As the disease progresses, a lack
        of insulin may also develop.

        Treatment includes lifestyle modifications, medication, and insulin therapy. Regular
        monitoring of blood glucose levels is essential. The target HbA1c level for most
        adults with diabetes is below 7%. Complications include cardiovascular disease,
        stroke, chronic kidney disease, foot ulcers, damage to the eyes, and cognitive
        impairment.
        """
    },
    {
        "title": "Hypertension Management Guidelines",
        "source": "Clinical Practice Guidelines 2024",
        "content": """
        Hypertension, or high blood pressure, is defined as systolic blood pressure ≥130 mmHg
        or diastolic blood pressure ≥80 mmHg. It is a major risk factor for cardiovascular
        disease, stroke, and kidney disease.

        First-line treatments include lifestyle modifications: weight loss, DASH diet (rich
        in fruits, vegetables, whole grains), reduced sodium intake (<2300mg/day), regular
        physical activity (150 minutes/week moderate intensity), and limited alcohol consumption.

        Pharmacological treatment is recommended for patients with BP ≥140/90 mmHg or those
        with cardiovascular risk factors at ≥130/80 mmHg. First-line medications include
        ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. Target BP
        is <130/80 mmHg for most patients.
        """
    },
    {
        "title": "COVID-19 Symptoms and Treatment",
        "source": "WHO Clinical Guidelines 2024",
        "content": """
        COVID-19 is caused by the SARS-CoV-2 virus. Common symptoms include fever, cough,
        fatigue, loss of taste or smell, shortness of breath, muscle aches, and headache.
        Symptoms typically appear 2-14 days after exposure.

        Most cases are mild and can be managed at home with rest, fluids, and over-the-counter
        medications for symptom relief. Severe cases may require hospitalization for oxygen
        therapy or mechanical ventilation.

        Prevention includes vaccination, wearing masks in crowded indoor spaces, hand hygiene,
        and maintaining physical distance. Antiviral treatments like Paxlovid may be prescribed
        for high-risk patients within 5 days of symptom onset.
        """
    },
    {
        "title": "Asthma Diagnosis and Treatment",
        "source": "Respiratory Medicine Textbook",
        "content": """
        Asthma is a chronic inflammatory disease of the airways characterized by variable
        and recurring symptoms, reversible airflow obstruction, and bronchospasm. Symptoms
        include wheezing, coughing, chest tightness, and shortness of breath.

        Diagnosis is based on clinical history, physical examination, and spirometry showing
        reversible airflow obstruction (≥12% improvement in FEV1 after bronchodilator).

        Treatment follows a stepwise approach. Quick-relief medications (short-acting beta
        agonists like albuterol) are used for acute symptoms. Controller medications include
        inhaled corticosteroids (first-line), long-acting beta agonists, and leukotriene
        modifiers. Severe asthma may require biologic therapies targeting specific inflammatory
        pathways.
        """
    },
    {
        "title": "Drug Interactions and Safety",
        "source": "Pharmacology Reference Guide",
        "content": """
        Drug interactions can significantly affect medication efficacy and safety. Common
        interaction types include pharmacokinetic (affecting absorption, distribution,
        metabolism, excretion) and pharmacodynamic (affecting drug action at receptor sites).

        Notable interactions to monitor:
        - Warfarin with NSAIDs: increased bleeding risk
        - ACE inhibitors with potassium supplements: hyperkalemia risk
        - Statins with grapefruit juice: increased statin levels
        - SSRIs with MAOIs: serotonin syndrome risk
        - Metformin with contrast dye: lactic acidosis risk

        Healthcare providers should review all medications including OTC drugs and supplements.
        Patients should be educated about potential interactions and warning signs.
        """
    }
]


def main():
    """Demo of the Medical RAG System."""
    print("=" * 60)
    print("MEDICAL RAG SYSTEM DEMO")
    print("=" * 60)

    # Initialize the RAG system
    # For demo without API key, we'll show the architecture
    rag = MedicalRAG(
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="groq"  # Set GROQ_API_KEY env variable
    )

    # Ingest sample medical documents
    rag.ingest_documents(SAMPLE_MEDICAL_DOCS)

    # Demo queries
    demo_questions = [
        "What are the symptoms of diabetes and how is it treated?",
        "What is the target blood pressure for hypertensive patients?",
        "What medications should not be combined with warfarin?",
        "How is asthma diagnosed?"
    ]

    print("\n" + "=" * 60)
    print("QUERYING THE RAG SYSTEM")
    print("=" * 60)

    for question in demo_questions:
        print(f"\n{'─' * 60}")
        print(f"QUESTION: {question}")
        print('─' * 60)

        try:
            response = rag.query(question)
            print(f"\nANSWER:\n{response.answer}")
            print(f"\nCONFIDENCE: {response.confidence:.2%}")
            print(f"\nSOURCES:")
            for src in response.sources[:3]:
                print(f"  - {src.title} ({src.source})")
        except Exception as e:
            print(f"Error (API key may not be set): {e}")
            print("Set GROQ_API_KEY environment variable for full demo")
            break


if __name__ == "__main__":
    main()
