#!/usr/bin/env python3
"""
Medical RAG System - Demo Entry Point
=====================================
Run this script to see the Medical RAG system in action.

Usage:
    python main.py

Requirements:
    - Set GROQ_API_KEY environment variable, OR
    - Have Ollama running locally with llama3.1 model
"""

from medical_rag import MedicalRAG
from medical_rag.data import SAMPLE_MEDICAL_DOCS


def main():
    """Demo of the Medical RAG System."""
    print("=" * 60)
    print("MEDICAL RAG SYSTEM DEMO")
    print("=" * 60)

    # Initialize the RAG system
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
