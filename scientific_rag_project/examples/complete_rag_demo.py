# examples/complete_rag_demo.py
import sys

sys.path.append('../src')

from rag_integration import ScientificRAGSystem


def complete_rag_example():
    # Initialize RAG system
    rag = ScientificRAGSystem(
        db_path="../data/vector_db",
        chunk_size=512,
        embedding_model="all-MiniLM-L6-v2"
    )

    # Add a PDF paper
    result = rag.process_pdf_document(
        pdf_source="../data/papers/sample_paper.pdf",
        document_id="important_paper_2024",
        metadata={
            "title": "Important Scientific Discovery",
            "authors": ["Dr. Smith", "Dr. Jones"],
            "year": 2024,
            "venue": "Nature",
            "domain": "machine_learning"
        }
    )

    # Query the system
    response = rag.query_papers(
        question="What machine learning methods were used?",
        top_k=5
    )

    print(f"Answer: {response['answer']}")
    print(f"Sources: {len(response['sources'])}")


if __name__ == "__main__":
    complete_rag_example()