import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_system import ScientificRAGSystem
from src.vector_database import VectorDatabase


def test_with_existing_papers():
    # Initialize
    rag = ScientificRAGSystem(db_path="./test_db_current")

    # Process all papers in data/pdf
    pdf_dir = "data/pdfs"
    for i, filename in enumerate(os.listdir(pdf_dir)):
        if filename.endswith(".pdf"):
            paper_path = os.path.join(pdf_dir, filename)
            doc_id = f"paper_{i + 1}"

            print(f"\nProcessing {filename}...")
            result = rag.process_pdf_document(
                pdf_source=paper_path,
                document_id=doc_id,
                metadata={"source": filename}
            )
            print(f"→ Result: {result['status']}")
            print(f"→ Chunks: {result.get('chunks_created', 0)}")

    # Test queries
    test_queries = [
        "What is the main contribution of this work?",
        "What methods were used?",
        "What were the key findings?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        response = rag.query_papers(query, top_k=2)
        for j, result in enumerate(response['search_results']):
            print(f"{j + 1}. [{result['metadata']['document_id']}]")
            print(f"   Score: {result['similarity_score']:.3f}")
            print(f"   Text: {result['text'][:80]}...")

    # Cleanup
    VectorDatabase("./test_db_current").reset_collection()


if __name__ == "__main__":
    test_with_existing_papers()