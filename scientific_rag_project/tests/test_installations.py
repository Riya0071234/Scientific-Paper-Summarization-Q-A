# tests/test_installation.py
from src.embedding import TextProcessor
from src.rag import ScientificRAGSystem


def test_installation():
    # Quick test script
    processor = TextProcessor()
    rag_system = ScientificRAGSystem()

    print("✅ Phase 3 setup complete!")
    print(f"Embedding model: {processor.get_embedding_stats()['model_name']}")
    print(f"Embedding dimension: {processor.get_embedding_stats()['embedding_dimension']}")


if __name__ == "__main__":
    test_installation()