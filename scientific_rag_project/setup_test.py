from config.settings import settings
from sentence_transformers import SentenceTransformer
import chromadb


def test_complete_setup():
    print("Testing complete setup...")

    # Test configuration loading
    print(f"✓ Chunk size: {settings.CHUNK_SIZE}")
    print(f"✓ Embedding model: {settings.EMBEDDING_MODEL}")

    # Test embedding model loading
    try:
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("✓ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False

    # Test ChromaDB
    try:
        client = chromadb.Client()
        print("✓ ChromaDB client created successfully")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

    # Test sample PDF processing (we'll use this in next phase)
    try:
        import fitz
        print("✓ PDF processing ready")
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return False

    print("\n🎉 Complete setup verified!")
    print("Ready for Phase 2: PDF Processing Pipeline")
    return True


if __name__ == "__main__":
    test_complete_setup()