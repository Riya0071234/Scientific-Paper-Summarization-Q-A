import pytest
from src.text_processor import TextProcessor, ScientificTextChunker

def test_chunking():
    chunker = ScientificTextChunker(chunk_size=500, chunk_overlap=50)
    sample_text = "..."  # Add sample scientific text
    chunks = chunker.semantic_chunking(sample_text, "test_doc")
    assert len(chunks) > 0
    assert all(len(chunk.text) > 0 for chunk in chunks)

def test_embeddings():
    processor = TextProcessor()
    embeddings = processor.embedding_generator.generate_embeddings(["sample text"])
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == processor.embedding_generator.embedding_dimension