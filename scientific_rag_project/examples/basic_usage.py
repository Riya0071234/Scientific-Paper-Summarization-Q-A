# examples/basic_usage.py
import sys

sys.path.append('../src')

from embedding import TextProcessor


def basic_text_processing_example():
    # Initialize processor with custom settings
    processor = TextProcessor(
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2"
    )

    # Process a scientific paper
    text = "Your scientific paper text here..."
    chunks, embeddings = processor.process_document(
        text=text,
        document_id="paper_001",
        metadata={"title": "Sample Paper", "year": 2024}
    )

    print(f"Created {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    basic_text_processing_example()