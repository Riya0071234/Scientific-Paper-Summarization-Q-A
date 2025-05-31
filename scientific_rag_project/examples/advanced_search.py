# examples/advanced_search.py
import sys

sys.path.append('../src')

from rag_integration import ScientificRAGSystem


def advanced_search_example():
    rag = ScientificRAGSystem()

    # Search with metadata filters
    results = rag.search_papers(
        query="transformer architecture",
        top_k=10,
        filter_metadata={"year": 2017, "domain": "deep_learning"}
    )

    for result in results:
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Paper: {result['metadata']['title']}")
        print(f"Text: {result['text'][:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    advanced_search_example()