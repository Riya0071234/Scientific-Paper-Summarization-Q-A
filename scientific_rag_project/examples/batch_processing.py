# examples/batch_processing.py
import sys

sys.path.append('../src')

from rag_integration import ScientificRAGSystem


def batch_processing_example():
    rag = ScientificRAGSystem()

    # Define multiple papers
    papers = [
        {
            "pdf_source": "https://arxiv.org/pdf/1706.03762.pdf",
            "document_id": "attention_is_all_you_need",
            "metadata": {
                "title": "Attention Is All You Need",
                "authors": ["Vaswani et al."],
                "year": 2017,
                "domain": "deep_learning"
            },
            "is_url": True
        },
        {
            "pdf_source": "../data/papers/bert_paper.pdf",
            "document_id": "bert_2018",
            "metadata": {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2018,
                "domain": "nlp"
            },
            "is_url": False
        }
    ]

    # Process all papers
    results = rag.add_multiple_papers(papers)
    print(f"Processed {results['successful']} papers successfully")


if __name__ == "__main__":
    batch_processing_example()