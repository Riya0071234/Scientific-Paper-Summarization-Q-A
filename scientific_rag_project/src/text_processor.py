import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    source_document: str
    page_number: Optional[int] = None
    section_type: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: Optional[int] = None


class ScientificTextChunker:
    """Advanced text chunking specifically designed for scientific papers"""

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize the chunker with scientific paper optimizations

        Args:
            chunk_size: Target size for each chunk (in tokens)
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to avoid tiny fragments
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Scientific paper section patterns
        self.section_patterns = {
            'abstract': r'\b(abstract|summary)\b',
            'introduction': r'\b(introduction|background)\b',
            'methods': r'\b(methods?|methodology|experimental|procedure)\b',
            'results': r'\b(results?|findings?|outcomes?)\b',
            'discussion': r'\b(discussion|analysis|interpretation)\b',
            'conclusion': r'\b(conclusion|summary|final remarks)\b',
            'references': r'\b(references?|bibliography|citations?)\b'
        }

        # Load spaCy model for better sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            print("Warning: spaCy model not found. Using NLTK for sentence tokenization.")
            self.nlp = None

    def identify_section_type(self, text: str) -> str:
        """Identify the section type based on content patterns"""
        text_lower = text.lower()

        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return section_type

        return 'content'

    def smart_sentence_split(self, text: str) -> List[str]:
        """Enhanced sentence splitting for scientific text"""
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = sent_tokenize(text)

        # Post-process to handle scientific text quirks
        processed_sentences = []
        for sentence in sentences:
            # Skip very short sentences (likely artifacts)
            if len(sentence.strip()) < 10:
                continue

            # Handle common scientific abbreviations
            sentence = self._fix_scientific_abbreviations(sentence)
            processed_sentences.append(sentence.strip())

        return processed_sentences

    def _fix_scientific_abbreviations(self, text: str) -> str:
        """Fix common scientific abbreviation splitting issues"""
        # Common scientific abbreviations that shouldn't be split
        abbreviations = [
            r'\be\.g\.',
            r'\bi\.e\.',
            r'\bet al\.',
            r'\bvs\.',
            r'\bca\.',
            r'\bcf\.',
            r'\bpp\.',
            r'\bvol\.',
            r'\bno\.',
            r'\bfig\.',
            r'\btable\s+\d+',
            r'\bequation\s+\d+',
        ]

        for abbrev in abbreviations:
            text = re.sub(abbrev, lambda m: m.group().replace('.', '◦'), text, flags=re.IGNORECASE)

        # Restore periods after processing
        text = text.replace('◦', '.')
        return text

    def semantic_chunking(self, text: str, document_id: str) -> List[TextChunk]:
        """
        Create semantically coherent chunks using sentence boundaries
        and scientific paper structure awareness
        """
        sentences = self.smart_sentence_split(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_counter = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = len(sentence.split())

            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)

                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk_id = f"{document_id}_chunk_{chunk_counter}"
                    section_type = self.identify_section_type(chunk_text)

                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source_document=document_id,
                        section_type=section_type,
                        token_count=current_tokens
                    ))
                    chunk_counter += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_id = f"{document_id}_chunk_{chunk_counter}"
                section_type = self.identify_section_type(chunk_text)

                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_document=document_id,
                    section_type=section_type,
                    token_count=current_tokens
                ))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences:
            return []

        overlap_sentences = []
        token_count = 0

        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = len(sentence.split())
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break

        return overlap_sentences


class EmbeddingGenerator:
    """Generate high-quality embeddings for scientific text"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model optimized for scientific text

        Args:
            model_name: Name of the sentence transformer model
                       Options:
                       - "all-MiniLM-L6-v2" (fast, good general performance)
                       - "all-mpnet-base-v2" (better quality, slower)
                       - "sentence-transformers/allenai-specter" (scientific papers)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        print(f"Generating embeddings for {len(texts)} texts...")

        # Process in batches to manage memory
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)
        print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

        return embeddings

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode([text])[0]


class TextProcessor:
    """Main class that orchestrates text chunking and embedding generation"""

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the text processor

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            embedding_model: Sentence transformer model name
        """
        self.chunker = ScientificTextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(embedding_model)

    def process_document(self,
                         text: str,
                         document_id: str,
                         metadata: Optional[Dict] = None) -> Tuple[List[TextChunk], np.ndarray]:
        """
        Process a document: chunk text and generate embeddings

        Args:
            text: Document text
            document_id: Unique identifier for the document
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (chunks, embeddings)
        """
        print(f"Processing document: {document_id}")

        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)

        # Create chunks
        chunks = self.chunker.semantic_chunking(cleaned_text, document_id)
        print(f"Created {len(chunks)} chunks")

        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(chunk_texts)

        # Add metadata if provided
        if metadata:
            for chunk in chunks:
                for key, value in metadata.items():
                    setattr(chunk, key, value)

        return chunks, embeddings

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better chunking"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common PDF extraction issues
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation across lines
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces

        # Clean up special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\(\)\[\]\-\'\"]', ' ', text)

        return text.strip()

    def similarity_search(self,
                          query: str,
                          chunk_embeddings: np.ndarray,
                          chunks: List[TextChunk],
                          top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        Perform similarity search to find most relevant chunks

        Args:
            query: Search query
            chunk_embeddings: Pre-computed chunk embeddings
            chunks: List of text chunks
            top_k: Number of top results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)

        # Calculate cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((chunks[idx], float(similarities[idx])))

        return results

    def get_embedding_stats(self) -> Dict:
        """Get statistics about the embedding model"""
        return {
            'model_name': self.embedding_generator.model_name,
            'embedding_dimension': self.embedding_generator.embedding_dimension,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.chunk_overlap
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the text processor
    processor = TextProcessor(
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2"
    )

    # Example scientific text
    sample_text = """
    Abstract
    This study investigates the effects of machine learning algorithms on scientific paper analysis.
    We developed a novel approach using transformer-based models to extract key information from research documents.

    Introduction
    Scientific literature is growing exponentially, making it challenging for researchers to stay current with developments.
    Traditional methods of literature review are time-consuming and often incomplete.
    Recent advances in natural language processing offer promising solutions for automated text analysis.

    Methods
    We utilized a combination of text chunking strategies and embedding generation techniques.
    The corpus consisted of 1000 peer-reviewed articles from various scientific domains.
    Each document was processed using semantic segmentation and vector representations were computed using sentence transformers.

    Results
    Our approach achieved 87% accuracy in retrieving relevant information compared to manual annotation.
    The system demonstrated particular strength in identifying methodological sections and experimental procedures.
    Response times averaged 0.3 seconds per query across the entire corpus.

    Discussion
    The results suggest that automated approaches can significantly enhance literature review processes.
    However, domain-specific fine-tuning may be necessary for optimal performance in specialized fields.
    Future work should explore integration with existing research databases and citation networks.

    Conclusion
    We present a robust framework for scientific document analysis that combines semantic chunking with high-quality embeddings.
    This approach offers substantial improvements over traditional keyword-based search methods.
    """

    # Process the document
    chunks, embeddings = processor.process_document(
        text=sample_text,
        document_id="sample_paper_001",
        metadata={"title": "ML for Scientific Analysis", "year": 2024}
    )

    # Print processing results
    print(f"\nProcessing Results:")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")

    # Display sample chunks
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i + 1} (Section: {chunk.section_type}):")
        print(f"Text: {chunk.text[:100]}...")
        print(f"Tokens: {chunk.token_count}")
        print("-" * 50)

    # Test similarity search
    query = "What machine learning methods were used in the study?"
    results = processor.similarity_search(query, embeddings, chunks, top_k=3)

    print(f"\nSimilarity Search Results for: '{query}'")
    for i, (chunk, score) in enumerate(results):
        print(f"Result {i + 1} (Score: {score:.3f}):")
        print(f"Section: {chunk.section_type}")
        print(f"Text: {chunk.text[:150]}...")
        print("-" * 50)

    # Display embedding statistics
    stats = processor.get_embedding_stats()
    print(f"\nEmbedding Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")