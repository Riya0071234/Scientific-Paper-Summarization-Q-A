import chromadb
import chromadb.errors
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import json
from pathlib import Path
import uuid
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Remove circular import - comment out the self-referencing import
# from vector_database import VectorDatabase
from src.text_processor import TextProcessor


# Example usage with proper text variable definition
def example_usage_with_text_processor():
    """Example showing how to use the database with TextProcessor"""

    # Initialize components
    vector_db = VectorDatabase("./existing_db")
    text_processor = TextProcessor()

    # Define the text variable that was missing
    text = """
    Your document text content goes here. This could be loaded from a file,
    retrieved from a database, or passed as a parameter to your function.
    """

    # Process and store documents
    chunks, embeddings = text_processor.process_document(text, "doc_id")
    vector_db.add_documents(
        texts=[chunk.text for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"chunk_id": chunk.chunk_id} for chunk in chunks],
        ids=[chunk.chunk_id for chunk in chunks]
    )


class VectorDatabase:
    def __init__(self,
                 collection_name: str = "scientific_papers",
                 persist_directory: str = "data/chromadb"):

        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        print(f"Initializing ChromaDB at: {self.persist_directory}")

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection using helper method
        self.collection = self._get_or_create_collection(collection_name)

        print(f"Collection count: {self.collection.count()}")

    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one if it doesn't exist."""
        try:
            collection = self.client.get_collection(name=collection_name)
            print(f"✓ Connected to existing collection: {collection_name}")
            return collection
        except (ValueError, chromadb.errors.NotFoundError):
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Scientific papers embeddings", "hnsw:space": "cosine"}
            )
            print(f"✓ Created new collection: {collection_name}")
            return collection
        except Exception as e:
            # Handle any other unexpected errors
            print(f"❌ Unexpected error accessing collection: {e}")
            raise

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(name=self.collection_name)
            return True
        except (ValueError, chromadb.errors.NotFoundError):
            return False

    def add_embeddings(self,
                       chunk_embeddings: List[Dict],
                       batch_size: int = 100) -> bool:
        """Add embeddings to the vector database"""

        if not chunk_embeddings:
            print("No embeddings to add")
            return False

        print(f"Adding {len(chunk_embeddings)} embeddings to database...")

        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in chunk_embeddings:
                # Generate unique ID
                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                ids.append(chunk_id)

                # Extract embedding
                embedding = chunk['embedding']
                if isinstance(embedding, list):
                    embeddings.append(embedding)
                else:
                    embeddings.append(embedding.tolist())

                # Extract document text
                documents.append(chunk.get('content', ''))

                # Prepare metadata (ChromaDB doesn't support all types)
                metadata = {
                    'source_file': chunk.get('source_file', 'unknown'),
                    'section_type': chunk.get('section_type', 'unknown'),
                    'token_count': int(chunk.get('token_count', 0)),
                    'priority_score': float(chunk.get('priority_score', 0.5)),
                    'page_numbers': json.dumps(chunk.get('page_numbers', [])),
                    'embedding_model': chunk.get('embedding_model', 'unknown')
                }
                metadatas.append(metadata)

            # Add to collection in batches
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))

                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

                print(f"✓ Added batch {i // batch_size + 1}/{(len(ids) + batch_size - 1) // batch_size}")

            print(f"✅ Successfully added {len(chunk_embeddings)} embeddings")
            print(f"Total collection size: {self.collection.count()}")
            return True

        except Exception as e:
            print(f"❌ Error adding embeddings: {e}")
            return False

    def add_documents(self, texts: List[str], embeddings: List[List[float]],
                      metadatas: List[Dict], ids: List[str]) -> bool:
        """Add documents with their embeddings to the database"""
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"✅ Successfully added {len(texts)} documents")
            return True
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False

    def similarity_search(self,
                          query_embedding: np.ndarray,
                          top_k: int = 5,
                          filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar chunks using embedding"""

        try:
            # Convert embedding to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata

            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=['metadatas', 'documents', 'distances']
            )

            # Format results
            formatted_results = []
            if results['ids'][0]:  # Check if we have results
                for i, chunk_id in enumerate(results['ids'][0]):
                    result = {
                        'chunk_id': chunk_id,
                        'content': results['documents'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i]
                    }

                    # Parse page_numbers back from JSON
                    if 'page_numbers' in result['metadata']:
                        try:
                            result['metadata']['page_numbers'] = json.loads(result['metadata']['page_numbers'])
                        except:
                            result['metadata']['page_numbers'] = []

                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []

    def search_by_text(self,
                       query_text: str,
                       embedding_generator,
                       top_k: int = 5,
                       filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search using text query (generates embedding automatically)"""

        # Generate embedding for query
        query_embedding = embedding_generator.generate_single_embedding(query_text)

        # Perform similarity search
        return self.similarity_search(query_embedding, top_k, filter_metadata)

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()

            # Get a sample to analyze
            sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])

            stats = {
                'total_chunks': count,
                'collection_name': self.collection_name
            }

            if sample_results['metadatas']:
                # Analyze source files
                source_files = set()
                section_types = set()
                total_tokens = 0
                priority_scores = []

                for metadata in sample_results['metadatas']:
                    source_files.add(metadata.get('source_file', 'unknown'))
                    section_types.add(metadata.get('section_type', 'unknown'))
                    total_tokens += metadata.get('token_count', 0)
                    priority_scores.append(metadata.get('priority_score', 0.5))

                stats.update({
                    'unique_source_files': len(source_files),
                    'source_files': list(source_files),
                    'section_types': list(section_types),
                    'avg_tokens_per_chunk': total_tokens / len(sample_results['metadatas']) if sample_results[
                        'metadatas'] else 0,
                    'avg_priority_score': sum(priority_scores) / len(priority_scores) if priority_scores else 0
                })

            return stats

        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return {'total_chunks': 0, 'collection_name': self.collection_name}

    def delete_by_source_file(self, source_file: str) -> bool:
        """Delete all chunks from a specific source file"""
        try:
            # Get all IDs for this source file
            results = self.collection.get(
                where={"source_file": source_file},
                include=['ids']
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"✓ Deleted {len(results['ids'])} chunks from {source_file}")
                return True
            else:
                print(f"No chunks found for source file: {source_file}")
                return False

        except Exception as e:
            print(f"❌ Error deleting chunks: {e}")
            return False

    def reset_collection(self) -> bool:
        """Reset the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Scientific papers embeddings"}
            )
            print(f"✓ Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error resetting collection: {e}")
            return False


# Test function
def test_vector_database():
    """Test the vector database functionality"""
    print("=== Testing Vector Database ===")

    # Initialize database
    db = VectorDatabase("test_collection", "data/test_chromadb")

    # Create some test embeddings
    test_embeddings = [
        {
            'chunk_id': 'test_1',
            'content': 'Machine learning algorithms for protein structure prediction.',
            'source_file': 'test_paper_1.pdf',
            'section_type': 'abstract',
            'token_count': 50,
            'priority_score': 0.9,
            'page_numbers': [1],
            'embedding_model': 'test',
            'embedding': np.random.rand(384).tolist()  # Random embedding for testing
        },
        {
            'chunk_id': 'test_2',
            'content': 'Deep learning approaches in computational biology applications.',
            'source_file': 'test_paper_2.pdf',
            'section_type': 'introduction',
            'token_count': 75,
            'priority_score': 0.8,
            'page_numbers': [2, 3],
            'embedding_model': 'test',
            'embedding': np.random.rand(384).tolist()
        }
    ]

    # Test adding embeddings
    success = db.add_embeddings(test_embeddings)
    print(f"Add embeddings success: {success}")

    # Test similarity search
    query_embedding = np.random.rand(384)
    results = db.similarity_search(query_embedding, top_k=2)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result['chunk_id']}: {result['similarity']:.3f}")

    # Test collection stats
    stats = db.get_collection_stats()
    print(f"Collection stats: {stats}")

    # Clean up test collection
    db.reset_collection()
    print("✓ Vector database test completed!")


if __name__ == "__main__":
    test_vector_database()