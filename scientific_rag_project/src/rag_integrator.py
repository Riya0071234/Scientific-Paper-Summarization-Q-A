from text_processor import TextProcessor
from vector_database import VectorDatabase


class RAGIntegration:
    def __init__(self, db_path="data/chromadb"):
        self.text_processor = TextProcessor()
        self.vector_db = VectorDatabase(persist_directory=db_path)

    def process_document(self, text, document_id, metadata=None):
        """Process a document through the full pipeline"""
        chunks, embeddings = self.text_processor.process_document(
            text=text,
            document_id=document_id,
            metadata=metadata
        )

        # Prepare for vector DB
        texts = [chunk.text for chunk in chunks]
        embeddings = embeddings.tolist()
        metadatas = [{
            'chunk_id': chunk.chunk_id,
            'document_id': document_id,
            'section_type': chunk.section_type,
            'token_count': chunk.token_count
        } for chunk in chunks]

        # Add to vector DB
        self.vector_db.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[chunk.chunk_id for chunk in chunks]
        )

        return chunks

    def query(self, question, top_k=5):
        """Query the system with a question"""
        # Generate query embedding
        query_embedding = self.text_processor.embedding_generator.generate_single_embedding(question)

        # Search vector DB
        results = self.vector_db.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results