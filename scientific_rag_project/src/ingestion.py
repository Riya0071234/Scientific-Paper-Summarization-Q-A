from src.text_processor import TextProcessor
from src.vector_database import VectorDatabase
import PyPDF2
import requests
import os


class PDFIngestor:
    def __init__(self, vector_db: VectorDatabase, text_processor: TextProcessor):
        self.vector_db = vector_db
        self.text_processor = text_processor

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

    def download_pdf_from_url(self, url: str, save_path: str = None) -> str:
        """Download PDF from URL"""
        if save_path is None:
            save_path = f"./data/papers/{url.split('/')[-1]}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path

    def ingest_pdf(self, pdf_source: str, document_id: str, is_url: bool = False) -> bool:
        """Full ingestion pipeline for a PDF"""
        try:
            if is_url:
                pdf_path = self.download_pdf_from_url(pdf_source)
            else:
                pdf_path = pdf_source

            text = self.extract_text_from_pdf(pdf_path)
            chunks, embeddings = self.text_processor.process_document(text, document_id)

            self.vector_db.add_embeddings(
                [{
                    'content': chunk.text,
                    'chunk_id': chunk.chunk_id,
                    'source_file': document_id,
                    'section_type': chunk.section_type,
                    'token_count': chunk.token_count,
                    'embedding': embeddings[i]
                } for i, chunk in enumerate(chunks)]
            )
            return True
        except Exception as e:
            print(f"Ingestion failed: {e}")
            return False