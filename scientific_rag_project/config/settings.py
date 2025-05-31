import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys - Only OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Text Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    # Vector Database
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "scientific_papers")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths
    DATA_DIR = "data"
    PDF_DIR = "data/pdfs"
    PROCESSED_DIR = "data/processed"


settings = Settings()