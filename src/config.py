"""Configuration management for the AI Research Assistant."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    openai_api_key: str
    chroma_persist_dir: Path
    pdf_storage_dir: Path
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
        pdf_dir = Path(os.getenv("PDF_STORAGE_DIR", "./data/pdfs"))

        chroma_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            openai_api_key=api_key,
            chroma_persist_dir=chroma_dir,
            pdf_storage_dir=pdf_dir,
        )


config = Config.from_env() if os.getenv("OPENAI_API_KEY") else None


def get_config() -> Config:
    """Get the application configuration."""
    global config
    if config is None:
        config = Config.from_env()
    return config
