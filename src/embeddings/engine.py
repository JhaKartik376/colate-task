import logging
from typing import Callable

from openai import OpenAI

from src.config import get_config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    BATCH_SIZE = 50  # Smaller batches for stability

    def __init__(self):
        config = get_config()
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.embedding_model

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        if len(texts) <= self.BATCH_SIZE:
            logger.info(f"Embedding {len(texts)} texts in single batch")
            response = self.client.embeddings.create(input=texts, model=self.model)
            if progress_callback:
                progress_callback(1.0)
            return [item.embedding for item in response.data]

        all_embeddings = []
        total = len(texts)
        logger.info(f"Embedding {total} texts in {(total + self.BATCH_SIZE - 1) // self.BATCH_SIZE} batches")

        for i in range(0, total, self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            batch_num = i // self.BATCH_SIZE + 1
            logger.info(f"Processing batch {batch_num} ({len(batch)} texts)")

            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([item.embedding for item in response.data])

            progress = min(i + self.BATCH_SIZE, total) / total
            if progress_callback:
                progress_callback(progress)
            logger.info(f"Batch {batch_num} complete ({int(progress * 100)}%)")

        return all_embeddings
