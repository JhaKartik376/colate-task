from typing import Callable

import chromadb
from chromadb.config import Settings

from src.config import get_config
from src.embeddings import EmbeddingEngine
from src.pdf.extractor import TextChunk


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        config = get_config()
        self.client = chromadb.PersistentClient(
            path=str(config.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_engine = EmbeddingEngine()
        self.top_k = config.top_k_results

    def add_chunks(
        self,
        chunks: list[TextChunk],
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_engine.embed_texts(texts, progress_callback)

        ids = [f"{chunk.source_file}_{chunk.chunk_index}" for chunk in chunks]
        metadatas = [
            {
                "page_number": chunk.page_number,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        query_embedding = self.embedding_engine.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return search_results

    def delete_document(self, source_file: str) -> None:
        self.collection.delete(where={"source_file": source_file})

    def list_documents(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for metadata in results["metadatas"]:
            if metadata and "source_file" in metadata:
                sources.add(metadata["source_file"])
        return list(sources)
