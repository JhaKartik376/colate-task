from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class TextChunk:
    text: str
    page_number: int
    chunk_index: int
    source_file: str


class PDFExtractor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def extract_chunks(self, pdf_path: Path) -> list[TextChunk]:
        doc = fitz.open(pdf_path)
        chunks = []
        chunk_index = 0

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            page_chunks = self._split_text(page_text)

            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(
                        TextChunk(
                            text=chunk_text.strip(),
                            page_number=page_num,
                            chunk_index=chunk_index,
                            source_file=str(pdf_path.name),
                        )
                    )
                    chunk_index += 1

        doc.close()
        return chunks

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                break_point = text.rfind(". ", start, end)
                if break_point == -1 or break_point <= start:
                    break_point = text.rfind(" ", start, end)
                if break_point > start:
                    end = break_point + 1

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks
