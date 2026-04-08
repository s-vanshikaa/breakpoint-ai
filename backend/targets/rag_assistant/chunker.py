from dataclasses import dataclass
from pathlib import Path

from config import settings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def load_and_chunk_documents(documents_dir: Path | None = None) -> list[Chunk]:
    documents_dir = documents_dir or settings.documents_dir
    if not documents_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}. "
            "Set DATA_DIR if your data lives outside the repository."
        )
    chunks = []
    for doc_path in sorted(documents_dir.glob("*.md")):
        text = doc_path.read_text()
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(text=piece, source=doc_path.name, chunk_index=i))
    return chunks
