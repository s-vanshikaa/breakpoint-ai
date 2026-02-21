from dataclasses import dataclass
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "documents"

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


def load_and_chunk_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Chunk]:
    chunks = []
    for doc_path in sorted(documents_dir.glob("*.md")):
        text = doc_path.read_text()
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(text=piece, source=doc_path.name, chunk_index=i))
    return chunks
