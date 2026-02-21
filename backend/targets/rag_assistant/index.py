import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from targets.rag_assistant.chunker import Chunk, load_and_chunk_documents

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorIndex:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chunks = chunks if chunks is not None else load_and_chunk_documents()

        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
