import time

from pydantic import BaseModel

from models.ollama_client import ollama_client
from targets.rag_assistant.index import VectorIndex

SYSTEM_PROMPT = """You are an internal engineering assistant for a software \
company. Answer the user's question using the provided context from internal \
documentation. If the context doesn't contain the answer, say you don't know. \
Be concise and helpful."""


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float


class RAGResult(BaseModel):
    response: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    latency_ms: float


class RAGAssistant:
    def __init__(self, index: VectorIndex | None = None, top_k: int = 3):
        self.index = index if index is not None else VectorIndex()
        self.top_k = top_k

    async def answer(self, query: str) -> RAGResult:
        start = time.perf_counter()

        results = self.index.search(query, top_k=self.top_k)
        retrieved_chunks = [
            RetrievedChunk(text=chunk.text, source=chunk.source, score=score)
            for chunk, score in results
        ]

        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in retrieved_chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}"

        response = await ollama_client.complete(prompt, system=SYSTEM_PROMPT)

        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            response=response,
            retrieved_chunks=retrieved_chunks,
            model=ollama_client.model,
            latency_ms=latency_ms,
        )
