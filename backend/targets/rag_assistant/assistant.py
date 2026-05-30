import time

from pydantic import BaseModel

from guardrails.input_guardrail import REFUSAL_MESSAGE, check_input
from guardrails.retrieval_guardrail import filter_retrieved_chunks
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
    block_reason: str | None = None
    # 0 when a guardrail blocked the request before any model call was attempted.
    model_attempts: int = 0
    model_retried: bool = False


class RAGAssistant:
    def __init__(self, index: VectorIndex | None = None, top_k: int = 3):
        self.index = index if index is not None else VectorIndex()
        self.top_k = top_k

    async def answer(self, query: str, guardrails_enabled: bool = False) -> RAGResult:
        start = time.perf_counter()

        if guardrails_enabled:
            block_reason = check_input(query)
            if block_reason:
                return RAGResult(
                    response=REFUSAL_MESSAGE,
                    retrieved_chunks=[],
                    model=ollama_client.model,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    block_reason=block_reason,
                )

        results = self.index.search(query, top_k=self.top_k)
        retrieved_chunks = [
            RetrievedChunk(text=chunk.text, source=chunk.source, score=score)
            for chunk, score in results
        ]

        if guardrails_enabled:
            retrieved_chunks = [
                RetrievedChunk(text=text, source=source, score=score)
                for text, source, score in filter_retrieved_chunks(retrieved_chunks)
            ]

        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in retrieved_chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}"

        result = await ollama_client.complete_with_retry(prompt, system=SYSTEM_PROMPT)

        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            response=result.text,
            retrieved_chunks=retrieved_chunks,
            model=ollama_client.model,
            latency_ms=latency_ms,
            model_attempts=result.attempts,
            model_retried=result.retried,
        )
