"""Synthetic benchmark jobs: dispatches N jobs through the real OllamaClient (retry, backoff,
timeout classification, connection pooling all real) with the same bounded-concurrency,
stable-ordering, one-job-runs-exactly-once guarantees as runner.runner.run_tests - reimplemented
here rather than imported because run_tests is specialized to TestCase/RAGAssistant/ToolAgent,
whereas a benchmark job is just a prompt. Job outcomes are converted to real TestRecords so they
can be checkpointed with the exact same runner.checkpoint module production experiments use.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from models.ollama_client import OllamaClient, OllamaError, classify_error
from runner.schema import TestRecord

BENCHMARK_CATEGORY = "benchmark"


@dataclass(frozen=True)
class JobOutcome:
    job_id: str
    status: str  # "ok" or one of runner.schema's failure statuses
    attempts: int
    retried: bool
    latency_ms: float
    error: str | None = None


def job_ids(n: int, prefix: str = "job") -> list[str]:
    width = len(str(max(n - 1, 0)))
    return [f"{prefix}-{i:0{width}d}" for i in range(n)]


def _prompt(job_id: str) -> str:
    return f"benchmark job {job_id}"


async def run_job(client: OllamaClient, job_id: str) -> JobOutcome:
    start = time.perf_counter()
    try:
        result = await client.complete_with_retry(_prompt(job_id))
        return JobOutcome(
            job_id=job_id, status="ok", attempts=result.attempts, retried=result.retried,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    except OllamaError as e:
        return JobOutcome(
            job_id=job_id, status=classify_error(e), attempts=e.attempts or 1,
            retried=bool(e.retried), latency_ms=(time.perf_counter() - start) * 1000, error=str(e),
        )


async def run_jobs(
    client: OllamaClient,
    ids: list[str],
    concurrency: int,
    on_outcome: Callable[[int, int, JobOutcome], None] | None = None,
) -> list[JobOutcome]:
    """Bounded-concurrency dispatch: at most `concurrency` jobs in flight, every job in `ids`
    runs exactly once, and the returned list is reordered to match `ids` regardless of
    completion order (mirrors runner.runner.run_tests's ordering guarantee)."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def run_one(index: int, job_id: str) -> tuple[int, JobOutcome]:
        nonlocal completed
        async with semaphore:
            outcome = await run_job(client, job_id)
        completed += 1
        if on_outcome:
            on_outcome(completed, len(ids), outcome)
        return index, outcome

    results = await asyncio.gather(*(run_one(i, jid) for i, jid in enumerate(ids)))
    results.sort(key=lambda pair: pair[0])
    return [outcome for _, outcome in results]


def outcome_to_record(outcome: JobOutcome) -> TestRecord:
    """A benchmark job's outcome as a real TestRecord, so it can be checkpointed/loaded with the
    exact same runner.checkpoint module a production experiment uses."""
    return TestRecord(
        test_id=outcome.job_id,
        category=BENCHMARK_CATEGORY,
        target=BENCHMARK_CATEGORY,
        prompt=_prompt(outcome.job_id),
        response="mock response" if outcome.status == "ok" else "",
        passed=outcome.status == "ok",
        reason="benchmark job (not a security evaluation)",
        latency_ms=outcome.latency_ms,
        guardrails_enabled=False,
        status=outcome.status,
        attempt_count=outcome.attempts,
        retried=outcome.retried,
        error=outcome.error,
    )
