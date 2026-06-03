"""Aggregation for benchmark job outcomes. Pure logic, no I/O - reuses runner.aggregate's
percentile() so the two benchmark tables and the security-evaluation aggregates use identical
math."""

from collections import Counter

from pydantic import BaseModel

from benchmark.jobs import JobOutcome
from runner.aggregate import FAILURE_STATUSES, percentile


class PerformanceResult(BaseModel):
    concurrency: int
    total_jobs: int
    valid_jobs: int  # status == "ok"
    failed_jobs: int
    duration_seconds: float
    throughput_per_sec: float  # valid_jobs / duration_seconds
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    retry_count: int  # sum(attempts - 1) across all jobs
    timeout_count: int  # jobs whose final status == "timeout"


class ReliabilityResult(BaseModel):
    injected_failure_rate: float
    total_jobs: int
    completed: int  # status == "ok" (an alias of valid, named to match the brief's vocabulary)
    completion_rate: float | None  # completed / total_jobs
    recovered: int  # retried and ultimately succeeded (status == "ok")
    permanent_failures: int  # status != "ok" after retries were exhausted (or not retryable)
    total_retries: int
    duplicate_job_ids: int  # count of job IDs appearing more than once in the final result set
    duration_seconds: float
    failures_by_status: dict[str, int]


def _latencies(outcomes: list[JobOutcome]) -> list[float]:
    return [o.latency_ms for o in outcomes]


def summarize_performance(
    concurrency: int, outcomes: list[JobOutcome], duration_seconds: float
) -> PerformanceResult:
    valid = [o for o in outcomes if o.status == "ok"]
    latencies = _latencies(outcomes)
    return PerformanceResult(
        concurrency=concurrency,
        total_jobs=len(outcomes),
        valid_jobs=len(valid),
        failed_jobs=len(outcomes) - len(valid),
        duration_seconds=duration_seconds,
        throughput_per_sec=len(valid) / duration_seconds if duration_seconds > 0 else 0.0,
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        retry_count=sum(o.attempts - 1 for o in outcomes),
        timeout_count=sum(1 for o in outcomes if o.status == "timeout"),
    )


def summarize_reliability(
    injected_failure_rate: float,
    outcomes: list[JobOutcome],
    duration_seconds: float,
    duplicate_job_ids: int = 0,
) -> ReliabilityResult:
    total = len(outcomes)
    completed = sum(1 for o in outcomes if o.status == "ok")
    recovered = sum(1 for o in outcomes if o.status == "ok" and o.retried)
    permanent_failures = total - completed
    failures_by_status = Counter(o.status for o in outcomes if o.status != "ok")
    return ReliabilityResult(
        injected_failure_rate=injected_failure_rate,
        total_jobs=total,
        completed=completed,
        completion_rate=completed / total if total else None,
        recovered=recovered,
        permanent_failures=permanent_failures,
        total_retries=sum(o.attempts - 1 for o in outcomes),
        duplicate_job_ids=duplicate_job_ids,
        duration_seconds=duration_seconds,
        failures_by_status={s: failures_by_status.get(s, 0) for s in FAILURE_STATUSES},
    )


def count_duplicates(ids: list[str]) -> int:
    counts = Counter(ids)
    return sum(n - 1 for n in counts.values() if n > 1)
