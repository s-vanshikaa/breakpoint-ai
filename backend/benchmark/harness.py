"""Orchestrates the performance and reliability benchmarks against benchmark.mock_server, and
persists machine-readable results under data/benchmarks/ (see runner.workflow.write_json for
the atomic-write helper reused here).

Every run uses the real models.ollama_client.OllamaClient (real HTTP, real retry/backoff/
timeout logic, real connection pooling) and the real runner.checkpoint module. Only the network
endpoint is fake. See the module docstring in benchmark/mock_server.py for why a mock endpoint,
rather than real Ollama, is the right target for measuring the *runner's* performance and
reliability.
"""

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from benchmark.jobs import job_ids, outcome_to_record, run_jobs
from benchmark.metrics import (
    PerformanceResult,
    ReliabilityResult,
    count_duplicates,
    summarize_performance,
    summarize_reliability,
)
from benchmark.mock_server import MockConfig, MockServer
from models.ollama_client import OllamaClient
from models.retry import RetryPolicy
from runner import checkpoint
from runner.workflow import write_json

DEFAULT_CONCURRENCIES = (1, 2, 4)
DEFAULT_FAILURE_RATES = (0.0, 0.05, 0.10, 0.20)
RESULT_FILENAME_PREFIX = "benchmark-"
LATEST_FILENAME = "latest.json"


class ResumeDrillResult(BaseModel):
    """Proves the checkpoint+resume path never re-runs or duplicates a completed job: half the
    jobs are run and checkpointed, the "crash" is simulated by simply stopping there, and then
    the same runner.checkpoint module used by production experiments is used to resume."""

    total_jobs: int
    jobs_before_interruption: int
    jobs_after_resume: int
    duplicate_job_ids: int
    all_ids_present_exactly_once: bool


class BenchmarkConfig(BaseModel):
    seed: int
    concurrencies: list[int]
    performance_jobs_per_run: int
    performance_latency_ms: float
    reliability_failure_rates: list[float]
    reliability_jobs_per_run: int
    reliability_concurrency: int
    client_timeout_seconds: float
    max_retries: int


class Environment(BaseModel):
    python_version: str
    platform: str


class BenchmarkReport(BaseModel):
    benchmark_id: str
    created_at: str
    config: BenchmarkConfig
    environment: Environment
    performance: list[PerformanceResult]
    reliability: list[ReliabilityResult]
    resume_drill: ResumeDrillResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_benchmark_id() -> str:
    return RESULT_FILENAME_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _retry_policy(max_retries: int) -> RetryPolicy:
    # Short, fixed backoff for the benchmark itself - long enough to be real exponential
    # backoff, short enough that hundreds of retried requests don't dominate wall-clock time.
    return RetryPolicy(
        max_attempts=max_retries, base_delay_seconds=0.02, max_delay_seconds=0.1,
        jitter_seconds=0.01,
    )


async def _run_batch(
    config: MockConfig, ids: list[str], concurrency: int, timeout: float, max_retries: int
) -> tuple[list, float]:
    async with MockServer(config) as server:
        client = OllamaClient(host=server.host, model=config.model, timeout=timeout)
        client.retry_policy = _retry_policy(max_retries)
        async with client:
            start = time.perf_counter()
            outcomes = await run_jobs(client, ids, concurrency=concurrency)
            duration = time.perf_counter() - start
    return outcomes, duration


async def run_performance_benchmark(
    seed: int,
    concurrencies: list[int],
    jobs_per_run: int,
    latency_ms: float,
    timeout: float = 2.0,
) -> list[PerformanceResult]:
    """One point per concurrency value, at 0% injected failure - isolates scheduling/pooling
    overhead from retry noise, so a runtime difference between concurrency values reflects only
    how well the runner overlaps requests."""
    results = []
    for concurrency in concurrencies:
        config = MockConfig(seed=seed, mode="fixed_latency", fixed_latency_ms=latency_ms)
        outcomes, duration = await _run_batch(
            config, job_ids(jobs_per_run), concurrency, timeout, max_retries=1
        )
        results.append(summarize_performance(concurrency, outcomes, duration))
    return results


async def run_reliability_benchmark(
    seed: int,
    failure_rates: list[float],
    concurrency: int,
    jobs_per_run: int,
    timeout: float = 2.0,
    max_retries: int = 3,
) -> list[ReliabilityResult]:
    """One point per injected transient-failure rate, at a fixed concurrency - measures how
    much of a given failure rate the retry policy actually recovers from."""
    results = []
    for rate in failure_rates:
        config = MockConfig(
            seed=seed, mode="success", failure_rate=rate, timeout_sleep_seconds=timeout * 2
        )
        outcomes, duration = await _run_batch(
            config, job_ids(jobs_per_run), concurrency, timeout, max_retries
        )
        duplicates = count_duplicates([o.job_id for o in outcomes])
        results.append(
            summarize_reliability(rate, outcomes, duration, duplicate_job_ids=duplicates)
        )
    return results


async def run_resume_drill(
    scratch_dir: Path,
    seed: int,
    n_jobs: int = 40,
    split: int = 20,
    failure_rate: float = 0.1,
    concurrency: int = 4,
    timeout: float = 2.0,
    max_retries: int = 3,
) -> ResumeDrillResult:
    ckpt_path = scratch_dir / "resume-drill.checkpoint.jsonl"
    checkpoint.clear(ckpt_path)
    all_ids = job_ids(n_jobs)
    first_half = all_ids[:split]
    config = MockConfig(
        seed=seed, mode="success", failure_rate=failure_rate, timeout_sleep_seconds=timeout * 2
    )

    async with MockServer(config) as server:
        client = OllamaClient(host=server.host, model=config.model, timeout=timeout)
        client.retry_policy = _retry_policy(max_retries)
        async with client:

            def checkpoint_outcome(i: int, total: int, outcome) -> None:
                checkpoint.append_record(ckpt_path, outcome_to_record(outcome))

            await run_jobs(
                client, first_half, concurrency=concurrency, on_outcome=checkpoint_outcome
            )
            # Simulated crash: a real interruption just stops here. What's on disk right now is
            # exactly what a resume has to work with.

            already_done = {r.test_id for r in checkpoint.load_records(ckpt_path)}
            remaining = [jid for jid in all_ids if jid not in already_done]
            await run_jobs(
                client, remaining, concurrency=concurrency, on_outcome=checkpoint_outcome
            )

    final_ids = [r.test_id for r in checkpoint.load_records(ckpt_path)]
    checkpoint.clear(ckpt_path)

    return ResumeDrillResult(
        total_jobs=n_jobs,
        jobs_before_interruption=len(first_half),
        jobs_after_resume=len(remaining),
        duplicate_job_ids=count_duplicates(final_ids),
        all_ids_present_exactly_once=(
            sorted(final_ids) == sorted(all_ids) and len(final_ids) == len(set(final_ids))
        ),
    )


async def run_full_benchmark(
    benchmarks_dir: Path,
    seed: int = 0,
    concurrencies: list[int] | None = None,
    performance_jobs_per_run: int = 300,
    performance_latency_ms: float = 30.0,
    reliability_failure_rates: list[float] | None = None,
    reliability_jobs_per_run: int = 300,
    reliability_concurrency: int = 4,
    client_timeout_seconds: float = 2.0,
    max_retries: int = 3,
) -> tuple[Path, BenchmarkReport]:
    """Runs the full performance + reliability + resume-drill suite and writes a timestamped
    result file plus latest.json, both under `benchmarks_dir`. Never overwrites a previous
    timestamped result; only latest.json is replaced."""
    concurrencies = list(concurrencies or DEFAULT_CONCURRENCIES)
    reliability_failure_rates = list(reliability_failure_rates or DEFAULT_FAILURE_RATES)
    benchmark_id = new_benchmark_id()

    performance = await run_performance_benchmark(
        seed, concurrencies, performance_jobs_per_run, performance_latency_ms,
        timeout=client_timeout_seconds,
    )
    reliability = await run_reliability_benchmark(
        seed, reliability_failure_rates, reliability_concurrency, reliability_jobs_per_run,
        timeout=client_timeout_seconds, max_retries=max_retries,
    )
    resume_drill = await run_resume_drill(
        benchmarks_dir, seed, timeout=client_timeout_seconds, max_retries=max_retries,
    )

    report = BenchmarkReport(
        benchmark_id=benchmark_id,
        created_at=_now_iso(),
        config=BenchmarkConfig(
            seed=seed,
            concurrencies=concurrencies,
            performance_jobs_per_run=performance_jobs_per_run,
            performance_latency_ms=performance_latency_ms,
            reliability_failure_rates=reliability_failure_rates,
            reliability_jobs_per_run=reliability_jobs_per_run,
            reliability_concurrency=reliability_concurrency,
            client_timeout_seconds=client_timeout_seconds,
            max_retries=max_retries,
        ),
        environment=Environment(
            python_version=sys.version.split()[0], platform=platform.platform()
        ),
        performance=performance,
        reliability=reliability,
        resume_drill=resume_drill,
    )

    result_path = benchmarks_dir / f"{benchmark_id}.json"
    write_json(result_path, report.model_dump())
    write_json(benchmarks_dir / LATEST_FILENAME, report.model_dump())
    return result_path, report
