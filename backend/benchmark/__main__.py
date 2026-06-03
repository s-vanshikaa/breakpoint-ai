"""python -m benchmark [options]

Benchmarks the BreakPoint AI runner itself (scheduling, retries, checkpointing) against a
deterministic mock inference server - not real Ollama. See benchmark/mock_server.py for why.
"""

import argparse
import asyncio
from pathlib import Path

from benchmark.harness import DEFAULT_CONCURRENCIES, DEFAULT_FAILURE_RATES, run_full_benchmark
from benchmark.report import format_report
from config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description=(
            "Benchmark the runner's scheduling/retry/checkpoint path against a mock "
            "inference server."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed (default: 0).")
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(DEFAULT_CONCURRENCIES),
        help=(
            "Concurrency values for the performance sweep "
            f"(default: {list(DEFAULT_CONCURRENCIES)})."
        ),
    )
    parser.add_argument("--performance-jobs", type=int, default=300, dest="performance_jobs")
    parser.add_argument(
        "--performance-latency-ms", type=float, default=30.0, dest="performance_latency_ms"
    )
    parser.add_argument(
        "--reliability-failure-rates", type=float, nargs="+",
        default=list(DEFAULT_FAILURE_RATES), dest="reliability_failure_rates",
    )
    parser.add_argument("--reliability-jobs", type=int, default=300, dest="reliability_jobs")
    parser.add_argument(
        "--reliability-concurrency", type=int, default=4, dest="reliability_concurrency"
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=3, dest="max_retries")
    parser.add_argument(
        "--benchmarks-dir", type=Path, default=None,
        help=f"Where results are written (default: {settings.benchmarks_dir}).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path, report = asyncio.run(
        run_full_benchmark(
            args.benchmarks_dir or settings.benchmarks_dir,
            seed=args.seed,
            concurrencies=args.concurrencies,
            performance_jobs_per_run=args.performance_jobs,
            performance_latency_ms=args.performance_latency_ms,
            reliability_failure_rates=args.reliability_failure_rates,
            reliability_jobs_per_run=args.reliability_jobs,
            reliability_concurrency=args.reliability_concurrency,
            client_timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
    )
    print(f"Saved benchmark to {path}\n")
    print(format_report(report))


if __name__ == "__main__":
    main()
