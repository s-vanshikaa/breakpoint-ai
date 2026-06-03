"""Plain-text tables for a BenchmarkReport. Pure formatting, no I/O."""

from benchmark.harness import BenchmarkReport


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def format_performance_table(report: BenchmarkReport) -> str:
    header = (
        f"{'Concurrency':>11} | {'Jobs':>5} | {'Runtime':>9} | {'Throughput':>10} | "
        f"{'p50':>7} | {'p95':>7} | {'p99':>7} | {'Success':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in report.performance:
        success = r.valid_jobs / r.total_jobs if r.total_jobs else 0.0
        lines.append(
            f"{r.concurrency:>11} | {r.total_jobs:>5} | {r.duration_seconds:>8.2f}s | "
            f"{r.throughput_per_sec:>9.1f}/s | {ms(r.p50_ms):>6}ms | {ms(r.p95_ms):>6}ms | "
            f"{ms(r.p99_ms):>6}ms | {pct(success):>7}"
        )
    return "\n".join(lines)


def format_reliability_table(report: BenchmarkReport) -> str:
    header = (
        f"{'Inj. failure':>12} | {'Jobs':>5} | {'Completion':>10} | {'Recovered':>9} | "
        f"{'Permanent':>9} | {'Retries':>7} | {'Duplicates':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in report.reliability:
        lines.append(
            f"{pct(r.injected_failure_rate):>12} | {r.total_jobs:>5} | "
            f"{pct(r.completion_rate):>10} | {r.recovered:>9} | {r.permanent_failures:>9} | "
            f"{r.total_retries:>7} | {r.duplicate_job_ids:>10}"
        )
    return "\n".join(lines)


def format_report(report: BenchmarkReport) -> str:
    lines = [
        f"Benchmark {report.benchmark_id} ({report.created_at})",
        f"Environment: Python {report.environment.python_version}, {report.environment.platform}",
        "",
        "MOCK-SERVER PERFORMANCE (runner scheduling/pooling only - not real LLM throughput):",
        format_performance_table(report),
        "",
        "MOCK-SERVER RELIABILITY (retry/checkpoint recovery under injected transient failures):",
        format_reliability_table(report),
        "",
        "Resume drill (crash after "
        f"{report.resume_drill.jobs_before_interruption}/{report.resume_drill.total_jobs} jobs, "
        "then resumed from checkpoint):",
        f"  jobs after resume: {report.resume_drill.jobs_after_resume}",
        f"  duplicate job IDs: {report.resume_drill.duplicate_job_ids}",
        f"  every job present exactly once: {report.resume_drill.all_ids_present_exactly_once}",
    ]
    return "\n".join(lines)
