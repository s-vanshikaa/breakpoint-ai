import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from attacks.schema import TestCase
from models.ollama_client import benchmark_session, ollama_client
from runner.common import ResultsError, ensure_ollama_ready
from runner.comparison import ComparisonReport, compute_comparison
from runner.metrics import BaselineMetrics, compute_baseline_metrics
from runner.runner import DEFAULT_CONCURRENCY, run_tests
from runner.schema import TestRecord

RUN_NAMES = ("baseline", "guarded")
COMPARISON_FILENAME = "comparison.json"


class StoredRun(BaseModel):
    metrics: BaselineMetrics
    records: list[TestRecord]
    meta: dict | None = None  # absent in results produced before run metadata existed


def run_path(results_dir: Path, name: str) -> Path:
    return results_dir / f"{name}.json"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # an interrupted run never leaves a half-written results file


def load_run(path: Path, hint: str) -> StoredRun:
    if not path.is_file():
        raise ResultsError(f"{path} not found. {hint}")
    try:
        return StoredRun.model_validate_json(path.read_text())
    except (ValidationError, ValueError) as e:
        raise ResultsError(f"{path} is malformed ({e}). {hint}") from e


def print_progress(index: int, total: int, record: TestRecord) -> None:
    status = "PASS" if record.passed else "FAIL"
    print(f"  [{index:>{len(str(total))}}/{total}] {record.test_id:<16}{status}", flush=True)


async def execute_run(
    name: str,
    test_cases: list[TestCase],
    results_dir: Path,
    seed: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> StoredRun:
    """Runs the benchmark with guardrails off ('baseline') or on ('guarded') and saves it."""
    if name not in RUN_NAMES:
        raise ValueError(f"Unknown run '{name}'. Expected one of {RUN_NAMES}.")
    guardrails_enabled = name == "guarded"

    await ensure_ollama_ready()
    ollama_client.default_options = {"seed": seed} if seed is not None else {}

    seed_text = seed if seed is not None else "none (sampling is not fixed)"
    print(
        f"Running {name} benchmark: {len(test_cases)} cases, "
        f"guardrails {'ON' if guardrails_enabled else 'OFF'}, "
        f"model {ollama_client.model}, seed {seed_text}, concurrency {concurrency}"
    )
    async with benchmark_session(ollama_client, timeout=timeout, max_retries=max_retries):
        records = await run_tests(
            test_cases, guardrails_enabled, on_record=print_progress, concurrency=concurrency
        )
    metrics = compute_baseline_metrics(records, test_cases, guardrails_enabled=guardrails_enabled)
    meta = {
        "run": name,
        "model": ollama_client.model,
        "seed": seed,
        "concurrency": concurrency,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    path = run_path(results_dir, name)
    write_json(
        path,
        {
            "meta": meta,
            "metrics": metrics.model_dump(),
            "records": [r.model_dump() for r in records],
        },
    )
    print(f"Saved {name} results to {path}")
    return StoredRun(metrics=metrics, records=records, meta=meta)


def build_comparison(baseline: StoredRun, guarded: StoredRun) -> ComparisonReport:
    if baseline.metrics.guardrails_enabled or not guarded.metrics.guardrails_enabled:
        raise ResultsError(
            "baseline.json must be a guardrails-off run and guarded.json a guardrails-on run."
        )
    baseline_ids = {r.test_id for r in baseline.records}
    guarded_ids = {r.test_id for r in guarded.records}
    if baseline_ids != guarded_ids:
        differing = sorted(baseline_ids ^ guarded_ids)
        raise ResultsError(
            "baseline and guarded runs cover different test cases "
            f"({len(differing)} differ, e.g. {', '.join(differing[:3])}). "
            "Re-run both against the same dataset."
        )
    return compute_comparison(baseline.metrics, guarded.metrics)


def compare_saved_runs(results_dir: Path) -> tuple[ComparisonReport, StoredRun, StoredRun]:
    """Builds and saves comparison.json from the saved baseline and guarded runs."""
    baseline = load_run(
        run_path(results_dir, "baseline"),
        "Run `python -m runner baseline` first.",
    )
    guarded = load_run(
        run_path(results_dir, "guarded"),
        "Run `python -m runner guarded` first.",
    )
    comparison = build_comparison(baseline, guarded)
    path = results_dir / COMPARISON_FILENAME
    write_json(path, comparison.model_dump())
    print(f"Saved comparison to {path}")
    return comparison, baseline, guarded

