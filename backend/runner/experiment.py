"""Multi-trial experiments: the benchmark repeated for baseline and guarded over several seeds.

Layout written under <experiments_dir>/<experiment-id>/:

    manifest.json                 what was run, on what, and the status of every trial
    trials/<config>-seed-<n>.json one file per trial (full records, never only averages)
    aggregate.json                cross-trial metrics (see runner.aggregate)

Every file is written atomically, and the manifest is rewritten after each trial, so an
interrupted experiment leaves a valid manifest describing what finished.
"""

import hashlib
import json
import random
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from attacks.schema import TestCase
from config import REPO_ROOT
from models.ollama_client import OllamaError, ollama_client
from runner.aggregate import CONFIGS, GUARDED, ExperimentAggregate, Trial, aggregate_experiment
from runner.common import ResultsError, ensure_ollama_ready
from runner.metrics import compute_baseline_metrics
from runner.runner import run_tests
from runner.schema import TestRecord
from runner.workflow import print_progress, write_json

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
TRIALS_DIRNAME = "trials"

DETERMINISM_NOTE = (
    "Test order is fixed (dataset order) and each trial passes its seed to Ollama as the "
    "sampling seed, so trials are repeatable in intent. Local LLM inference is not guaranteed "
    "bit-for-bit reproducible across hardware, Ollama versions or GPU/CPU backends."
)


class ExperimentError(ResultsError):
    """The experiment request is invalid (bad seeds, existing output folder, ...)."""


class TrialEntry(BaseModel):
    config: str
    seed: int
    status: str  # "completed" | "failed"
    file: str
    error: str | None = None
    started_at: str
    completed_at: str


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    experiment_id: str
    created_at: str
    completed_at: str | None = None
    status: str  # "running" | "complete" | "partial"
    git_sha: str | None = None
    git_dirty: bool | None = None
    model: str
    ollama_host: str
    configurations: list[str]
    seeds: list[int]
    test_case_count: int
    dataset_sha256: str  # hash of the test cases as loaded, in run order
    determinism_note: str = DETERMINISM_NOTE
    trials: list[TrialEntry] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_experiment_id() -> str:
    return "exp-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def trial_filename(config: str, seed: int) -> str:
    return f"{config}-seed-{seed}.json"


def dataset_fingerprint(test_cases: list[TestCase]) -> str:
    digest = hashlib.sha256()
    for tc in test_cases:
        digest.update(tc.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def git_state(repo: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """Best-effort (commit SHA, has uncommitted changes); (None, None) when git isn't usable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=5
        )
        if sha.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=5
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return sha.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def validate_seeds(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ExperimentError("An experiment needs at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ExperimentError(f"Seeds must be unique, got {seeds}.")
    return sorted(seeds)  # fixed execution order regardless of how seeds were typed


def _write_trial(
    path: Path,
    experiment_id: str,
    config: str,
    seed: int,
    test_cases: list[TestCase],
    records: list[TestRecord],
    error: str | None,
    started_at: str,
    completed_at: str,
) -> None:
    data: dict = {
        "meta": {
            "experiment_id": experiment_id,
            "run": config,
            "seed": seed,
            "model": ollama_client.model,
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "status": "failed" if error else "completed",
        "error": error,
        "records": [r.model_dump() for r in records],
    }
    if not error:
        metrics = compute_baseline_metrics(
            records, test_cases, guardrails_enabled=config == GUARDED
        )
        data["metrics"] = metrics.model_dump()
    write_json(path, data)


async def run_experiment(
    test_cases: list[TestCase],
    seeds: list[int],
    experiments_dir: Path,
    experiment_id: str | None = None,
    on_record: Callable[[int, int, TestRecord], None] | None = print_progress,
) -> tuple[Manifest, ExperimentAggregate]:
    """Runs baseline and guarded for every seed and persists trials, manifest and aggregate.

    A trial whose model calls fail is recorded as failed and the remaining trials still run.
    """
    if not test_cases:
        raise ExperimentError("The dataset has no test cases.")
    seeds = validate_seeds(seeds)

    experiment_id = experiment_id or new_experiment_id()
    root = experiments_dir / experiment_id
    if root.exists():
        raise ExperimentError(f"{root} already exists; refusing to overwrite an experiment.")

    await ensure_ollama_ready()

    sha, dirty = git_state()
    manifest = Manifest(
        experiment_id=experiment_id,
        created_at=_now(),
        status="running",
        git_sha=sha,
        git_dirty=dirty,
        model=ollama_client.model,
        ollama_host=ollama_client.host,
        configurations=list(CONFIGS),
        seeds=seeds,
        test_case_count=len(test_cases),
        dataset_sha256=dataset_fingerprint(test_cases),
    )
    manifest_path = root / MANIFEST_FILENAME
    write_json(manifest_path, manifest.model_dump())

    planned = [(seed, config) for seed in seeds for config in CONFIGS]
    print(
        f"Experiment {experiment_id}: {len(test_cases)} cases x {len(CONFIGS)} configs "
        f"x {len(seeds)} seeds = {len(test_cases) * len(planned)} evaluations, "
        f"model {ollama_client.model}"
    )

    trials: list[Trial] = []
    previous_options = ollama_client.default_options
    try:
        for n, (seed, config) in enumerate(planned, start=1):
            print(f"\n[trial {n}/{len(planned)}] {config}, seed {seed}")
            random.seed(seed)
            ollama_client.default_options = {"seed": seed}
            started_at = _now()
            records: list[TestRecord] = []
            error: str | None = None
            try:
                records = await run_tests(test_cases, config == GUARDED, on_record=on_record)
            except OllamaError as e:
                error = str(e)
                print(f"  trial failed: {error}")
            completed_at = _now()

            filename = trial_filename(config, seed)
            _write_trial(
                root / TRIALS_DIRNAME / filename,
                experiment_id, config, seed, test_cases, records, error, started_at, completed_at,
            )
            trials.append(
                Trial(
                    config=config,
                    seed=seed,
                    status="failed" if error else "completed",
                    error=error,
                    records=records,
                )
            )
            manifest.trials.append(
                TrialEntry(
                    config=config,
                    seed=seed,
                    status="failed" if error else "completed",
                    file=f"{TRIALS_DIRNAME}/{filename}",
                    error=error,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            )
            write_json(manifest_path, manifest.model_dump())
    finally:
        ollama_client.default_options = previous_options

    aggregate = aggregate_experiment(experiment_id, trials, test_cases, planned=len(planned))
    write_json(root / AGGREGATE_FILENAME, aggregate.model_dump())
    manifest.status = "complete" if not aggregate.trials_failed else "partial"
    manifest.completed_at = _now()
    write_json(manifest_path, manifest.model_dump())
    print(f"\nSaved experiment to {root}")
    return manifest, aggregate


def load_experiment(root: Path) -> tuple[Manifest, list[Trial]]:
    """Reads a saved experiment's manifest and every trial file it lists."""
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ResultsError(f"{manifest_path} not found.")
    try:
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        trials = []
        for entry in manifest.trials:
            data = json.loads((root / entry.file).read_text())
            trials.append(
                Trial(
                    config=entry.config,
                    seed=entry.seed,
                    status=data["status"],
                    error=data.get("error"),
                    records=data["records"],
                )
            )
    except (ValidationError, ValueError, KeyError, OSError) as e:
        raise ResultsError(f"{root} is not a readable experiment ({e}).") from e
    return manifest, trials
