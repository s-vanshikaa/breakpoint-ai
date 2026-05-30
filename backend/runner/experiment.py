"""Multi-trial experiments: the benchmark repeated for baseline and guarded over several seeds.

Layout written under <experiments_dir>/<experiment-id>/:

    manifest.json                          what was run, on what, and the status of every trial
    trials/<config>-seed-<n>.json          one file per finished trial (full records, never
                                            only averages)
    trials/<config>-seed-<n>.checkpoint.jsonl   completed evaluations for a trial still in
                                            progress (or interrupted); removed once the trial
                                            finishes
    aggregate.json                         cross-trial metrics (see runner.aggregate)

Every file is written atomically, and the manifest is rewritten after each trial, so an
interrupted experiment leaves a valid manifest describing what finished. Within a trial, every
evaluation is additionally checkpointed as it completes (see runner.checkpoint), so a crash
mid-trial loses at most the one evaluation that was in flight, not the whole trial. Use
resume_experiment() to pick an interrupted experiment back up: it skips trials already marked
"completed" in the manifest and, for any other planned trial, resumes from its checkpoint
instead of re-running cases that already finished.
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
from models.ollama_client import benchmark_session, ollama_client
from runner import checkpoint
from runner.aggregate import CONFIGS, GUARDED, ExperimentAggregate, Trial, aggregate_experiment
from runner.common import ResultsError, ensure_ollama_ready
from runner.metrics import compute_baseline_metrics
from runner.runner import DEFAULT_CONCURRENCY, run_tests
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
    concurrency: int = DEFAULT_CONCURRENCY
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
    concurrency: int,
    status: str,
) -> None:
    data: dict = {
        "meta": {
            "experiment_id": experiment_id,
            "run": config,
            "seed": seed,
            "model": ollama_client.model,
            "concurrency": concurrency,
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "status": status,
        "error": error,
        "records": [r.model_dump() for r in records],
    }
    if status == "completed":
        metrics = compute_baseline_metrics(
            records, test_cases, guardrails_enabled=config == GUARDED
        )
        data["metrics"] = metrics.model_dump()
    write_json(path, data)


async def _run_trial(
    test_cases: list[TestCase],
    config: str,
    seed: int,
    trials_dir: Path,
    experiment_id: str,
    concurrency: int,
    on_record: Callable[[int, int, TestRecord], None] | None,
) -> tuple[Trial, TrialEntry]:
    """Runs one trial to completion, resuming from its checkpoint if one exists.

    Every completed evaluation is appended to the trial's checkpoint file as it finishes. If
    `test_cases` includes IDs already present in the checkpoint (a previous attempt got partway
    through), those are skipped and only the remaining cases are run — so calling this twice for
    the same trial never produces duplicate evaluations. The final consolidated trial file is
    always written (even on failure, so the manifest's `file` reference is always valid); the
    checkpoint is only cleared once every case has a result.
    """
    ckpt_path = checkpoint.checkpoint_path(trials_dir, config, seed)
    done_records = checkpoint.load_records(ckpt_path)
    done_ids = {r.test_id for r in done_records}
    remaining = [tc for tc in test_cases if tc.id not in done_ids]
    if done_records:
        print(f"  resuming: {len(done_records)}/{len(test_cases)} cases already checkpointed")

    def record_and_checkpoint(i: int, total: int, record: TestRecord) -> None:
        checkpoint.append_record(ckpt_path, record)
        if on_record:
            on_record(i, total, record)

    started_at = _now()
    error: str | None = None
    new_records: list[TestRecord] = []
    if remaining:
        random.seed(seed)
        ollama_client.default_options = {"seed": seed}
        try:
            new_records = await run_tests(
                remaining, config == GUARDED, on_record=record_and_checkpoint,
                concurrency=concurrency,
            )
        except Exception as e:  # a genuinely unexpected bug; isolate it to this trial
            error = str(e)
            print(f"  trial failed: {error}")
    completed_at = _now()

    by_id = {r.test_id: r for r in (*done_records, *new_records)}
    records = [by_id[tc.id] for tc in test_cases if tc.id in by_id]  # original dataset order
    status = "completed" if error is None and len(records) == len(test_cases) else "failed"

    filename = trial_filename(config, seed)
    _write_trial(
        trials_dir / filename, experiment_id, config, seed, test_cases, records, error,
        started_at, completed_at, concurrency, status,
    )
    if status == "completed":
        checkpoint.clear(ckpt_path)  # nothing left to resume

    trial = Trial(config=config, seed=seed, status=status, error=error, records=records)
    entry = TrialEntry(
        config=config, seed=seed, status=status, file=f"{TRIALS_DIRNAME}/{filename}",
        error=error, started_at=started_at, completed_at=completed_at,
    )
    return trial, entry


def _finalize(
    manifest: Manifest,
    manifest_path: Path,
    root: Path,
    experiment_id: str,
    trials: list[Trial],
    test_cases: list[TestCase],
    planned_count: int,
) -> tuple[Manifest, ExperimentAggregate]:
    aggregate = aggregate_experiment(experiment_id, trials, test_cases, planned=planned_count)
    write_json(root / AGGREGATE_FILENAME, aggregate.model_dump())
    manifest.status = "complete" if not aggregate.trials_failed else "partial"
    manifest.completed_at = _now()
    write_json(manifest_path, manifest.model_dump())
    print(f"\nSaved experiment to {root}")
    return manifest, aggregate


async def run_experiment(
    test_cases: list[TestCase],
    seeds: list[int],
    experiments_dir: Path,
    experiment_id: str | None = None,
    on_record: Callable[[int, int, TestRecord], None] | None = print_progress,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> tuple[Manifest, ExperimentAggregate]:
    """Runs baseline and guarded for every seed and persists trials, manifest and aggregate.

    A trial whose model calls fail after retries are exhausted is recorded per-case as a failed
    evaluation (see runner.runner.run_test_case), not aborted; the remaining trials still run.
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
        concurrency=concurrency,
    )
    manifest_path = root / MANIFEST_FILENAME
    write_json(manifest_path, manifest.model_dump())

    planned = [(seed, config) for seed in seeds for config in CONFIGS]
    print(
        f"Experiment {experiment_id}: {len(test_cases)} cases x {len(CONFIGS)} configs "
        f"x {len(seeds)} seeds = {len(test_cases) * len(planned)} evaluations, "
        f"model {ollama_client.model}, concurrency {concurrency}"
    )

    trials: list[Trial] = []
    previous_options = ollama_client.default_options
    try:
        async with benchmark_session(ollama_client, timeout=timeout, max_retries=max_retries):
            for n, (seed, config) in enumerate(planned, start=1):
                print(f"\n[trial {n}/{len(planned)}] {config}, seed {seed}")
                trial, entry = await _run_trial(
                    test_cases, config, seed, root / TRIALS_DIRNAME, experiment_id,
                    concurrency, on_record,
                )
                trials.append(trial)
                manifest.trials.append(entry)
                write_json(manifest_path, manifest.model_dump())
    finally:
        ollama_client.default_options = previous_options

    return _finalize(manifest, manifest_path, root, experiment_id, trials, test_cases, len(planned))


async def resume_experiment(
    test_cases: list[TestCase],
    experiments_dir: Path,
    experiment_id: str,
    on_record: Callable[[int, int, TestRecord], None] | None = print_progress,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> tuple[Manifest, ExperimentAggregate]:
    """Resumes an experiment by ID: reuses its original seeds, configurations and concurrency
    from the manifest (they are not re-taken from the caller, so a resumed run can't silently
    diverge from what was originally planned), skips trials already marked "completed", and for
    every other planned trial, resumes from its checkpoint (or starts it if it never began).

    `timeout`/`max_retries` may still be overridden on resume: they only affect how robustly a
    request is retried, not what's being evaluated, so they aren't part of the experiment's
    fixed configuration the way seeds/concurrency are.
    """
    root = experiments_dir / experiment_id
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ExperimentError(f"No experiment found at {root} (missing {MANIFEST_FILENAME}).")
    try:
        manifest = Manifest.model_validate_json(manifest_path.read_text())
    except (ValidationError, ValueError) as e:
        raise ExperimentError(f"{manifest_path} is malformed ({e}).") from e

    fingerprint = dataset_fingerprint(test_cases)
    if fingerprint != manifest.dataset_sha256:
        raise ExperimentError(
            f"The current dataset doesn't match experiment {experiment_id}'s original dataset "
            f"({fingerprint[:12]} vs {manifest.dataset_sha256[:12]}); refusing to resume with "
            "different test cases. Start a new experiment instead."
        )

    await ensure_ollama_ready()

    completed_entries = {(e.config, e.seed): e for e in manifest.trials if e.status == "completed"}
    planned = [(seed, config) for seed in manifest.seeds for config in manifest.configurations]
    print(
        f"Resuming experiment {experiment_id}: {len(planned)} planned trials, "
        f"{len(completed_entries)} already completed, concurrency {manifest.concurrency}"
    )

    trials: list[Trial] = []
    resumed_entries: dict[tuple[str, int], TrialEntry] = {}
    previous_options = ollama_client.default_options
    try:
        async with benchmark_session(ollama_client, timeout=timeout, max_retries=max_retries):
            for n, (seed, config) in enumerate(planned, start=1):
                key = (config, seed)
                if key in completed_entries:
                    entry = completed_entries[key]
                    print(f"\n[trial {n}/{len(planned)}] {config}, seed {seed}: already complete")
                    data = json.loads((root / entry.file).read_text())
                    trials.append(
                        Trial(config=config, seed=seed, status="completed", records=data["records"])
                    )
                    continue

                print(f"\n[trial {n}/{len(planned)}] {config}, seed {seed}: resuming")
                trial, entry = await _run_trial(
                    test_cases, config, seed, root / TRIALS_DIRNAME, experiment_id,
                    manifest.concurrency, on_record,
                )
                trials.append(trial)
                resumed_entries[key] = entry
    finally:
        ollama_client.default_options = previous_options

    # Rebuild manifest.trials in the original planned order: completed entries are kept
    # untouched, everything else is replaced by what this resume just produced.
    manifest.trials = [
        completed_entries[(config, seed)] if (config, seed) in completed_entries
        else resumed_entries[(config, seed)]
        for seed, config in planned
    ]
    write_json(manifest_path, manifest.model_dump())

    return _finalize(manifest, manifest_path, root, experiment_id, trials, test_cases, len(planned))


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
