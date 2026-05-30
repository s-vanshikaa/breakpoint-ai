"""Crash-safe incremental checkpointing for one trial's evaluations.

Each completed TestRecord is appended to a trial's `.checkpoint.jsonl` file as soon as it's
done (one JSON object per line, flushed immediately), independent of the final consolidated
`<config>-seed-<n>.json` that's written only once every case in the trial has run. If the
process dies mid-trial, the checkpoint file has every case that finished before the crash;
resuming re-reads it, skips those cases, and only runs what's left. Once a trial's final file
is written, its checkpoint is no longer needed and is removed.
"""

from pathlib import Path

from pydantic import ValidationError

from runner.schema import TestRecord


def checkpoint_path(trials_dir: Path, config: str, seed: int) -> Path:
    return trials_dir / f"{config}-seed-{seed}.checkpoint.jsonl"


def append_record(path: Path, record: TestRecord) -> None:
    """Appends one record as a single JSON line, flushed so a crash right after doesn't lose
    it. A crash *during* the write can still leave a truncated final line; load_records()
    tolerates that by stopping at the first line it can't parse instead of raising."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json())
        f.write("\n")
        f.flush()


def load_records(path: Path) -> list[TestRecord]:
    """Every record successfully checkpointed before path was last written to (or truncated)."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(TestRecord.model_validate_json(line))
        except (ValidationError, ValueError):
            break  # a partially-written trailing line from a crash; nothing after it is valid
    return records


def clear(path: Path) -> None:
    path.unlink(missing_ok=True)
