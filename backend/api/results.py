import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"

router = APIRouter(prefix="/results", tags=["results"])


def _load_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"{filename} not found. Run the benchmark first."
        )
    with open(path) as f:
        return json.load(f)


@router.get("/summary")
async def get_summary() -> dict:
    return _load_json("comparison.json")


@router.get("/records")
async def get_records(
    run: str = Query("guarded", pattern="^(baseline|guarded)$"),
    category: str | None = None,
) -> list[dict]:
    data = _load_json(f"{run}.json")
    records = data["records"]
    if category:
        records = [r for r in records if r["category"] == category]
    return records
