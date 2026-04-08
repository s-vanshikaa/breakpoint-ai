import json

from fastapi import APIRouter, HTTPException, Query

from config import settings

router = APIRouter(prefix="/results", tags=["results"])


def _load_json(filename: str, required_keys: tuple[str, ...] = ()) -> dict:
    path = settings.results_dir / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"{filename} not found. Run the benchmark first "
                "(see the README for the baseline and guarded commands)."
            ),
        )
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{filename} is not valid JSON ({e.msg}). "
                "Re-run the benchmark to regenerate it."
            ),
        ) from e

    missing = [k for k in required_keys if not isinstance(data, dict) or k not in data]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{filename} is malformed: missing {', '.join(missing)}. "
                "Re-run the benchmark to regenerate it."
            ),
        )
    return data


@router.get("/summary")
async def get_summary() -> dict:
    return _load_json("comparison.json", required_keys=("baseline", "guarded"))


@router.get("/records")
async def get_records(
    run: str = Query("guarded", pattern="^(baseline|guarded)$"),
    category: str | None = None,
) -> list[dict]:
    data = _load_json(f"{run}.json", required_keys=("records",))
    records = data["records"]
    if category:
        records = [r for r in records if r.get("category") == category]
    return records
