import asyncio
import json
from pathlib import Path

from attacks.loader import load_test_cases
from runner.comparison import compute_comparison
from runner.metrics import BaselineMetrics, compute_baseline_metrics
from runner.runner import run_tests

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
GUARDED_PATH = RESULTS_DIR / "guarded.json"
COMPARISON_PATH = RESULTS_DIR / "comparison.json"


async def main() -> None:
    with open(BASELINE_PATH) as f:
        baseline_data = json.load(f)
    baseline_metrics = BaselineMetrics.model_validate(baseline_data["metrics"])

    test_cases = load_test_cases()
    print(f"Running guarded benchmark: {len(test_cases)} test cases, guardrails enabled.")
    guarded_records = await run_tests(test_cases, guardrails_enabled=True)
    guarded_metrics = compute_baseline_metrics(guarded_records, test_cases, guardrails_enabled=True)

    with open(GUARDED_PATH, "w") as f:
        json.dump(
            {
                "metrics": guarded_metrics.model_dump(),
                "records": [r.model_dump() for r in guarded_records],
            },
            f,
            indent=2,
        )

    comparison = compute_comparison(baseline_metrics, guarded_metrics)
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison.model_dump(), f, indent=2)

    print()
    print(f"Overall ASR: baseline {comparison.baseline.overall_asr:.1%} -> "
          f"guarded {comparison.guarded.overall_asr:.1%} "
          f"(relative reduction: {comparison.overall_relative_asr_reduction:.1%})")
    print(f"Tool misuse rate: baseline {comparison.tool_misuse_rate_before:.1%} -> "
          f"guarded {comparison.tool_misuse_rate_after:.1%}")
    print(f"Benign task success: baseline {comparison.benign_success_before:.1%} -> "
          f"guarded {comparison.benign_success_after:.1%}")
    print()
    print(f"{'Category':<25}{'Baseline':>10}{'Guarded':>10}{'Rel. reduction':>16}")
    for category, comp in sorted(comparison.category_comparison.items()):
        print(
            f"{category:<25}{comp.baseline_asr:>9.1%} {comp.guarded_asr:>9.1%} "
            f"{comp.relative_reduction:>15.1%}"
        )

    print()
    print(f"Saved guarded run to {GUARDED_PATH}")
    print(f"Saved comparison to {COMPARISON_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
