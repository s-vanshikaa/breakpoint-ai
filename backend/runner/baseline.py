import json

from attacks.loader import load_test_cases
from config import settings
from runner.common import ensure_ollama_ready, run_cli
from runner.metrics import compute_baseline_metrics
from runner.runner import run_tests

OUTPUT_PATH = settings.results_dir / "baseline.json"


async def main() -> None:
    test_cases = load_test_cases()
    await ensure_ollama_ready()

    print(f"Running baseline benchmark: {len(test_cases)} test cases, guardrails disabled.")
    records = await run_tests(test_cases, guardrails_enabled=False)
    metrics = compute_baseline_metrics(records, test_cases, guardrails_enabled=False)

    print()
    print(f"Overall attack success rate (ASR): {metrics.overall_asr:.1%}")
    print(f"Tool misuse rate: {metrics.tool_misuse_rate:.1%}")
    print(f"Benign task success rate: {metrics.benign_success_rate:.1%}")
    print()
    print("ASR by category:")
    for category, asr in sorted(metrics.asr_by_category.items()):
        print(f"  {category}: {asr:.1%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metrics": metrics.model_dump(),
        "records": [r.model_dump() for r in records],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Saved baseline to {OUTPUT_PATH}")
    print(
        f"Note: results come from a live local LLM ({settings.ollama_model}) and are "
        "not bit-for-bit deterministic across runs, even with low temperature on "
        "tool decisions. 'Reproducible' here means the benchmark command and "
        "test set are fixed and re-runnable, not that outputs are frozen."
    )


if __name__ == "__main__":
    run_cli(main)
