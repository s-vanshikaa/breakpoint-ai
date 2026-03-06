import argparse
import asyncio
import json

from attacks.loader import load_test_cases
from runner.runner import run_tests, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BreakPoint AI benchmark.")
    parser.add_argument("--test-id", help="Run a single test case by ID.")
    parser.add_argument("--category", help="Run only tests in this category.")
    parser.add_argument(
        "--guardrails",
        action="store_true",
        help="Mark this run as guardrails-enabled (no-op until guardrails exist).",
    )
    parser.add_argument("--output", help="Path to save results as JSON.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    all_cases = load_test_cases()

    if args.test_id:
        cases = [c for c in all_cases if c.id == args.test_id]
        if not cases:
            print(f"No test case found with id '{args.test_id}'")
            return
    elif args.category:
        cases = [c for c in all_cases if c.category.value == args.category]
        if not cases:
            print(f"No test cases found in category '{args.category}'")
            return
    else:
        cases = all_cases

    print(f"Running {len(cases)} test case(s) (guardrails_enabled={args.guardrails})...")
    records = await run_tests(cases, guardrails_enabled=args.guardrails)
    summary = summarize(records, guardrails_enabled=args.guardrails)

    print()
    print(
        f"Total: {summary.total}  Passed: {summary.passed}  "
        f"Failed: {summary.failed}  Pass rate: {summary.pass_rate:.1%}"
    )
    print()
    print("By category:")
    for category, cat_summary in sorted(summary.category_breakdown.items()):
        print(
            f"  {category}: {cat_summary.passed}/{cat_summary.total} passed "
            f"({cat_summary.pass_rate:.1%})"
        )

    if args.output:
        output_data = {
            "summary": summary.model_dump(),
            "records": [r.model_dump() for r in records],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print()
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
