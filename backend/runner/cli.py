"""BreakPoint AI benchmark CLI.

    python -m runner validate     # check the dataset (no LLM needed)
    python -m runner baseline     # run all cases with guardrails off
    python -m runner guarded      # run all cases with guardrails on
    python -m runner compare      # compare saved baseline/guarded runs (no LLM needed)
    python -m runner all          # baseline + guarded + compare
    python -m runner run          # ad hoc: one test / one category
    python -m runner experiment   # baseline + guarded over several seeds, with aggregate metrics
"""

import argparse
import json
from pathlib import Path

from attacks.loader import DatasetError, filter_by_category, load_test_cases
from attacks.schema import AttackCategory
from attacks.validate import find_problems
from config import settings
from models.ollama_client import ollama_client
from runner import experiment, report, workflow
from runner.common import ensure_ollama_ready, run_cli
from runner.runner import run_tests, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m runner", description="Run the BreakPoint AI benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_results_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--results-dir",
            type=Path,
            default=None,
            help=f"Where result files are read/written (default: {settings.results_dir}).",
        )

    def add_seed(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Fix the model's sampling seed to reduce run-to-run variance "
                "(default: unseeded). Local inference can still differ slightly between runs."
            ),
        )

    sub.add_parser("validate", help="Validate the benchmark dataset (no LLM needed).")

    for name, help_text in (
        ("baseline", "Run every case with guardrails OFF and save baseline.json."),
        ("guarded", "Run every case with guardrails ON and save guarded.json."),
        ("all", "Run baseline, guarded, then compare."),
    ):
        p = sub.add_parser(name, help=help_text)
        add_seed(p)
        add_results_dir(p)

    p = sub.add_parser(
        "compare", help="Compare saved baseline/guarded runs and save comparison.json."
    )
    add_results_dir(p)

    p = sub.add_parser(
        "experiment",
        help="Run baseline and guarded for several seeds and aggregate the trials.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="One trial per seed and configuration (default: 1 2 3 4 5).",
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help="Name for the output folder (default: exp-<UTC timestamp>).",
    )
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help=f"Where experiments are written (default: {settings.experiments_dir}).",
    )

    p = sub.add_parser("run", help="Ad hoc run of one test or one category; nothing is saved.")
    p.add_argument("--test-id", help="Run a single test case by ID.")
    p.add_argument("--category", choices=[c.value for c in AttackCategory])
    p.add_argument("--guardrails", action="store_true", help="Enable guardrails.")
    p.add_argument("--output", type=Path, help="Optional path to save the raw records as JSON.")
    add_seed(p)
    return parser


def _cmd_validate() -> None:
    test_cases = load_test_cases()
    problems = find_problems(test_cases)
    print(report.format_dataset_summary(test_cases))
    if problems:
        print()
        for problem in problems:
            print(f"  - {problem}")
        raise DatasetError(f"{len(problems)} dataset problem(s) found.")
    print("\nDataset OK.")


async def _cmd_run_benchmark(names: list[str], args: argparse.Namespace) -> None:
    test_cases = load_test_cases()
    results_dir = args.results_dir or settings.results_dir
    for name in names:
        stored = await workflow.execute_run(name, test_cases, results_dir, seed=args.seed)
        print()
        print(report.format_run_report(name, stored.metrics, stored.records, stored.meta))
        print()


def _cmd_compare(args: argparse.Namespace) -> None:
    results_dir = args.results_dir or settings.results_dir
    comparison, baseline, guarded = workflow.compare_saved_runs(results_dir)
    print()
    print(report.format_comparison_report(comparison, baseline.records, guarded.records))


async def _cmd_experiment(args: argparse.Namespace) -> None:
    _, aggregate = await experiment.run_experiment(
        load_test_cases(),
        args.seeds,
        args.experiments_dir or settings.experiments_dir,
        experiment_id=args.experiment_id,
    )
    print()
    print(report.format_experiment_report(aggregate))


async def _cmd_run_adhoc(args: argparse.Namespace) -> None:
    cases = load_test_cases()
    if args.test_id:
        cases = [c for c in cases if c.id == args.test_id]
        if not cases:
            raise DatasetError(f"No test case found with id '{args.test_id}'.")
    elif args.category:
        cases = filter_by_category(cases, AttackCategory(args.category))

    await ensure_ollama_ready()
    ollama_client.default_options = {"seed": args.seed} if args.seed is not None else {}
    print(f"Running {len(cases)} test case(s) (guardrails_enabled={args.guardrails})...")
    records = await run_tests(cases, args.guardrails, on_record=workflow.print_progress)
    summary = summarize(records, guardrails_enabled=args.guardrails)

    print(
        f"\nTotal: {summary.total}  Passed: {summary.passed}  "
        f"Failed: {summary.failed}  Pass rate: {report.pct(summary.pass_rate)}"
    )
    print("\nBy category:")
    for category, cat in sorted(summary.category_breakdown.items()):
        print(f"  {category}: {cat.passed}/{cat.total} passed ({report.pct(cat.pass_rate)})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"summary": summary.model_dump(), "records": [r.model_dump() for r in records]},
                indent=2,
            )
        )
        print(f"\nSaved results to {args.output}")


async def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "validate":
        _cmd_validate()
    elif args.command in ("baseline", "guarded"):
        await _cmd_run_benchmark([args.command], args)
    elif args.command == "all":
        await _cmd_run_benchmark(["baseline", "guarded"], args)
        _cmd_compare(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "experiment":
        await _cmd_experiment(args)
    else:
        await _cmd_run_adhoc(args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_cli(lambda: _dispatch(args))


if __name__ == "__main__":
    main()
