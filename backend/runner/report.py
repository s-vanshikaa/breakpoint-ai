"""Plain-text reports for benchmark runs. Pure formatting: no I/O, no LLM calls."""

from collections import Counter

from attacks.schema import TestCase
from runner.aggregate import BASELINE, GUARDED, ExperimentAggregate, RateSummary
from runner.comparison import ComparisonReport
from runner.metrics import BENIGN_CATEGORY, BaselineMetrics
from runner.schema import TestRecord


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.1f} pp"


def format_dataset_summary(test_cases: list[TestCase]) -> str:
    by_category = Counter(tc.category.value for tc in test_cases)
    by_target = Counter(tc.target.value for tc in test_cases)
    lines = [f"Total cases: {len(test_cases)}", "Cases by category:"]
    lines += [f"  {name:<24}{count:>3}" for name, count in sorted(by_category.items())]
    lines.append("Cases by target:")
    lines += [f"  {name:<24}{count:>3}" for name, count in sorted(by_target.items())]
    return "\n".join(lines)


def _attack_counts(records: list[TestRecord]) -> dict[str, tuple[int, int]]:
    """category -> (attacks that succeeded, total attacks), adversarial categories only."""
    counts: dict[str, list[int]] = {}
    for r in records:
        if r.category == BENIGN_CATEGORY:
            continue
        succeeded, total = counts.setdefault(r.category, [0, 0])
        counts[r.category] = [succeeded + (not r.passed), total + 1]
    return {k: (v[0], v[1]) for k, v in counts.items()}


def format_run_report(
    name: str, metrics: BaselineMetrics, records: list[TestRecord], meta: dict | None = None
) -> str:
    counts = _attack_counts(records)
    attacks_succeeded = sum(s for s, _ in counts.values())
    attacks_total = sum(t for _, t in counts.values())
    benign = [r for r in records if r.category == BENIGN_CATEGORY]
    guard = "ON" if metrics.guardrails_enabled else "OFF"

    header = f"{name.capitalize()} run: {metrics.total_tests} cases, guardrails {guard}"
    if meta:
        header += f", model {meta.get('model')}, seed {meta.get('seed')}"
    lines = [
        header,
        f"  Overall attack success rate: {pct(metrics.overall_asr)} "
        f"({attacks_succeeded}/{attacks_total} attacks succeeded)",
        f"  Tool misuse rate:            {pct(metrics.tool_misuse_rate)}",
        f"  Benign task success:         {pct(metrics.benign_success_rate)} "
        f"({sum(r.passed for r in benign)}/{len(benign)})",
        "  Attack success by category:",
    ]
    for category, asr in sorted(metrics.asr_by_category.items()):
        succeeded, total = counts.get(category, (0, 0))
        lines.append(f"    {category:<24}{pct(asr):>7}  ({succeeded}/{total})")
    return "\n".join(lines)


def format_comparison_report(
    comparison: ComparisonReport,
    baseline_records: list[TestRecord],
    guarded_records: list[TestRecord],
) -> str:
    b, g = comparison.baseline, comparison.guarded
    counts = _attack_counts(baseline_records)
    lines = [
        f"Baseline vs guarded ({b.total_tests} cases each)",
        "",
        f"{'Metric':<30}{'Baseline':>10}{'Guarded':>10}{'Change':>12}",
        f"{'Overall attack success rate':<30}{pct(b.overall_asr):>10}{pct(g.overall_asr):>10}"
        f"{_pp(g.overall_asr - b.overall_asr):>12}",
        f"{'Tool misuse rate':<30}{pct(comparison.tool_misuse_rate_before):>10}"
        f"{pct(comparison.tool_misuse_rate_after):>10}"
        f"{_pp(comparison.tool_misuse_rate_after - comparison.tool_misuse_rate_before):>12}",
        f"{'Benign task success':<30}{pct(comparison.benign_success_before):>10}"
        f"{pct(comparison.benign_success_after):>10}"
        f"{_pp(comparison.benign_success_after - comparison.benign_success_before):>12}",
        "",
        f"Relative reduction in attack success rate: "
        f"{pct(comparison.overall_relative_asr_reduction)}",
        "",
        f"{'Attack category':<26}{'Attacks':>8}{'Baseline':>10}{'Guarded':>10}{'Reduction':>11}",
    ]
    for category, comp in sorted(comparison.category_comparison.items()):
        total = counts.get(category, (0, 0))[1]
        lines.append(
            f"{category:<26}{total:>8}{pct(comp.baseline_asr):>10}{pct(comp.guarded_asr):>10}"
            f"{pct(comp.relative_reduction):>11}"
        )

    guarded_by_id = {r.test_id: r for r in guarded_records}
    still_succeeding = [
        r.test_id for r in guarded_records if r.category != BENIGN_CATEGORY and not r.passed
    ]
    benign_broken = [
        r.test_id
        for r in baseline_records
        if r.category == BENIGN_CATEGORY and r.passed and not guarded_by_id[r.test_id].passed
    ]
    lines += [
        "",
        f"Attacks still succeeding with guardrails ({len(still_succeeding)}): "
        f"{', '.join(still_succeeding) or 'none'}",
        f"Benign tests broken by guardrails ({len(benign_broken)}): "
        f"{', '.join(benign_broken) or 'none'}",
    ]
    return "\n".join(lines)


def _mean_std(rate: RateSummary) -> str:
    return f"{pct(rate.mean)} +/- {pct(rate.std)}"


def format_experiment_report(agg: ExperimentAggregate) -> str:
    lines = [
        f"Experiment {agg.experiment_id}: {agg.total_evaluations} evaluations, "
        f"{agg.trials_completed}/{agg.trials_planned} trials completed "
        f"(seeds {', '.join(map(str, agg.seeds))})",
    ]
    for failed in agg.failed_trials:
        lines.append(f"  FAILED trial {failed.config} seed {failed.seed}: {failed.error}")

    lines += ["", f"{'':<25}{'Baseline':>20}{'Guarded':>20}"]
    base, guard = agg.configs[BASELINE], agg.configs[GUARDED]
    rows = (
        ("Attack success (mean)", base.asr, guard.asr),
        ("Tool misuse (mean)", base.tool_misuse_rate, guard.tool_misuse_rate),
        ("Benign success (mean)", base.benign_success_rate, guard.benign_success_rate),
    )
    for label, b, g in rows:
        lines.append(f"{label:<25}{_mean_std(b):>20}{_mean_std(g):>20}")
    lines.append(
        f"{'Attack success (pooled)':<25}{pct(base.asr.pooled):>20}{pct(guard.asr.pooled):>20}"
    )
    lines.append(
        f"{'Latency p50 / p95':<25}"
        f"{f'{base.latency.p50_ms or 0:.0f} / {base.latency.p95_ms or 0:.0f} ms':>20}"
        f"{f'{guard.latency.p50_ms or 0:.0f} / {guard.latency.p95_ms or 0:.0f} ms':>20}"
    )

    comp = agg.comparison
    if comp is None:
        lines += ["", "No comparison: both configurations need at least one completed trial."]
        return "\n".join(lines)

    lines += [
        "",
        f"Attack success reduction: {_pp(comp.absolute_asr_reduction or 0)} absolute, "
        f"{pct(comp.relative_asr_reduction)} relative",
        "",
        f"{'Category':<26}{'Baseline':>10}{'Guarded':>10}{'Rel. red.':>11}",
    ]
    for category, delta in comp.by_category.items():
        lines.append(
            f"{category:<26}{pct(delta.baseline_asr):>10}{pct(delta.guarded_asr):>10}"
            f"{pct(delta.relative_reduction):>11}"
        )
    return "\n".join(lines)
