"""Pure aggregation of multi-trial experiment results. No I/O, no LLM.

A "trial" is one full benchmark run of one configuration (baseline or guarded) under one seed.
Rates whose denominator is zero are reported as None (undefined), never as 0.0, so an empty
category can't be mistaken for a perfect one.

Outcome metrics (attack success, tool misuse, benign success, and their category/comparison
derivatives) are computed only from records with `status == "ok"`. A timeout, model failure,
malformed response or evaluator bug never produced a real verdict, so it is excluded from those
numerators and denominators rather than being counted as either a successful attack or a
successful defense. Coverage of how much was actually evaluated is reported explicitly via each
ConfigAggregate's total_evaluations/valid_evaluations/failed_evaluations/failures_by_status.
"""

import statistics
from collections import Counter, defaultdict

from pydantic import BaseModel

from attacks.schema import TestCase
from runner.metrics import BENIGN_CATEGORY
from runner.schema import EXECUTION_STATUSES, TestRecord

BASELINE = "baseline"
GUARDED = "guarded"
CONFIGS = (BASELINE, GUARDED)

# Every non-"ok" status a TestRecord can carry (see runner.schema.EXECUTION_STATUSES).
FAILURE_STATUSES = tuple(s for s in EXECUTION_STATUSES if s != "ok")


class Trial(BaseModel):
    """A trial as seen by the aggregator: identity, outcome, and its records."""

    config: str
    seed: int
    status: str = "completed"  # "completed" | "failed"
    error: str | None = None
    records: list[TestRecord] = []


class RateSummary(BaseModel):
    numerator: int  # events pooled across trials
    denominator: int
    pooled: float | None  # numerator / denominator across every trial
    mean: float | None  # mean of the per-trial rates
    std: float | None  # sample standard deviation (n-1) of per-trial rates; 0.0 for one trial
    per_trial: dict[str, float | None]  # keyed by seed


class LatencySummary(BaseModel):
    count: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


class ConfigAggregate(BaseModel):
    trials_completed: int
    total_evaluations: int  # every record, regardless of status - a scale/throughput count
    valid_evaluations: int  # status == "ok" - eligible for the outcome metrics below
    failed_evaluations: int  # total_evaluations - valid_evaluations
    valid_evaluation_rate: float | None  # valid_evaluations / total_evaluations
    failures_by_status: dict[str, int]  # timeout/model_failure/invalid_response/evaluator_failure
    asr: RateSummary
    asr_by_category: dict[str, RateSummary]
    tool_misuse_rate: RateSummary
    benign_success_rate: RateSummary
    latency: LatencySummary  # computed over every record; a timeout's latency is still real data


class CategoryDelta(BaseModel):
    baseline_asr: float | None
    guarded_asr: float | None
    absolute_reduction: float | None  # baseline - guarded, as a fraction (0.2 = 20 points)
    relative_reduction: float | None  # (baseline - guarded) / baseline


class Comparison(BaseModel):
    baseline_asr: float | None  # mean across trials
    guarded_asr: float | None
    absolute_asr_reduction: float | None  # percentage points, as a fraction
    relative_asr_reduction: float | None
    baseline_tool_misuse_rate: float | None
    guarded_tool_misuse_rate: float | None
    benign_success_change: float | None  # guarded - baseline
    by_category: dict[str, CategoryDelta]


class FailedTrial(BaseModel):
    config: str
    seed: int
    error: str | None


class ExperimentAggregate(BaseModel):
    experiment_id: str
    trials_planned: int
    trials_completed: int
    trials_failed: int
    total_evaluations: int
    seeds: list[int]
    failed_trials: list[FailedTrial]
    configs: dict[str, ConfigAggregate]
    comparison: Comparison | None  # None unless both configs have a completed trial


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def relative_reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (before - after) / before


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default); q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_latency(values: list[float]) -> LatencySummary:
    return LatencySummary(
        count=len(values),
        mean_ms=mean_or_none(values),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
        p99_ms=percentile(values, 99),
    )


def _rate_summary(per_trial_counts: dict[int, tuple[int, int]]) -> RateSummary:
    """per_trial_counts maps seed -> (events, denominator)."""
    numerator = sum(n for n, _ in per_trial_counts.values())
    denominator = sum(d for _, d in per_trial_counts.values())
    rates = {seed: ratio(n, d) for seed, (n, d) in per_trial_counts.items()}
    defined = [r for r in rates.values() if r is not None]
    return RateSummary(
        numerator=numerator,
        denominator=denominator,
        pooled=ratio(numerator, denominator),
        mean=mean_or_none(defined),
        std=std_or_none(defined),
        per_trial={str(seed): rate for seed, rate in rates.items()},
    )


def _trial_counts(records: list[TestRecord], cases_by_id: dict[str, TestCase]) -> dict:
    """Numerators/denominators for one trial, using the same definitions as runner.metrics.

    Outcome numerators/denominators (asr/tool/benign/categories) are computed only from
    `status == "ok"` records - see the module docstring for why. `total`/`valid`/`failures`
    describe coverage of the trial itself and count every record regardless of status.
    """
    valid = [r for r in records if r.status == "ok"]
    adversarial = [r for r in valid if r.category != BENIGN_CATEGORY]
    benign = [r for r in valid if r.category == BENIGN_CATEGORY]
    tool_cases = [r for r in valid if cases_by_id[r.test_id].forbidden_tool is not None]

    by_category: dict[str, list[TestRecord]] = defaultdict(list)
    for r in adversarial:
        by_category[r.category].append(r)

    return {
        "asr": (sum(1 for r in adversarial if not r.passed), len(adversarial)),
        "tool": (sum(1 for r in tool_cases if not r.passed), len(tool_cases)),
        "benign": (sum(1 for r in benign if r.passed), len(benign)),
        "categories": {
            c: (sum(1 for r in recs if not r.passed), len(recs)) for c, recs in by_category.items()
        },
        "total": len(records),
        "valid": len(valid),
        "failures_by_status": Counter(r.status for r in records if r.status != "ok"),
    }


def aggregate_config(trials: list[Trial], cases_by_id: dict[str, TestCase]) -> ConfigAggregate:
    """Aggregates the completed trials of one configuration, in seed order."""
    ordered = sorted((t for t in trials if t.status == "completed"), key=lambda t: t.seed)
    counts = {t.seed: _trial_counts(t.records, cases_by_id) for t in ordered}

    categories = sorted({c for tc in counts.values() for c in tc["categories"]})
    latencies = [r.latency_ms for t in ordered for r in t.records]

    total_evaluations = sum(tc["total"] for tc in counts.values())
    valid_evaluations = sum(tc["valid"] for tc in counts.values())
    failures_by_status: Counter = Counter()
    for tc in counts.values():
        failures_by_status.update(tc["failures_by_status"])

    return ConfigAggregate(
        trials_completed=len(ordered),
        total_evaluations=total_evaluations,
        valid_evaluations=valid_evaluations,
        failed_evaluations=total_evaluations - valid_evaluations,
        valid_evaluation_rate=ratio(valid_evaluations, total_evaluations),
        failures_by_status={s: failures_by_status.get(s, 0) for s in FAILURE_STATUSES},
        asr=_rate_summary({s: c["asr"] for s, c in counts.items()}),
        asr_by_category={
            cat: _rate_summary({s: c["categories"].get(cat, (0, 0)) for s, c in counts.items()})
            for cat in categories
        },
        tool_misuse_rate=_rate_summary({s: c["tool"] for s, c in counts.items()}),
        benign_success_rate=_rate_summary({s: c["benign"] for s, c in counts.items()}),
        latency=summarize_latency(latencies),
    )


def compare_configs(baseline: ConfigAggregate, guarded: ConfigAggregate) -> Comparison:
    b_asr, g_asr = baseline.asr.mean, guarded.asr.mean
    categories = sorted(set(baseline.asr_by_category) | set(guarded.asr_by_category))

    def cat_mean(agg: ConfigAggregate, cat: str) -> float | None:
        summary = agg.asr_by_category.get(cat)
        return summary.mean if summary else None

    by_category = {}
    for cat in categories:
        b, g = cat_mean(baseline, cat), cat_mean(guarded, cat)
        by_category[cat] = CategoryDelta(
            baseline_asr=b,
            guarded_asr=g,
            absolute_reduction=None if b is None or g is None else b - g,
            relative_reduction=relative_reduction(b, g),
        )

    b_benign, g_benign = baseline.benign_success_rate.mean, guarded.benign_success_rate.mean
    return Comparison(
        baseline_asr=b_asr,
        guarded_asr=g_asr,
        absolute_asr_reduction=None if b_asr is None or g_asr is None else b_asr - g_asr,
        relative_asr_reduction=relative_reduction(b_asr, g_asr),
        baseline_tool_misuse_rate=baseline.tool_misuse_rate.mean,
        guarded_tool_misuse_rate=guarded.tool_misuse_rate.mean,
        benign_success_change=(
            None if b_benign is None or g_benign is None else g_benign - b_benign
        ),
        by_category=by_category,
    )


def aggregate_experiment(
    experiment_id: str,
    trials: list[Trial],
    test_cases: list[TestCase],
    planned: int | None = None,
) -> ExperimentAggregate:
    """Builds the experiment aggregate. Failed trials are listed, never silently dropped.

    `planned` is the number of trials the experiment set out to run (defaults to len(trials));
    trials that never ran count as neither completed nor failed.
    """
    cases_by_id = {tc.id: tc for tc in test_cases}
    completed = [t for t in trials if t.status == "completed"]
    failed = sorted(
        (t for t in trials if t.status == "failed"), key=lambda t: (t.seed, t.config)
    )

    configs = {
        name: aggregate_config([t for t in trials if t.config == name], cases_by_id)
        for name in CONFIGS
    }
    both = all(configs[name].trials_completed for name in CONFIGS)

    return ExperimentAggregate(
        experiment_id=experiment_id,
        trials_planned=planned if planned is not None else len(trials),
        trials_completed=len(completed),
        trials_failed=len(failed),
        total_evaluations=sum(len(t.records) for t in completed),
        seeds=sorted({t.seed for t in trials}),
        failed_trials=[FailedTrial(config=t.config, seed=t.seed, error=t.error) for t in failed],
        configs=configs,
        comparison=compare_configs(configs[BASELINE], configs[GUARDED]) if both else None,
    )
