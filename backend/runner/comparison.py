from pydantic import BaseModel

from runner.metrics import BaselineMetrics


class CategoryComparison(BaseModel):
    baseline_asr: float
    guarded_asr: float
    absolute_reduction: float
    relative_reduction: float


class ComparisonReport(BaseModel):
    baseline: BaselineMetrics
    guarded: BaselineMetrics
    overall_relative_asr_reduction: float
    category_comparison: dict[str, CategoryComparison]
    tool_misuse_rate_before: float
    tool_misuse_rate_after: float
    benign_success_before: float
    benign_success_after: float


def _relative_reduction(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before


def compute_comparison(
    baseline_metrics: BaselineMetrics, guarded_metrics: BaselineMetrics
) -> ComparisonReport:
    category_comparison = {}
    categories = set(baseline_metrics.asr_by_category) | set(guarded_metrics.asr_by_category)
    for category in categories:
        baseline_asr = baseline_metrics.asr_by_category.get(category, 0.0)
        guarded_asr = guarded_metrics.asr_by_category.get(category, 0.0)
        category_comparison[category] = CategoryComparison(
            baseline_asr=baseline_asr,
            guarded_asr=guarded_asr,
            absolute_reduction=baseline_asr - guarded_asr,
            relative_reduction=_relative_reduction(baseline_asr, guarded_asr),
        )

    return ComparisonReport(
        baseline=baseline_metrics,
        guarded=guarded_metrics,
        overall_relative_asr_reduction=_relative_reduction(
            baseline_metrics.overall_asr, guarded_metrics.overall_asr
        ),
        category_comparison=category_comparison,
        tool_misuse_rate_before=baseline_metrics.tool_misuse_rate,
        tool_misuse_rate_after=guarded_metrics.tool_misuse_rate,
        benign_success_before=baseline_metrics.benign_success_rate,
        benign_success_after=guarded_metrics.benign_success_rate,
    )
