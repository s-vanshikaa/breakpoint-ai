import json

import pytest

from attacks.schema import AttackCategory, TargetApp
from config import settings
from runner.comparison import compute_comparison
from runner.metrics import BaselineMetrics, compute_baseline_metrics
from runner.runner import summarize
from tests.conftest import make_case, make_record


def direct(i: int):
    return make_case(id=f"direct_{i}")


def secret(i: int):
    return make_case(id=f"secret_{i}", category=AttackCategory.secret_extraction)


def tool(i: int):
    return make_case(
        id=f"tool_{i}",
        category=AttackCategory.tool_misuse,
        target=TargetApp.tool_agent,
        protected_value=None,
        forbidden_tool="delete_user_account",
        expected="must_not_call",
    )


def benign(i: int):
    return make_case(
        id=f"benign_{i}",
        category=AttackCategory.benign,
        protected_value=None,
        expected_keywords=["x"],
        expected="must_answer_correctly",
    )


def rec(case, passed: bool):
    return make_record(case.id, case.category.value, passed)


def metrics_for(cases_and_passed, guardrails_enabled=False):
    cases = [c for c, _ in cases_and_passed]
    records = [rec(c, p) for c, p in cases_and_passed]
    return compute_baseline_metrics(records, cases, guardrails_enabled=guardrails_enabled)


class TestBaselineMetrics:
    def test_attack_success_rate_counts_failed_adversarial_tests(self):
        # 4 adversarial tests, 1 failed (= attack succeeded).
        m = metrics_for([(direct(1), True), (direct(2), True), (secret(1), False), (tool(1), True)])
        assert m.overall_asr == pytest.approx(0.25)
        assert m.total_tests == 4

    def test_benign_tests_are_excluded_from_asr(self):
        m = metrics_for([(direct(1), True), (benign(1), False), (benign(2), False)])
        assert m.overall_asr == 0.0

    def test_benign_success_rate(self):
        m = metrics_for([(benign(1), True), (benign(2), True), (benign(3), True), (benign(4), False)])
        assert m.benign_success_rate == pytest.approx(0.75)

    def test_adversarial_tests_are_excluded_from_benign_success(self):
        m = metrics_for([(direct(1), False), (benign(1), True)])
        assert m.benign_success_rate == 1.0

    def test_tool_misuse_rate_uses_forbidden_tool_cases_only(self):
        m = metrics_for([(tool(1), False), (tool(2), True), (tool(3), True), (tool(4), True), (direct(1), False)])
        assert m.tool_misuse_rate == pytest.approx(0.25)

    def test_tool_misuse_rate_includes_direct_injection_cases_targeting_tools(self):
        tool_direct = make_case(
            id="direct_tool",
            target=TargetApp.tool_agent,
            protected_value=None,
            forbidden_tool="rotate_api_key",
            expected="must_not_call",
        )
        m = metrics_for([(tool(1), True), (tool_direct, False)])
        assert m.tool_misuse_rate == pytest.approx(0.5)

    def test_per_category_aggregation(self):
        m = metrics_for(
            [
                (direct(1), False),
                (direct(2), True),
                (secret(1), True),
                (secret(2), True),
                (tool(1), False),
                (benign(1), True),
            ]
        )
        assert m.asr_by_category == {
            "direct_injection": pytest.approx(0.5),
            "secret_extraction": 0.0,
            "tool_misuse": 1.0,
        }

    def test_benign_is_not_an_asr_category(self):
        assert "benign" not in metrics_for([(benign(1), True), (direct(1), True)]).asr_by_category

    def test_empty_run_is_all_zero(self):
        m = compute_baseline_metrics([], [])
        assert (m.total_tests, m.overall_asr, m.tool_misuse_rate, m.benign_success_rate) == (0, 0.0, 0.0, 0.0)
        assert m.asr_by_category == {}

    def test_category_without_tool_cases_gives_zero_misuse(self):
        assert metrics_for([(direct(1), False)]).tool_misuse_rate == 0.0

    def test_no_benign_cases_gives_zero_benign_success(self):
        assert metrics_for([(direct(1), True)]).benign_success_rate == 0.0

    def test_guardrails_flag_is_recorded(self):
        assert metrics_for([(direct(1), True)], guardrails_enabled=True).guardrails_enabled is True


def sample_metrics(**overrides) -> BaselineMetrics:
    fields = {
        "total_tests": 10,
        "overall_asr": 0.4,
        "asr_by_category": {"direct_injection": 0.5, "tool_misuse": 0.25},
        "tool_misuse_rate": 0.25,
        "benign_success_rate": 1.0,
        "guardrails_enabled": False,
    }
    fields.update(overrides)
    return BaselineMetrics(**fields)


class TestComparison:
    def test_overall_relative_reduction(self):
        c = compute_comparison(sample_metrics(overall_asr=0.4), sample_metrics(overall_asr=0.1))
        assert c.overall_relative_asr_reduction == pytest.approx(0.75)

    def test_per_category_reduction(self):
        guarded = sample_metrics(asr_by_category={"direct_injection": 0.1, "tool_misuse": 0.0})
        c = compute_comparison(sample_metrics(), guarded)
        di = c.category_comparison["direct_injection"]
        assert di.baseline_asr == 0.5 and di.guarded_asr == 0.1
        assert di.absolute_reduction == pytest.approx(0.4)
        assert di.relative_reduction == pytest.approx(0.8)
        assert c.category_comparison["tool_misuse"].relative_reduction == 1.0

    def test_before_after_fields(self):
        guarded = sample_metrics(tool_misuse_rate=0.0, benign_success_rate=0.9)
        c = compute_comparison(sample_metrics(), guarded)
        assert (c.tool_misuse_rate_before, c.tool_misuse_rate_after) == (0.25, 0.0)
        assert (c.benign_success_before, c.benign_success_after) == (1.0, pytest.approx(0.9))

    def test_zero_baseline_asr_gives_zero_reduction_not_a_crash(self):
        c = compute_comparison(
            sample_metrics(overall_asr=0.0, asr_by_category={"direct_injection": 0.0}),
            sample_metrics(overall_asr=0.0, asr_by_category={"direct_injection": 0.0}),
        )
        assert c.overall_relative_asr_reduction == 0.0
        assert c.category_comparison["direct_injection"].relative_reduction == 0.0

    def test_category_missing_from_one_run_defaults_to_zero(self):
        c = compute_comparison(
            sample_metrics(asr_by_category={"direct_injection": 0.5}),
            sample_metrics(asr_by_category={"secret_extraction": 0.2}),
        )
        assert c.category_comparison["direct_injection"].guarded_asr == 0.0
        assert c.category_comparison["secret_extraction"].baseline_asr == 0.0

    def test_guardrails_making_things_worse_gives_negative_reduction(self):
        c = compute_comparison(sample_metrics(overall_asr=0.1), sample_metrics(overall_asr=0.2))
        assert c.overall_relative_asr_reduction == pytest.approx(-1.0)


class TestSummarize:
    def test_totals_and_category_breakdown(self):
        records = [
            make_record("a", "direct_injection", True),
            make_record("b", "direct_injection", False),
            make_record("c", "benign", True),
        ]
        s = summarize(records, guardrails_enabled=True)
        assert (s.total, s.passed, s.failed) == (3, 2, 1)
        assert s.pass_rate == pytest.approx(2 / 3)
        assert s.guardrails_enabled is True
        assert s.category_breakdown["direct_injection"].pass_rate == 0.5
        assert s.category_breakdown["benign"].total == 1

    def test_empty(self):
        s = summarize([])
        assert (s.total, s.pass_rate, s.category_breakdown) == (0, 0.0, {})


RESULTS = settings.results_dir


def assert_close(actual, expected, path="root"):
    """Recursive comparison with float tolerance (pytest.approx can't nest dicts)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected), path
    else:
        assert actual == expected, path


@pytest.mark.skipif(
    not all((RESULTS / f).exists() for f in ("baseline.json", "guarded.json", "comparison.json")),
    reason="stored benchmark results not present",
)
class TestStoredResultsAreSelfConsistent:
    """The numbers shown in the dashboard/README must be derivable from the stored records."""

    @pytest.mark.parametrize("run", ["baseline", "guarded"])
    def test_stored_metrics_match_recomputed_metrics(self, run):
        from attacks.loader import load_test_cases
        from runner.schema import TestRecord

        data = json.loads((RESULTS / f"{run}.json").read_text())
        records = [TestRecord.model_validate(r) for r in data["records"]]
        recomputed = compute_baseline_metrics(
            records, load_test_cases(), guardrails_enabled=(run == "guarded")
        )
        stored = BaselineMetrics.model_validate(data["metrics"])
        assert_close(recomputed.model_dump(), stored.model_dump())

    def test_comparison_matches_baseline_and_guarded(self):
        load = lambda name: json.loads((RESULTS / name).read_text())  # noqa: E731
        expected = compute_comparison(
            BaselineMetrics.model_validate(load("baseline.json")["metrics"]),
            BaselineMetrics.model_validate(load("guarded.json")["metrics"]),
        )
        assert_close(load("comparison.json"), expected.model_dump())
