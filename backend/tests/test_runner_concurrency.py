"""Concurrency guarantees of runner.run_tests: bounded parallelism, stable ordering, and
failure isolation. run_test_case itself is mocked (it owns the real target/model calls,
which are covered elsewhere), so these tests exercise only the scheduling logic.
"""

import asyncio

import pytest

from attacks.schema import AttackCategory
from runner import runner as runner_module
from tests.conftest import make_case, make_record


def cases(n: int = 6) -> list:
    return [make_case(id=f"case_{i:03d}", category=AttackCategory.direct_injection) for i in range(n)]


class ConcurrencyTracker:
    """Fake run_test_case: tracks overlap, records call/completion order, can inject failures."""

    def __init__(self, delay: float = 0.01, fail_ids: frozenset[str] = frozenset()):
        self.delay = delay
        self.fail_ids = fail_ids
        self.active = 0
        self.max_active = 0
        self.call_order: list[str] = []
        self.completion_order: list[str] = []
        self.call_count: dict[str, int] = {}

    async def run_test_case(self, test_case, rag_assistant, tool_agent, guardrails_enabled=False):
        self.call_order.append(test_case.id)
        self.call_count[test_case.id] = self.call_count.get(test_case.id, 0) + 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if test_case.id in self.fail_ids:
                raise RuntimeError(f"{test_case.id} exploded")
            return make_record(test_case.id, test_case.category.value, True)
        finally:
            self.active -= 1
            self.completion_order.append(test_case.id)


@pytest.fixture(autouse=True)
def patch_run_test_case(monkeypatch):
    """Swaps in the tracker for the module-level run_test_case that run_tests calls."""
    tracker = ConcurrencyTracker()

    async def call(test_case, rag_assistant, tool_agent, guardrails_enabled=False):
        return await tracker.run_test_case(test_case, rag_assistant, tool_agent, guardrails_enabled)

    monkeypatch.setattr(runner_module, "run_test_case", call)
    return tracker


def run(test_cases, concurrency=1, on_record=None):
    return asyncio.run(
        runner_module.run_tests(test_cases, on_record=on_record, concurrency=concurrency)
    )


class TestConcurrencyBound:
    def test_concurrency_one_never_overlaps(self, patch_run_test_case):
        run(cases(5), concurrency=1)
        assert patch_run_test_case.max_active == 1

    def test_concurrency_respects_the_configured_maximum(self, patch_run_test_case):
        run(cases(10), concurrency=3)
        assert patch_run_test_case.max_active == 3

    def test_concurrency_higher_than_dataset_size_is_fine(self, patch_run_test_case):
        run(cases(3), concurrency=8)
        assert patch_run_test_case.max_active == 3

    def test_rejects_non_positive_concurrency(self):
        with pytest.raises(ValueError, match="at least 1"):
            run(cases(1), concurrency=0)


class TestEveryCaseRunsExactlyOnce:
    def test_no_case_skipped_or_duplicated(self, patch_run_test_case):
        run(cases(9), concurrency=4)
        assert sorted(patch_run_test_case.call_count) == [c.id for c in cases(9)]
        assert all(n == 1 for n in patch_run_test_case.call_count.values())


class TestStableOrdering:
    def test_output_order_matches_input_order_regardless_of_concurrency(
        self, patch_run_test_case
    ):
        test_cases = cases(12)
        records = run(test_cases, concurrency=5)
        assert [r.test_id for r in records] == [c.id for c in test_cases]

    def test_output_order_is_stable_even_when_completion_order_is_not(self, monkeypatch):
        # Later-scheduled cases finish first, so completion order differs from input order.
        async def call(test_case, rag_assistant, tool_agent, guardrails_enabled=False):
            # Reverse the delay so case_000 finishes last.
            delay = 0.02 if test_case.id == "case_000" else 0.0
            await asyncio.sleep(delay)
            return make_record(test_case.id, test_case.category.value, True)

        monkeypatch.setattr(runner_module, "run_test_case", call)
        test_cases = cases(4)
        records = run(test_cases, concurrency=4)
        assert [r.test_id for r in records] == [c.id for c in test_cases]

    def test_progress_callback_fires_once_per_case(self, patch_run_test_case):
        seen = []
        run(cases(5), concurrency=2, on_record=lambda i, total, r: seen.append((i, total, r.test_id)))
        assert len(seen) == 5
        assert [i for i, _, _ in seen] == [1, 2, 3, 4, 5]
        assert {rid for _, _, rid in seen} == {c.id for c in cases(5)}


class TestFailureIsolation:
    def test_one_failure_does_not_stop_unrelated_jobs(self, monkeypatch):
        tracker = ConcurrencyTracker(delay=0.01, fail_ids=frozenset({"case_002"}))
        monkeypatch.setattr(
            runner_module,
            "run_test_case",
            lambda tc, ra, ta, guardrails_enabled=False: tracker.run_test_case(
                tc, ra, ta, guardrails_enabled
            ),
        )
        test_cases = cases(6)
        with pytest.raises(RuntimeError, match="case_002 exploded"):
            run(test_cases, concurrency=3)
        # every case still ran to completion, including ones scheduled after the failure
        assert sorted(tracker.call_count) == [c.id for c in test_cases]
        assert all(n == 1 for n in tracker.call_count.values())

    def test_multiple_failures_all_run_and_first_is_raised(self, monkeypatch):
        tracker = ConcurrencyTracker(delay=0.01, fail_ids=frozenset({"case_001", "case_004"}))
        monkeypatch.setattr(
            runner_module,
            "run_test_case",
            lambda tc, ra, ta, guardrails_enabled=False: tracker.run_test_case(
                tc, ra, ta, guardrails_enabled
            ),
        )
        test_cases = cases(6)
        with pytest.raises(RuntimeError):
            run(test_cases, concurrency=6)
        assert len(tracker.call_count) == 6
