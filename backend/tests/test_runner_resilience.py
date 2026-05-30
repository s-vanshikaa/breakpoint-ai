"""Commit 3: run_test_case turns model-call and evaluator failures into structured TestRecords
instead of raising, so run_tests never needs to abort a run over one bad case. RAGAssistant and
ToolAgent are faked here (their own retry plumbing is covered in test_ollama_client.py); these
tests exercise only run_test_case's classification and record-building.
"""

import asyncio

import pytest

from attacks.schema import AttackCategory, TargetApp
from models.ollama_client import (
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from runner.runner import run_test_case, status_counts
from targets.rag_assistant.assistant import RAGResult
from tests.conftest import make_case


def raising_error(error):
    async def answer(prompt, guardrails_enabled=False):
        raise error

    return answer


class FakeAssistant:
    def __init__(self, answer_fn):
        self.answer = answer_fn


class FakeToolAgent:
    def __init__(self, handle_fn):
        self.handle = handle_fn


def ok_result(**overrides):
    fields = {
        "response": "hi",
        "retrieved_chunks": [],
        "model": "m",
        "latency_ms": 1.0,
        "model_attempts": 1,
        "model_retried": False,
    }
    fields.update(overrides)
    return RAGResult(**fields)


def run_one(case, assistant=None, agent=None):
    return asyncio.run(run_test_case(case, assistant, agent, guardrails_enabled=False))


class TestModelFailureClassification:
    @pytest.mark.parametrize(
        "error,expected_status",
        [
            (OllamaTimeoutError("slow"), "timeout"),
            (OllamaUnavailableError("down"), "model_failure"),
            (OllamaModelNotFoundError("missing"), "model_failure"),
            (OllamaResponseError("garbage"), "invalid_response"),
        ],
    )
    def test_each_error_type_maps_to_its_category(self, error, expected_status):
        error.attempts = 2
        error.retried = True
        case = make_case()
        record = run_one(case, assistant=FakeAssistant(raising_error(error)))
        assert record.status == expected_status
        assert record.passed is False
        assert record.attempt_count == 2
        assert record.retried is True
        assert str(error) in record.error
        assert record.response == ""

    def test_missing_attempts_on_the_error_defaults_to_one(self):
        error = OllamaTimeoutError("slow")  # attempts/retried never set
        case = make_case()
        record = run_one(case, assistant=FakeAssistant(raising_error(error)))
        assert record.attempt_count == 1
        assert record.retried is False

    def test_tool_agent_model_failure_is_also_classified(self):
        error = OllamaUnavailableError("down")
        error.attempts = 1
        error.retried = False
        case = make_case(
            id="tool_1",
            category=AttackCategory.tool_misuse,
            target=TargetApp.tool_agent,
            protected_value=None,
            forbidden_tool="delete_user",
        )
        record = run_one(case, agent=FakeToolAgent(raising_error(error)))
        assert record.status == "model_failure"
        assert record.target == "tool_agent"


class TestEvaluatorFailureIsolation:
    def test_evaluator_bug_becomes_a_structured_record_not_a_crash(self):
        case = make_case()

        async def answer(prompt, guardrails_enabled=False):
            return ok_result(response="totally fine response")

        # protected_value set but response is fine; force evaluate_test_case to blow up by
        # patching it out from under run_test_case via a case with no gradeable fields... instead
        # directly simulate an evaluator exception through a case whose evaluator path raises.
        import runner.runner as runner_module

        original = runner_module.evaluate_test_case

        def boom(*a, **kw):
            raise ValueError("evaluator exploded")

        runner_module.evaluate_test_case = boom
        try:
            record = run_one(case, assistant=FakeAssistant(answer))
        finally:
            runner_module.evaluate_test_case = original

        assert record.status == "evaluator_failure"
        assert record.passed is False
        assert "evaluator exploded" in record.error
        assert record.response == "totally fine response"  # the model's answer is preserved
        assert record.attempt_count == 1  # taken from the successful model call


class TestSuccessPathAttemptMetadata:
    def test_successful_case_records_attempts_and_retried(self):
        case = make_case()

        async def answer(prompt, guardrails_enabled=False):
            return ok_result(response=case.protected_value + " leaked", model_attempts=2, model_retried=True)

        record = run_one(case, assistant=FakeAssistant(answer))
        assert record.status == "ok"
        assert record.attempt_count == 2
        assert record.retried is True
        assert record.passed is False  # the secret leaked, so this is a real evaluator FAIL


class TestFullRunIsolatesFailures:
    def test_run_tests_completes_all_cases_even_when_some_fail(self, monkeypatch):
        import runner.runner as runner_module

        cases = [make_case(id=f"c{i}") for i in range(5)]
        fail_ids = {"c1", "c3"}

        async def fake_run_test_case(test_case, rag_assistant, tool_agent, guardrails_enabled=False):
            if test_case.id in fail_ids:
                error = OllamaTimeoutError("slow")
                error.attempts = 3
                error.retried = True
                from runner.runner import _failed_record

                return _failed_record(test_case, guardrails_enabled, "timeout", str(error), 5.0, 3, True)
            return ok_record(test_case)

        def ok_record(test_case):
            from runner.schema import TestRecord

            return TestRecord(
                test_id=test_case.id,
                category=test_case.category.value,
                target=test_case.target.value,
                prompt=test_case.prompt,
                response="fine",
                passed=True,
                reason="ok",
                latency_ms=1.0,
                guardrails_enabled=False,
            )

        monkeypatch.setattr(runner_module, "run_test_case", fake_run_test_case)
        records = asyncio.run(runner_module.run_tests(cases, concurrency=3))

        # no exception propagated, every case produced exactly one record, in order
        assert [r.test_id for r in records] == [c.id for c in cases]
        assert status_counts(records) == {"ok": 3, "timeout": 2}
        assert {r.test_id for r in records if r.status == "timeout"} == fail_ids
