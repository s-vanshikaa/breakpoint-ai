"""Shared factories. Everything here is deterministic and never touches Ollama."""

import pytest

from attacks.schema import AttackCategory, TargetApp, TestCase
from runner.schema import TestRecord
from targets.tool_agent.schema import ToolExecutionResult

SECRET = "BREAKPOINT_INTERNAL_TOKEN_4821"


def make_case(**overrides) -> TestCase:
    fields = {
        "id": "case_001",
        "category": AttackCategory.direct_injection,
        "target": TargetApp.rag_assistant,
        "prompt": "prompt",
        "expected": "must_not_leak_secret",
        "protected_value": SECRET,
    }
    fields.update(overrides)
    return TestCase(**fields)


def make_record(test_id: str, category: str, passed: bool, **overrides) -> TestRecord:
    fields = {
        "test_id": test_id,
        "category": category,
        "target": "rag_assistant",
        "prompt": "p",
        "response": "r",
        "passed": passed,
        "reason": "",
        "latency_ms": 1.0,
        "guardrails_enabled": False,
    }
    fields.update(overrides)
    return TestRecord(**fields)


def executed(tool: str, **arguments: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool=tool, arguments=arguments, result="ok", permission="allowed"
    )


@pytest.fixture
def secret() -> str:
    return SECRET
