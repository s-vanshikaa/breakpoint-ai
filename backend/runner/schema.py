from pydantic import BaseModel

from targets.rag_assistant.assistant import RetrievedChunk
from targets.tool_agent.schema import ToolCall, ToolExecutionResult


class TestRecord(BaseModel):
    __test__ = False  # not a pytest test class

    test_id: str
    category: str
    target: str
    prompt: str
    response: str
    retrieved_context: list[RetrievedChunk] | None = None
    requested_tool_call: ToolCall | None = None
    executed_tool_call: ToolExecutionResult | None = None
    passed: bool
    reason: str
    latency_ms: float
    guardrails_enabled: bool
    block_reason: str | None = None
    # Wall-clock bounds of this evaluation's actual work (after any concurrency wait). Absent
    # (None) in results produced before per-record timing existed.
    started_at: str | None = None
    completed_at: str | None = None


class CategorySummary(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float


class RunSummary(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    guardrails_enabled: bool
    category_breakdown: dict[str, CategorySummary]
