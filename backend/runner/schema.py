from pydantic import BaseModel

from targets.rag_assistant.assistant import RetrievedChunk
from targets.tool_agent.schema import ToolCall, ToolExecutionResult

# Execution status for a TestRecord. "ok" means the model/tool pipeline ran and produced a
# response the evaluator could grade (whether or not the attack succeeded). Any other value
# means grading never happened: no response was produced, so `passed` is conservatively False
# and `reason` explains that this wasn't a real evaluator verdict.
EXECUTION_STATUSES = ("ok", "timeout", "model_failure", "invalid_response", "evaluator_failure")


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
    # Fault-tolerance metadata (absent/default in results produced before Commit 3).
    status: str = "ok"  # one of EXECUTION_STATUSES
    attempt_count: int = 1  # total model-call attempts behind this evaluation (0 if none made)
    retried: bool = False
    error: str | None = None  # the exception message, only set when status != "ok"


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
