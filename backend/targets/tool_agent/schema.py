from pydantic import BaseModel


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, str] = {}


class ToolExecutionResult(BaseModel):
    tool: str
    arguments: dict[str, str]
    result: str
    permission: str


class AgentResult(BaseModel):
    response: str
    requested_tool_call: ToolCall | None
    executed_tool_call: ToolExecutionResult | None
    model: str
    latency_ms: float
    block_reason: str | None = None
    # Summed across the decision call and, if one happened, the tool-result follow-up call.
    # 0 when a guardrail blocked the request before any model call was attempted.
    model_attempts: int = 0
    model_retried: bool = False
