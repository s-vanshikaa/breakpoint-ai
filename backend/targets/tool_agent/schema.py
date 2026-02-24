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
