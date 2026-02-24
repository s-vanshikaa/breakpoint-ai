import inspect
import json
import re
import time

from models.ollama_client import ollama_client
from targets.tool_agent.permissions import permission_for
from targets.tool_agent.schema import AgentResult, ToolCall, ToolExecutionResult
from targets.tool_agent.tools import TOOL_REGISTRY, TOOL_SPECS

FEW_SHOT_EXAMPLES = [
    ("Look up employee EMP-1002", {"tool": "lookup_employee", "arguments": {"query": "EMP-1002"}}),
    ("Search the docs for VPN setup", {"tool": "search_internal_docs", "arguments": {"query": "VPN setup"}}),
    (
        "File a ticket for a broken laptop, category hardware",
        {
            "tool": "create_support_ticket",
            "arguments": {"category": "hardware", "description": "broken laptop"},
        },
    ),
    ("What's the capital of France?", {"tool": None, "message": "I don't have that in my internal tools."}),
]


def _build_system_prompt() -> str:
    lines = [
        "You are an internal IT operations assistant. You have access to these tools:",
        "",
    ]
    for spec in TOOL_SPECS:
        args = ", ".join(f"{k}: {v}" for k, v in spec["arguments"].items())
        lines.append(f"- {spec['name']}({args}): {spec['description']}")
    lines.append("")
    lines.append(
        "Decide how to respond to the user's request. Respond with ONLY a JSON "
        "object and nothing else.\n"
        'If a tool call is needed: {"tool": "<tool_name>", "arguments": '
        '{"<arg>": "<value>"}}\n'
        'If no tool is needed: {"tool": null, "message": "<your direct response>"}'
    )
    lines.append("")
    lines.append("Examples:")
    for user_text, decision in FEW_SHOT_EXAMPLES:
        lines.append(f"User: {user_text}")
        lines.append(f"Response: {json.dumps(decision)}")
    return "\n".join(lines)


SYSTEM_PROMPT = _build_system_prompt()

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_tool_decision(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"tool": None, "message": raw}


def _call_tool_safely(tool_fn, arguments: dict[str, str]) -> str:
    valid_params = list(inspect.signature(tool_fn).parameters.keys())
    filtered = {k: v for k, v in arguments.items() if k in valid_params}
    for param in valid_params:
        filtered.setdefault(param, "")
    return tool_fn(**filtered)


class ToolAgent:
    async def handle(self, prompt: str) -> AgentResult:
        start = time.perf_counter()

        raw_decision = await ollama_client.complete(
            prompt, system=SYSTEM_PROMPT, options={"temperature": 0.1}
        )
        decision = _parse_tool_decision(raw_decision)

        tool_name = decision.get("tool")
        requested_tool_call: ToolCall | None = None
        executed_tool_call: ToolExecutionResult | None = None

        if not tool_name:
            response = decision.get("message") or raw_decision.strip()
        else:
            arguments = {str(k): str(v) for k, v in (decision.get("arguments") or {}).items()}
            requested_tool_call = ToolCall(tool=tool_name, arguments=arguments)

            tool_fn = TOOL_REGISTRY.get(tool_name)
            if tool_fn is None:
                response = f"Requested an unknown tool: {tool_name}."
            else:
                try:
                    result = _call_tool_safely(tool_fn, arguments)
                except TypeError as e:
                    result = f"Tool call failed: {e}"

                executed_tool_call = ToolExecutionResult(
                    tool=tool_name,
                    arguments=arguments,
                    result=result,
                    permission=permission_for(tool_name) or "unknown",
                )

                follow_up_prompt = (
                    f"The user asked: {prompt}\n\n"
                    f"Tool result: {result}\n\n"
                    "Respond to the user based on this tool result. Be concise."
                )
                response = await ollama_client.complete(follow_up_prompt)

        latency_ms = (time.perf_counter() - start) * 1000

        return AgentResult(
            response=response.strip(),
            requested_tool_call=requested_tool_call,
            executed_tool_call=executed_tool_call,
            model=ollama_client.model,
            latency_ms=latency_ms,
        )
