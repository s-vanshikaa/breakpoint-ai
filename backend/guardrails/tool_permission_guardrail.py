from targets.tool_agent.permissions import ALLOWED, permission_for


def check_tool_permission(tool_name: str) -> str | None:
    permission = permission_for(tool_name)
    if permission != ALLOWED:
        return (
            f"Tool permission guardrail: '{tool_name}' is {permission or 'unrecognized'}, "
            "denied."
        )
    return None
