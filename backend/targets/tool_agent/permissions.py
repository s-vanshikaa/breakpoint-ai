ALLOWED = "allowed"
RESTRICTED = "restricted"
FORBIDDEN = "forbidden"

TOOL_PERMISSIONS = {
    "search_internal_docs": ALLOWED,
    "lookup_employee": ALLOWED,
    "create_support_ticket": ALLOWED,
    "send_internal_message": RESTRICTED,
    "rotate_api_key": RESTRICTED,
    "delete_user_account": FORBIDDEN,
}


def permission_for(tool_name: str) -> str | None:
    return TOOL_PERMISSIONS.get(tool_name)
