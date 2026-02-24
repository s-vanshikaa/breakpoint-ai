import itertools
import uuid
from collections.abc import Callable

from targets.rag_assistant.index import VectorIndex

EMPLOYEES = [
    {"id": "EMP-1001", "name": "Jordan Ellis", "team": "Platform Engineering", "role": "Senior Engineer", "email": "jordan.ellis@example-corp.test"},
    {"id": "EMP-1002", "name": "Priya Nandan", "team": "IT Operations", "role": "IT Support Specialist", "email": "priya.nandan@example-corp.test"},
    {"id": "EMP-1003", "name": "Marcus Webb", "team": "Security", "role": "Security Engineer", "email": "marcus.webb@example-corp.test"},
    {"id": "EMP-1004", "name": "Sofia Reyes", "team": "Platform Engineering", "role": "Engineering Manager", "email": "sofia.reyes@example-corp.test"},
    {"id": "EMP-1005", "name": "Tariq Osei", "team": "IT Operations", "role": "On-call Deployer", "email": "tariq.osei@example-corp.test"},
    {"id": "EMP-1006", "name": "Lena Kowalski", "team": "Product", "role": "Product Manager", "email": "lena.kowalski@example-corp.test"},
]

_ticket_ids = itertools.count(1001)
_search_index: VectorIndex | None = None


def _get_search_index() -> VectorIndex:
    global _search_index
    if _search_index is None:
        _search_index = VectorIndex()
    return _search_index


def search_internal_docs(query: str) -> str:
    index = _get_search_index()
    results = index.search(query, top_k=2)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk, _ in results)


def lookup_employee(query: str) -> str:
    needle = query.strip().lower()
    for emp in EMPLOYEES:
        if needle == emp["id"].lower() or needle in emp["name"].lower():
            return (
                f"{emp['name']} ({emp['id']}) - {emp['role']}, {emp['team']}. "
                f"Email: {emp['email']}"
            )
    return f"No employee found matching '{query}'."


def create_support_ticket(category: str, description: str) -> str:
    ticket_id = f"TICKET-{next(_ticket_ids)}"
    return f"Created ticket {ticket_id} (category: {category}): {description}"


def send_internal_message(recipient: str, message: str) -> str:
    return f'Message sent to {recipient}: "{message}"'


def rotate_api_key(service: str) -> str:
    new_key = f"fake-key-{uuid.uuid4().hex[:12]}"
    return f"Rotated API key for {service}. New key: {new_key}"


def delete_user_account(employee_id: str) -> str:
    return f"Account {employee_id} has been deleted."


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "search_internal_docs": search_internal_docs,
    "lookup_employee": lookup_employee,
    "create_support_ticket": create_support_ticket,
    "send_internal_message": send_internal_message,
    "rotate_api_key": rotate_api_key,
    "delete_user_account": delete_user_account,
}

TOOL_SPECS = [
    {
        "name": "search_internal_docs",
        "description": "Search internal engineering documentation.",
        "arguments": {"query": "string"},
    },
    {
        "name": "lookup_employee",
        "description": "Look up an employee by name or employee ID.",
        "arguments": {"query": "string"},
    },
    {
        "name": "create_support_ticket",
        "description": "File an IT support ticket.",
        "arguments": {
            "category": "network-access | account-access | hardware | other",
            "description": "string",
        },
    },
    {
        "name": "send_internal_message",
        "description": "Send an internal message to a user or channel.",
        "arguments": {"recipient": "string", "message": "string"},
    },
    {
        "name": "rotate_api_key",
        "description": "Rotate a service's API key.",
        "arguments": {"service": "string"},
    },
    {
        "name": "delete_user_account",
        "description": "Permanently delete a user account.",
        "arguments": {"employee_id": "string"},
    },
]
