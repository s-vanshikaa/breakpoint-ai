from attacks.schema import AttackCategory, TargetApp, TestCase
from config import settings
from targets.tool_agent.permissions import ALLOWED, TOOL_PERMISSIONS

EXPECTED_BEHAVIOR = {
    AttackCategory.direct_injection: {"must_not_leak_secret", "must_not_call"},
    AttackCategory.indirect_rag_injection: {"must_not_follow_injected_instruction"},
    AttackCategory.secret_extraction: {"must_not_leak_secret"},
    AttackCategory.tool_misuse: {"must_not_call"},
    AttackCategory.benign: {"must_answer_correctly", "must_call_expected_tool"},
}


def find_problems(test_cases: list[TestCase]) -> list[str]:
    """Semantic checks beyond schema validation. Returns human-readable problems."""
    problems: list[str] = []
    corpus = "".join(p.read_text() for p in sorted(settings.documents_dir.glob("*.md")))

    for tc in test_cases:
        if tc.expected not in EXPECTED_BEHAVIOR[tc.category]:
            problems.append(
                f"{tc.id}: expected '{tc.expected}' is not valid "
                f"for category '{tc.category.value}'."
            )
        if tc.forbidden_tool:
            if tc.forbidden_tool not in TOOL_PERMISSIONS:
                problems.append(f"{tc.id}: unknown forbidden_tool '{tc.forbidden_tool}'.")
            elif TOOL_PERMISSIONS[tc.forbidden_tool] == ALLOWED:
                problems.append(
                    f"{tc.id}: forbidden_tool '{tc.forbidden_tool}' is an allowed tool."
                )
            if tc.target != TargetApp.tool_agent:
                problems.append(f"{tc.id}: forbidden_tool requires target tool_agent.")
        if tc.expected_tool:
            if TOOL_PERMISSIONS.get(tc.expected_tool) != ALLOWED:
                problems.append(
                    f"{tc.id}: expected_tool '{tc.expected_tool}' is not an allowed tool."
                )
            if tc.target != TargetApp.tool_agent:
                problems.append(f"{tc.id}: expected_tool requires target tool_agent.")
        if tc.expected_keywords and tc.target != TargetApp.rag_assistant:
            problems.append(f"{tc.id}: expected_keywords requires target rag_assistant.")
        if tc.protected_value:
            if tc.target != TargetApp.rag_assistant:
                problems.append(f"{tc.id}: protected_value requires target rag_assistant.")
            if tc.protected_value not in corpus:
                problems.append(
                    f"{tc.id}: protected_value '{tc.protected_value}' does not appear in "
                    "data/documents, so it could never be leaked."
                )
    return problems
