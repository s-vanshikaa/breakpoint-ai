import json
from collections import Counter

import pytest

from attacks.loader import DatasetError, filter_by_category, load_test_cases
from attacks.schema import AttackCategory, TargetApp, TestCase
from config import settings
from targets.tool_agent.permissions import ALLOWED, TOOL_PERMISSIONS

CASES = load_test_cases()

EXPECTED_BEHAVIOR = {
    AttackCategory.direct_injection: {"must_not_leak_secret", "must_not_call"},
    AttackCategory.indirect_rag_injection: {"must_not_follow_injected_instruction"},
    AttackCategory.secret_extraction: {"must_not_leak_secret"},
    AttackCategory.tool_misuse: {"must_not_call"},
    AttackCategory.benign: {"must_answer_correctly", "must_call_expected_tool"},
}


def ids(cases):
    return [c.id for c in cases]


class TestDatasetIntegrity:
    def test_dataset_is_not_empty_and_covers_all_five_categories(self):
        assert {c.category for c in CASES} == set(AttackCategory)

    def test_ids_are_unique(self):
        duplicates = [i for i, n in Counter(ids(CASES)).items() if n > 1]
        assert duplicates == []

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_required_fields_are_non_empty(self, case):
        assert case.id.strip()
        assert case.prompt.strip()
        assert case.expected.strip()

    def test_raw_categories_and_targets_are_valid_enum_values(self):
        raw = [json.loads(line) for line in settings.test_cases_path.read_text().splitlines() if line]
        assert {r["category"] for r in raw} <= {c.value for c in AttackCategory}
        assert {r["target"] for r in raw} <= {t.value for t in TargetApp}

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_expected_behavior_matches_category(self, case):
        assert case.expected in EXPECTED_BEHAVIOR[case.category]

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_id_prefix_matches_category(self, case):
        prefix = {
            AttackCategory.direct_injection: "direct",
            AttackCategory.indirect_rag_injection: "indirect",
            AttackCategory.secret_extraction: "secret",
            AttackCategory.tool_misuse: "tool",
            AttackCategory.benign: "benign",
        }[case.category]
        assert case.id.startswith(f"{prefix}_")

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_evaluation_metadata_matches_expected_behavior(self, case):
        if case.expected == "must_not_leak_secret" or case.expected == "must_not_follow_injected_instruction":
            assert case.protected_value
            assert case.target == TargetApp.rag_assistant
        elif case.expected == "must_not_call":
            assert case.forbidden_tool
            assert case.target == TargetApp.tool_agent
        elif case.expected == "must_call_expected_tool":
            assert case.expected_tool
            assert case.target == TargetApp.tool_agent
        elif case.expected == "must_answer_correctly":
            assert case.expected_keywords
            assert case.target == TargetApp.rag_assistant

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_referenced_tools_exist_and_forbidden_tools_are_not_allowed(self, case):
        if case.forbidden_tool:
            assert case.forbidden_tool in TOOL_PERMISSIONS
            assert TOOL_PERMISSIONS[case.forbidden_tool] != ALLOWED
        if case.expected_tool:
            assert TOOL_PERMISSIONS.get(case.expected_tool) == ALLOWED

    def test_protected_values_exist_in_the_document_corpus(self):
        corpus = "".join(p.read_text() for p in settings.documents_dir.glob("*.md"))
        for value in {c.protected_value for c in CASES if c.protected_value}:
            assert value in corpus

    def test_benign_prompts_are_not_flagged_by_the_input_guardrail(self):
        from guardrails.input_guardrail import check_input

        flagged = [c.id for c in CASES if c.category == AttackCategory.benign and check_input(c.prompt)]
        assert flagged == []


class TestFilterByCategory:
    def test_filters_and_preserves_order(self):
        tool_cases = filter_by_category(CASES, AttackCategory.tool_misuse)
        assert tool_cases and all(c.category == AttackCategory.tool_misuse for c in tool_cases)
        assert ids(tool_cases) == [i for i in ids(CASES) if i in ids(tool_cases)]


def write_jsonl(tmp_path, *lines: str):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


VALID = {
    "id": "a_1",
    "category": "benign",
    "target": "tool_agent",
    "prompt": "hi",
    "expected": "must_call_expected_tool",
    "expected_tool": "lookup_employee",
}


class TestLoaderValidation:
    def test_valid_file_loads_and_skips_blank_lines(self, tmp_path):
        path = write_jsonl(tmp_path, json.dumps(VALID), "", json.dumps({**VALID, "id": "a_2"}))
        assert ids(load_test_cases(path)) == ["a_1", "a_2"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_test_cases(tmp_path / "missing.jsonl")

    def test_invalid_json_reports_line_number(self, tmp_path):
        path = write_jsonl(tmp_path, json.dumps(VALID), "{not json")
        with pytest.raises(DatasetError, match=r"cases\.jsonl:2: invalid JSON"):
            load_test_cases(path)

    @pytest.mark.parametrize("missing", ["id", "category", "target", "prompt", "expected"])
    def test_missing_required_field(self, tmp_path, missing):
        bad = {k: v for k, v in VALID.items() if k != missing}
        with pytest.raises(DatasetError, match=missing):
            load_test_cases(write_jsonl(tmp_path, json.dumps(bad)))

    def test_invalid_category(self, tmp_path):
        with pytest.raises(DatasetError, match="category"):
            load_test_cases(write_jsonl(tmp_path, json.dumps({**VALID, "category": "made_up"})))

    def test_invalid_target(self, tmp_path):
        with pytest.raises(DatasetError, match="target"):
            load_test_cases(write_jsonl(tmp_path, json.dumps({**VALID, "target": "made_up"})))

    def test_case_without_evaluation_criterion(self, tmp_path):
        bad = {k: v for k, v in VALID.items() if k != "expected_tool"}
        with pytest.raises(DatasetError, match="at least one of"):
            load_test_cases(write_jsonl(tmp_path, json.dumps(bad)))

    def test_duplicate_ids(self, tmp_path):
        path = write_jsonl(tmp_path, json.dumps(VALID), json.dumps(VALID))
        with pytest.raises(DatasetError, match="duplicate test id 'a_1'"):
            load_test_cases(path)

    def test_empty_file(self, tmp_path):
        with pytest.raises(DatasetError, match="no test cases"):
            load_test_cases(write_jsonl(tmp_path, ""))


def test_testcase_model_is_not_collected_by_pytest():
    assert TestCase.__test__ is False
