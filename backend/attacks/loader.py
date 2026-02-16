import json
from pathlib import Path

from attacks.schema import AttackCategory, TestCase

DEFAULT_TEST_CASES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "test_cases" / "test_cases.jsonl"
)


def load_test_cases(path: Path = DEFAULT_TEST_CASES_PATH) -> list[TestCase]:
    test_cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_cases.append(TestCase.model_validate(json.loads(line)))
    return test_cases


def filter_by_category(
    test_cases: list[TestCase], category: AttackCategory
) -> list[TestCase]:
    return [tc for tc in test_cases if tc.category == category]
