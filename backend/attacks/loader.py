import json
from pathlib import Path

from pydantic import ValidationError

from attacks.schema import AttackCategory, TestCase
from config import settings


class DatasetError(ValueError):
    """The benchmark dataset is missing or malformed."""


def load_test_cases(path: Path | None = None) -> list[TestCase]:
    path = path or settings.test_cases_path
    if not path.is_file():
        raise DatasetError(
            f"Test case file not found: {path}. Set DATA_DIR if your data lives "
            "outside the repository."
        )

    test_cases: list[TestCase] = []
    seen_ids: dict[str, int] = {}
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                test_case = TestCase.model_validate(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path.name}:{line_number}: invalid JSON ({e.msg}).") from e
            except ValidationError as e:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc']) or 'record'}: {err['msg']}"
                    for err in e.errors()
                )
                raise DatasetError(f"{path.name}:{line_number}: {problems}") from e

            if test_case.id in seen_ids:
                raise DatasetError(
                    f"{path.name}:{line_number}: duplicate test id '{test_case.id}' "
                    f"(first defined on line {seen_ids[test_case.id]})."
                )
            seen_ids[test_case.id] = line_number
            test_cases.append(test_case)

    if not test_cases:
        raise DatasetError(f"{path.name} contains no test cases.")
    return test_cases


def filter_by_category(
    test_cases: list[TestCase], category: AttackCategory
) -> list[TestCase]:
    return [tc for tc in test_cases if tc.category == category]
