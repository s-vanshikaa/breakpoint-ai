from enum import Enum

from pydantic import BaseModel, model_validator


class AttackCategory(str, Enum):
    direct_injection = "direct_injection"
    indirect_rag_injection = "indirect_rag_injection"
    secret_extraction = "secret_extraction"
    tool_misuse = "tool_misuse"
    benign = "benign"


class TargetApp(str, Enum):
    rag_assistant = "rag_assistant"
    tool_agent = "tool_agent"


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class

    id: str
    category: AttackCategory
    target: TargetApp
    prompt: str
    expected: str

    # secret_extraction / indirect_rag_injection
    protected_value: str | None = None

    # tool_misuse
    forbidden_tool: str | None = None

    # benign tool_agent cases
    expected_tool: str | None = None

    # benign rag_assistant cases
    expected_keywords: list[str] | None = None

    notes: str | None = None

    @model_validator(mode="after")
    def _has_evaluation_criterion(self) -> "TestCase":
        if not (
            self.forbidden_tool
            or self.expected_tool
            or self.protected_value
            or self.expected_keywords
        ):
            raise ValueError(
                "must set at least one of forbidden_tool, expected_tool, "
                "protected_value, expected_keywords"
            )
        return self
