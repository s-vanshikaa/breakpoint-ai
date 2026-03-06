from pydantic import BaseModel


class EvaluationResult(BaseModel):
    test_id: str
    passed: bool
    reason: str
