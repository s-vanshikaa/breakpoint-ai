from collections import defaultdict

from attacks.schema import TestCase
from evaluators.evaluators import evaluate_test_case
from runner.schema import CategorySummary, RunSummary, TestRecord
from targets.rag_assistant.assistant import RAGAssistant
from targets.tool_agent.agent import ToolAgent


async def run_test_case(
    test_case: TestCase,
    rag_assistant: RAGAssistant,
    tool_agent: ToolAgent,
    guardrails_enabled: bool = False,
) -> TestRecord:
    if test_case.target.value == "rag_assistant":
        result = await rag_assistant.answer(test_case.prompt)
        evaluation = evaluate_test_case(test_case, response_text=result.response)
        return TestRecord(
            test_id=test_case.id,
            category=test_case.category.value,
            target=test_case.target.value,
            prompt=test_case.prompt,
            response=result.response,
            retrieved_context=result.retrieved_chunks,
            passed=evaluation.passed,
            reason=evaluation.reason,
            latency_ms=result.latency_ms,
            guardrails_enabled=guardrails_enabled,
        )

    result = await tool_agent.handle(test_case.prompt)
    evaluation = evaluate_test_case(
        test_case,
        response_text=result.response,
        executed_tool_call=result.executed_tool_call,
    )
    return TestRecord(
        test_id=test_case.id,
        category=test_case.category.value,
        target=test_case.target.value,
        prompt=test_case.prompt,
        response=result.response,
        requested_tool_call=result.requested_tool_call,
        executed_tool_call=result.executed_tool_call,
        passed=evaluation.passed,
        reason=evaluation.reason,
        latency_ms=result.latency_ms,
        guardrails_enabled=guardrails_enabled,
    )


async def run_tests(
    test_cases: list[TestCase], guardrails_enabled: bool = False
) -> list[TestRecord]:
    rag_assistant = RAGAssistant()
    tool_agent = ToolAgent()

    records = []
    for test_case in test_cases:
        record = await run_test_case(test_case, rag_assistant, tool_agent, guardrails_enabled)
        records.append(record)
    return records


def summarize(records: list[TestRecord], guardrails_enabled: bool = False) -> RunSummary:
    total = len(records)
    passed = sum(1 for r in records if r.passed)
    failed = total - passed

    by_category: dict[str, list[TestRecord]] = defaultdict(list)
    for r in records:
        by_category[r.category].append(r)

    category_breakdown = {}
    for category, recs in by_category.items():
        cat_total = len(recs)
        cat_passed = sum(1 for r in recs if r.passed)
        category_breakdown[category] = CategorySummary(
            total=cat_total,
            passed=cat_passed,
            failed=cat_total - cat_passed,
            pass_rate=cat_passed / cat_total if cat_total else 0.0,
        )

    return RunSummary(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=passed / total if total else 0.0,
        guardrails_enabled=guardrails_enabled,
        category_breakdown=category_breakdown,
    )
