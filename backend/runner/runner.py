import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone

from attacks.schema import TestCase
from evaluators.evaluators import evaluate_test_case
from models.ollama_client import OllamaError, classify_error
from runner.schema import CategorySummary, RunSummary, TestRecord
from targets.rag_assistant.assistant import RAGAssistant
from targets.tool_agent.agent import ToolAgent

DEFAULT_CONCURRENCY = 1  # sequential by default; --concurrency opts into overlap


def _failed_record(
    test_case: TestCase,
    guardrails_enabled: bool,
    status: str,
    error: str,
    latency_ms: float,
    attempt_count: int,
    retried: bool,
    response: str = "",
) -> TestRecord:
    """A TestRecord for a case that never got a real evaluator verdict. `passed=False` is a
    conservative choice (an unverifiable attack is not a confirmed defense), not a graded
    result; `status`/`error` are what distinguish this from an actual evaluator FAIL."""
    return TestRecord(
        test_id=test_case.id,
        category=test_case.category.value,
        target=test_case.target.value,
        prompt=test_case.prompt,
        response=response,
        passed=False,
        reason=f"not evaluated ({status}): {error}",
        latency_ms=latency_ms,
        guardrails_enabled=guardrails_enabled,
        status=status,
        attempt_count=attempt_count,
        retried=retried,
        error=error,
    )


async def run_test_case(
    test_case: TestCase,
    rag_assistant: RAGAssistant,
    tool_agent: ToolAgent,
    guardrails_enabled: bool = False,
) -> TestRecord:
    """Runs one test case end to end. Model-call failures (after retries) and evaluator bugs
    are caught here and turned into a structured failed TestRecord rather than raised, so one
    bad case can't take down the rest of a run (see run_tests). Anything else still raises,
    since that's more likely a real bug than an expected failure mode."""
    start = time.perf_counter()

    if test_case.target.value == "rag_assistant":
        try:
            result = await rag_assistant.answer(
                test_case.prompt, guardrails_enabled=guardrails_enabled
            )
        except OllamaError as e:
            return _failed_record(
                test_case, guardrails_enabled, classify_error(e), str(e),
                (time.perf_counter() - start) * 1000, e.attempts or 1, bool(e.retried),
            )

        try:
            evaluation = evaluate_test_case(test_case, response_text=result.response)
        except Exception as e:
            return _failed_record(
                test_case, guardrails_enabled, "evaluator_failure", str(e),
                result.latency_ms, result.model_attempts, result.model_retried, result.response,
            )

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
            block_reason=result.block_reason,
            attempt_count=result.model_attempts,
            retried=result.model_retried,
        )

    try:
        result = await tool_agent.handle(test_case.prompt, guardrails_enabled=guardrails_enabled)
    except OllamaError as e:
        return _failed_record(
            test_case, guardrails_enabled, classify_error(e), str(e),
            (time.perf_counter() - start) * 1000, e.attempts or 1, bool(e.retried),
        )

    try:
        evaluation = evaluate_test_case(
            test_case,
            response_text=result.response,
            executed_tool_call=result.executed_tool_call,
        )
    except Exception as e:
        return _failed_record(
            test_case, guardrails_enabled, "evaluator_failure", str(e),
            result.latency_ms, result.model_attempts, result.model_retried, result.response,
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
        block_reason=result.block_reason,
        attempt_count=result.model_attempts,
        retried=result.model_retried,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def run_tests(
    test_cases: list[TestCase],
    guardrails_enabled: bool = False,
    on_record: Callable[[int, int, TestRecord], None] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[TestRecord]:
    """Runs every test case, at most `concurrency` at a time.

    Every case in `test_cases` is run exactly once; the returned list is always in the same
    order as `test_cases`, regardless of completion order or concurrency. If one case's model
    call raises, the rest still run to completion (only then is the first exception re-raised),
    so a single bad request can't strand or duplicate the others. `on_record` is called once
    per completed case, in completion order, for live progress reporting.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")

    rag_assistant = RAGAssistant()
    tool_agent = ToolAgent()
    semaphore = asyncio.Semaphore(concurrency)
    total = len(test_cases)
    completed = 0

    async def run_one(index: int, test_case: TestCase) -> tuple[int, TestRecord]:
        nonlocal completed
        async with semaphore:
            started_at = _now_iso()
            record = await run_test_case(test_case, rag_assistant, tool_agent, guardrails_enabled)
            record.started_at = started_at
            record.completed_at = _now_iso()
        completed += 1
        if on_record:
            on_record(completed, total, record)
        return index, record

    results = await asyncio.gather(
        *(run_one(i, tc) for i, tc in enumerate(test_cases)), return_exceptions=True
    )

    ok = [r for r in results if not isinstance(r, BaseException)]
    ok.sort(key=lambda pair: pair[0])  # stable order by original test-case index, not completion

    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        raise errors[0]
    return [record for _, record in ok]


def status_counts(records: list[TestRecord]) -> dict[str, int]:
    """How many records ended in each execution status (see runner.schema.EXECUTION_STATUSES)."""
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.status] += 1
    return dict(counts)


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
