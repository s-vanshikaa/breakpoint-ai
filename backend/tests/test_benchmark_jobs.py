import asyncio

import pytest

from benchmark.jobs import JobOutcome, job_ids, outcome_to_record, run_job, run_jobs
from benchmark.mock_server import MockConfig, MockServer
from models.ollama_client import OllamaClient


class TestJobIds:
    def test_generates_the_requested_count_uniquely(self):
        ids = job_ids(5)
        assert len(ids) == 5 and len(set(ids)) == 5

    def test_ids_sort_lexicographically_in_dataset_order(self):
        # zero-padded width, so string-sort matches numeric order even at 3-digit counts
        ids = job_ids(120)
        assert ids == sorted(ids)

    def test_custom_prefix(self):
        assert job_ids(2, prefix="x")[0].startswith("x-")


class TestRunJob:
    def test_successful_job_reports_one_attempt(self):
        async def go():
            config = MockConfig(seed=1)
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model=config.model, timeout=1.0)
                return await run_job(client, "job-0")

        outcome = asyncio.run(go())
        assert outcome.status == "ok"
        assert outcome.attempts == 1
        assert outcome.retried is False
        assert outcome.error is None
        assert outcome.latency_ms >= 0

    def test_failing_job_is_classified_and_carries_attempt_count(self):
        async def go():
            config = MockConfig(seed=1, mode="malformed")
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model=config.model, timeout=1.0)
                from models.retry import RetryPolicy

                client.retry_policy = RetryPolicy(max_attempts=2, base_delay_seconds=0, jitter_seconds=0)
                return await run_job(client, "job-0")

        outcome = asyncio.run(go())
        assert outcome.status == "invalid_response"
        assert outcome.attempts == 1  # malformed is not retryable, so no retry happened
        assert outcome.error is not None


class TestRunJobsConcurrency:
    async def _run(self, n, concurrency, failure_rate=0.0, mode="success"):
        config = MockConfig(seed=1, mode=mode, failure_rate=failure_rate)
        async with MockServer(config) as server:
            client = OllamaClient(host=server.host, model=config.model, timeout=1.0)
            async with client:
                return await run_jobs(client, job_ids(n), concurrency=concurrency)

    def test_every_job_runs_exactly_once(self):
        outcomes = asyncio.run(self._run(15, concurrency=4))
        ids = [o.job_id for o in outcomes]
        assert sorted(ids) == sorted(job_ids(15))
        assert len(ids) == len(set(ids))

    def test_output_order_matches_input_order(self):
        outcomes = asyncio.run(self._run(10, concurrency=3))
        assert [o.job_id for o in outcomes] == job_ids(10)

    def test_progress_callback_fires_once_per_job(self):
        seen = []

        async def go():
            config = MockConfig(seed=1)
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model=config.model, timeout=1.0)
                async with client:
                    return await run_jobs(
                        client, job_ids(6), concurrency=2,
                        on_outcome=lambda i, total, o: seen.append((i, total, o.job_id)),
                    )

        asyncio.run(go())
        assert len(seen) == 6
        assert [i for i, _, _ in seen] == [1, 2, 3, 4, 5, 6]

    def test_rejects_non_positive_concurrency(self):
        async def go():
            config = MockConfig(seed=1)
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model=config.model, timeout=1.0)
                await run_jobs(client, job_ids(1), concurrency=0)

        with pytest.raises(ValueError, match="at least 1"):
            asyncio.run(go())

    def test_mixed_success_and_failure_are_all_accounted_for(self):
        outcomes = asyncio.run(self._run(40, concurrency=4, failure_rate=0.3))
        assert len(outcomes) == 40
        assert {o.job_id for o in outcomes} == set(job_ids(40))
        assert any(o.status != "ok" for o in outcomes)
        assert any(o.status == "ok" for o in outcomes)


class TestOutcomeToRecord:
    def test_ok_outcome_is_passed_and_status_ok(self):
        outcome = JobOutcome(job_id="job-0", status="ok", attempts=1, retried=False, latency_ms=5.0)
        record = outcome_to_record(outcome)
        assert record.passed is True
        assert record.status == "ok"
        assert record.category == "benchmark"
        assert record.test_id == "job-0"

    def test_failed_outcome_is_not_passed_and_keeps_error(self):
        outcome = JobOutcome(
            job_id="job-1", status="timeout", attempts=3, retried=True, latency_ms=9.0,
            error="boom",
        )
        record = outcome_to_record(outcome)
        assert record.passed is False
        assert record.status == "timeout"
        assert record.attempt_count == 3
        assert record.retried is True
        assert record.error == "boom"
