import pytest

from benchmark.jobs import JobOutcome
from benchmark.metrics import count_duplicates, summarize_performance, summarize_reliability


def outcome(job_id, status="ok", attempts=1, retried=False, latency_ms=10.0, error=None):
    return JobOutcome(
        job_id=job_id, status=status, attempts=attempts, retried=retried,
        latency_ms=latency_ms, error=error,
    )


class TestSummarizePerformance:
    def test_all_successful_counts_and_throughput(self):
        outcomes = [outcome(f"j{i}", latency_ms=10.0 * (i + 1)) for i in range(4)]
        result = summarize_performance(concurrency=2, outcomes=outcomes, duration_seconds=2.0)
        assert result.concurrency == 2
        assert result.total_jobs == 4
        assert result.valid_jobs == 4
        assert result.failed_jobs == 0
        assert result.throughput_per_sec == pytest.approx(2.0)  # 4 valid / 2s
        assert result.p50_ms == pytest.approx(25.0)
        assert result.retry_count == 0
        assert result.timeout_count == 0

    def test_failures_reduce_valid_jobs_but_not_total(self):
        outcomes = [
            outcome("j0", status="ok"),
            outcome("j1", status="timeout", attempts=3),
            outcome("j2", status="model_failure", attempts=2),
        ]
        result = summarize_performance(1, outcomes, duration_seconds=1.0)
        assert result.total_jobs == 3
        assert result.valid_jobs == 1
        assert result.failed_jobs == 2
        assert result.timeout_count == 1
        assert result.retry_count == (3 - 1) + (2 - 1)  # attempts-1 summed across all jobs

    def test_zero_duration_does_not_divide_by_zero(self):
        result = summarize_performance(1, [outcome("j0")], duration_seconds=0.0)
        assert result.throughput_per_sec == 0.0

    def test_empty_outcomes_give_null_percentiles(self):
        result = summarize_performance(1, [], duration_seconds=1.0)
        assert result.p50_ms is None and result.p95_ms is None and result.p99_ms is None
        assert result.total_jobs == 0


class TestSummarizeReliability:
    def test_completion_and_recovery_counts(self):
        outcomes = [
            outcome("j0", status="ok", attempts=1, retried=False),
            outcome("j1", status="ok", attempts=2, retried=True),  # recovered
            outcome("j2", status="model_failure", attempts=3, retried=True),  # exhausted retries
            outcome("j3", status="invalid_response", attempts=1, retried=False),  # not retryable
        ]
        result = summarize_reliability(0.2, outcomes, duration_seconds=3.0)
        assert result.total_jobs == 4
        assert result.completed == 2
        assert result.completion_rate == pytest.approx(0.5)
        assert result.recovered == 1
        assert result.permanent_failures == 2
        assert result.total_retries == 0 + 1 + 2 + 0
        assert result.failures_by_status == {
            "timeout": 0, "model_failure": 1, "invalid_response": 1, "evaluator_failure": 0,
        }

    def test_zero_jobs_completion_rate_is_null(self):
        result = summarize_reliability(0.1, [], duration_seconds=0.0)
        assert result.completion_rate is None
        assert result.total_jobs == 0

    def test_all_recovered(self):
        outcomes = [outcome(f"j{i}", attempts=2, retried=True) for i in range(5)]
        result = summarize_reliability(0.1, outcomes, duration_seconds=1.0)
        assert result.recovered == 5
        assert result.permanent_failures == 0
        assert result.completion_rate == 1.0

    def test_duplicate_count_is_passed_through(self):
        result = summarize_reliability(
            0.1, [outcome("j0"), outcome("j0")], duration_seconds=1.0, duplicate_job_ids=1
        )
        assert result.duplicate_job_ids == 1


class TestCountDuplicates:
    def test_no_duplicates(self):
        assert count_duplicates(["a", "b", "c"]) == 0

    def test_counts_extras_not_occurrences(self):
        assert count_duplicates(["a", "a", "a", "b"]) == 2  # 3 a's = 2 extra

    def test_empty_list(self):
        assert count_duplicates([]) == 0
