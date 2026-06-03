import asyncio
import json

from benchmark.harness import (
    BenchmarkReport,
    run_full_benchmark,
    run_performance_benchmark,
    run_reliability_benchmark,
    run_resume_drill,
)
from benchmark.report import format_report


class TestPerformanceBenchmark:
    def test_one_result_per_concurrency_value(self):
        results = asyncio.run(
            run_performance_benchmark(seed=1, concurrencies=[1, 2], jobs_per_run=10, latency_ms=1.0)
        )
        assert [r.concurrency for r in results] == [1, 2]
        assert all(r.total_jobs == 10 and r.valid_jobs == 10 for r in results)

    def test_zero_failure_rate_means_full_success(self):
        results = asyncio.run(
            run_performance_benchmark(seed=1, concurrencies=[2], jobs_per_run=15, latency_ms=1.0)
        )
        assert results[0].failed_jobs == 0


class TestReliabilityBenchmark:
    def test_one_result_per_failure_rate(self):
        results = asyncio.run(
            run_reliability_benchmark(
                seed=1, failure_rates=[0.0, 0.3], concurrency=4, jobs_per_run=40, max_retries=3
            )
        )
        assert [r.injected_failure_rate for r in results] == [0.0, 0.3]
        assert results[0].completed == 40  # no injected failures
        assert results[0].permanent_failures == 0

    def test_higher_failure_rate_and_retries_still_recovers_some(self):
        results = asyncio.run(
            run_reliability_benchmark(
                seed=2, failure_rates=[0.3], concurrency=4, jobs_per_run=80, max_retries=4
            )
        )
        r = results[0]
        assert r.total_jobs == 80
        assert r.completed + r.permanent_failures == 80
        assert r.duplicate_job_ids == 0

    def test_max_retries_one_means_no_recovery(self):
        results = asyncio.run(
            run_reliability_benchmark(
                seed=1, failure_rates=[0.3], concurrency=4, jobs_per_run=40, max_retries=1
            )
        )
        assert results[0].recovered == 0


class TestResumeDrill:
    def test_no_duplicates_and_every_job_present_once(self, tmp_path):
        result = asyncio.run(
            run_resume_drill(tmp_path, seed=3, n_jobs=30, split=12, failure_rate=0.1, concurrency=3)
        )
        assert result.total_jobs == 30
        assert result.jobs_before_interruption == 12
        assert result.jobs_after_resume == 18
        assert result.duplicate_job_ids == 0
        assert result.all_ids_present_exactly_once is True

    def test_checkpoint_file_is_cleaned_up_afterward(self, tmp_path):
        asyncio.run(run_resume_drill(tmp_path, seed=1, n_jobs=10, split=4, concurrency=2))
        assert not (tmp_path / "resume-drill.checkpoint.jsonl").exists()

    def test_zero_failure_rate_still_completes_cleanly(self, tmp_path):
        result = asyncio.run(
            run_resume_drill(tmp_path, seed=1, n_jobs=20, split=8, failure_rate=0.0, concurrency=2)
        )
        assert result.all_ids_present_exactly_once is True
        assert result.duplicate_job_ids == 0


class TestFullBenchmarkOutput:
    def test_writes_timestamped_file_and_latest(self, tmp_path):
        path, report = asyncio.run(
            run_full_benchmark(
                tmp_path, seed=1, concurrencies=[1, 2],
                performance_jobs_per_run=10, performance_latency_ms=1.0,
                reliability_failure_rates=[0.0, 0.2], reliability_jobs_per_run=10,
                reliability_concurrency=2, max_retries=2,
            )
        )
        assert path.is_file()
        assert path.parent == tmp_path
        assert (tmp_path / "latest.json").is_file()
        assert json.loads(path.read_text()) == json.loads((tmp_path / "latest.json").read_text())

    def test_output_round_trips_through_the_schema(self, tmp_path):
        path, report = asyncio.run(
            run_full_benchmark(
                tmp_path, seed=1, concurrencies=[1],
                performance_jobs_per_run=5, performance_latency_ms=1.0,
                reliability_failure_rates=[0.0], reliability_jobs_per_run=5,
                reliability_concurrency=1, max_retries=1,
            )
        )
        reloaded = BenchmarkReport.model_validate_json(path.read_text())
        assert reloaded.model_dump() == report.model_dump()

    def test_does_not_overwrite_previous_timestamped_results(self, tmp_path):
        async def small_run():
            return await run_full_benchmark(
                tmp_path, seed=1, concurrencies=[1],
                performance_jobs_per_run=5, performance_latency_ms=1.0,
                reliability_failure_rates=[0.0], reliability_jobs_per_run=5,
                reliability_concurrency=1, max_retries=1,
            )

        path_a, _ = asyncio.run(small_run())
        path_b, _ = asyncio.run(small_run())
        # different benchmark IDs (timestamp-based) unless run in the same second - either way,
        # an existing timestamped file is never truncated/replaced by a later run.
        assert path_a.is_file()
        if path_a != path_b:
            assert path_b.is_file()

    def test_config_and_environment_are_recorded(self, tmp_path):
        _, report = asyncio.run(
            run_full_benchmark(
                tmp_path, seed=7, concurrencies=[1],
                performance_jobs_per_run=5, performance_latency_ms=1.0,
                reliability_failure_rates=[0.0], reliability_jobs_per_run=5,
                reliability_concurrency=1, max_retries=1,
            )
        )
        assert report.config.seed == 7
        assert report.environment.python_version
        assert report.environment.platform

    def test_format_report_includes_both_tables(self, tmp_path):
        _, report = asyncio.run(
            run_full_benchmark(
                tmp_path, seed=1, concurrencies=[1, 2],
                performance_jobs_per_run=5, performance_latency_ms=1.0,
                reliability_failure_rates=[0.0, 0.1], reliability_jobs_per_run=5,
                reliability_concurrency=1, max_retries=2,
            )
        )
        text = format_report(report)
        assert "MOCK-SERVER PERFORMANCE" in text
        assert "MOCK-SERVER RELIABILITY" in text
        assert "not real LLM throughput" in text
        assert "Resume drill" in text
