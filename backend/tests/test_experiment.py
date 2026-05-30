import asyncio
import json
import random

import pytest

from attacks.schema import AttackCategory, TargetApp
from models.ollama_client import OllamaError, ollama_client
from runner import aggregate as agg
from runner import checkpoint, cli, experiment, report
from runner.common import ResultsError
from tests.conftest import make_case, make_record


def dataset():
    return [
        make_case(id="direct_1"),
        make_case(id="direct_2"),
        make_case(id="secret_1", category=AttackCategory.secret_extraction),
        make_case(
            id="tool_1",
            category=AttackCategory.tool_misuse,
            target=TargetApp.tool_agent,
            protected_value=None,
            forbidden_tool="delete_user",
        ),
        make_case(
            id="benign_1",
            category=AttackCategory.benign,
            protected_value=None,
            expected_keywords=["x"],
            expected="must_answer_correctly",
        ),
    ]


# (seed, guarded) -> ids that FAIL (attack succeeded / benign task broken)
FAILS = {
    (1, False): {"direct_1", "direct_2", "secret_1", "tool_1"},
    (2, False): {"direct_1", "direct_2"},
    (1, True): set(),
    (2, True): {"direct_1", "benign_1"},
}


def make_records(seed: int, guarded: bool, cases=None) -> list:
    failing = FAILS.get((seed, guarded), set())
    return [
        make_record(
            c.id,
            c.category.value,
            c.id not in failing,
            guardrails_enabled=guarded,
            latency_ms=10.0 * i,
        )
        for i, c in enumerate(cases or dataset(), start=1)
    ]


def trials(seeds=(1, 2), configs=agg.CONFIGS, failed=()):
    out = []
    for seed in seeds:
        for config in configs:
            if (config, seed) in failed:
                out.append(agg.Trial(config=config, seed=seed, status="failed", error="boom"))
            else:
                out.append(
                    agg.Trial(
                        config=config, seed=seed, records=make_records(seed, config == "guarded")
                    )
                )
    return out


def aggregate(ts, planned=None):
    return agg.aggregate_experiment("exp", ts, dataset(), planned=planned)


class TestHelpers:
    def test_percentile_interpolates(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert agg.percentile(values, 50) == pytest.approx(25.0)
        assert agg.percentile(values, 0) == 10.0
        assert agg.percentile(values, 100) == 40.0
        assert agg.percentile([7.0], 95) == 7.0
        assert agg.percentile([], 50) is None

    def test_relative_reduction_with_zero_baseline_is_undefined(self):
        assert agg.relative_reduction(0.0, 0.0) is None
        assert agg.relative_reduction(None, 0.1) is None
        assert agg.relative_reduction(0.4, 0.1) == pytest.approx(0.75)

    def test_std_of_single_trial_is_zero_and_of_none_is_none(self):
        assert agg.std_or_none([0.3]) == 0.0
        assert agg.std_or_none([]) is None


class TestAggregate:
    def test_counts_scale_with_trials_and_dataset(self):
        result = aggregate(trials())
        assert result.total_evaluations == 5 * 2 * 2  # cases x trials x configs
        assert result.trials_completed == 4 and result.trials_failed == 0
        assert result.configs["baseline"].total_evaluations == 10

        three_seeds = aggregate(
            [
                agg.Trial(config=c, seed=s, records=make_records(1, c == "guarded"))
                for s in (1, 2, 3)
                for c in agg.CONFIGS
            ]
        )
        assert three_seeds.total_evaluations == 5 * 3 * 2
        assert three_seeds.seeds == [1, 2, 3]

    def test_asr_mean_std_and_pooled(self):
        base = aggregate(trials()).configs["baseline"].asr
        assert base.per_trial == {"1": 1.0, "2": 0.5}
        assert base.mean == pytest.approx(0.75)
        assert base.std == pytest.approx(0.35355339)
        assert (base.numerator, base.denominator) == (6, 8)
        assert base.pooled == pytest.approx(0.75)

        guarded = aggregate(trials()).configs["guarded"].asr
        assert guarded.mean == pytest.approx(0.125)
        assert (guarded.numerator, guarded.denominator) == (1, 8)

    def test_tool_misuse_and_benign(self):
        result = aggregate(trials())
        assert result.configs["baseline"].tool_misuse_rate.mean == pytest.approx(0.5)
        assert result.configs["guarded"].tool_misuse_rate.mean == 0.0
        assert result.configs["baseline"].benign_success_rate.mean == 1.0
        assert result.configs["guarded"].benign_success_rate.mean == pytest.approx(0.5)

    def test_comparison_separates_absolute_and_relative(self):
        comp = aggregate(trials()).comparison
        assert comp.absolute_asr_reduction == pytest.approx(0.625)
        assert comp.relative_asr_reduction == pytest.approx(0.625 / 0.75)
        assert comp.benign_success_change == pytest.approx(-0.5)

    def test_category_breakdown(self):
        result = aggregate(trials())
        direct = result.configs["baseline"].asr_by_category["direct_injection"]
        assert direct.per_trial == {"1": 1.0, "2": 1.0}
        assert result.configs["guarded"].asr_by_category["direct_injection"].mean == 0.25
        assert set(result.configs["baseline"].asr_by_category) == {
            "direct_injection",
            "secret_extraction",
            "tool_misuse",
        }  # benign is not an attack category
        by_cat = result.comparison.by_category
        assert by_cat["secret_extraction"].baseline_asr == pytest.approx(0.5)
        assert by_cat["secret_extraction"].relative_reduction == pytest.approx(1.0)
        assert by_cat["direct_injection"].absolute_reduction == pytest.approx(0.75)

    def test_latency_is_aggregated_across_all_records(self):
        latency = aggregate(trials()).configs["baseline"].latency
        assert latency.count == 10
        assert latency.mean_ms == pytest.approx(30.0)
        assert latency.p50_ms == pytest.approx(30.0)
        assert latency.p95_ms == pytest.approx(50.0)

    def test_failed_trial_is_reported_and_excluded(self):
        result = aggregate(trials(failed={("guarded", 2)}), planned=4)
        assert result.trials_completed == 3 and result.trials_failed == 1
        assert result.total_evaluations == 15
        assert [(f.config, f.seed, f.error) for f in result.failed_trials] == [
            ("guarded", 2, "boom")
        ]
        guarded = result.configs["guarded"]
        assert guarded.trials_completed == 1
        assert guarded.asr.per_trial == {"1": 0.0}  # the failed seed is absent, not 0
        assert guarded.asr.std == 0.0

    def test_trials_that_never_ran_count_against_planned(self):
        result = aggregate(trials(seeds=(1,)), planned=4)
        assert result.trials_planned == 4 and result.trials_completed == 2

    def test_no_comparison_when_a_config_has_no_completed_trial(self):
        result = aggregate(trials(failed={("guarded", 1), ("guarded", 2)}))
        assert result.comparison is None
        assert result.configs["guarded"].asr.mean is None
        assert result.configs["guarded"].asr.pooled is None
        assert result.configs["guarded"].latency.p50_ms is None
        assert "No comparison" in report.format_experiment_report(result)

    def test_no_trials_at_all(self):
        result = aggregate([], planned=0)
        assert result.total_evaluations == 0 and result.comparison is None

    def test_zero_denominators_are_none_not_zero(self):
        only_direct = [make_case(id="direct_1")]
        records = [make_record("direct_1", "direct_injection", False)]
        result = agg.aggregate_experiment(
            "exp",
            [
                agg.Trial(config="baseline", seed=1, records=records),
                agg.Trial(config="guarded", seed=1, records=records),
            ],
            only_direct,
        )
        base = result.configs["baseline"]
        assert base.asr.pooled == 1.0
        assert base.tool_misuse_rate.pooled is None and base.tool_misuse_rate.std is None
        assert base.benign_success_rate.mean is None
        assert result.comparison.relative_asr_reduction == 0.0
        assert result.comparison.benign_success_change is None

    def test_zero_baseline_asr_gives_undefined_relative_reduction(self):
        no_failures = [make_record("direct_1", "direct_injection", True)]
        result = agg.aggregate_experiment(
            "exp",
            [
                agg.Trial(config="baseline", seed=1, records=no_failures),
                agg.Trial(config="guarded", seed=1, records=no_failures),
            ],
            [make_case(id="direct_1")],
        )
        assert result.comparison.relative_asr_reduction is None
        assert result.comparison.absolute_asr_reduction == 0.0

    def test_input_order_does_not_change_the_aggregate(self):
        ordered = trials()
        shuffled = list(ordered)
        random.Random(0).shuffle(shuffled)
        assert aggregate(ordered).model_dump() == aggregate(shuffled).model_dump()
        assert list(aggregate(shuffled).configs) == ["baseline", "guarded"]


def both_configs(records, cases):
    """An experiment with one trial per config, both using the same records/cases - handy when
    a test only cares about one config's aggregate."""
    return agg.aggregate_experiment(
        "exp",
        [agg.Trial(config="baseline", seed=1, records=records),
         agg.Trial(config="guarded", seed=1, records=records)],
        cases,
    )  # fmt: skip


class TestExecutionFailuresExcludedFromOutcomeMetrics:
    """A timeout/model/invalid-response/evaluator failure never produced a real verdict, so
    outcome metrics (ASR, tool misuse, benign success) must count only status == "ok" records,
    with failures surfaced separately rather than folded into the pass/fail rate."""

    def test_timeouts_do_not_affect_attack_success_rate(self):
        # A naive passed-only count would treat the timed-out case (passed=False by convention)
        # as an attack success, inflating ASR from 0.0 to 0.5. It must be excluded entirely.
        records = [
            make_record("direct_1", "direct_injection", True, status="ok"),  # defended
            make_record("direct_2", "direct_injection", False, status="timeout"),
        ]
        result = both_configs(records, [make_case(id="direct_1"), make_case(id="direct_2")])
        asr = result.configs["baseline"].asr
        assert (asr.numerator, asr.denominator) == (0, 1)
        assert asr.pooled == 0.0

    def test_model_failures_do_not_affect_attack_success_rate(self):
        records = [
            make_record("direct_1", "direct_injection", True, status="ok"),
            make_record("direct_2", "direct_injection", False, status="model_failure"),
        ]
        result = both_configs(records, [make_case(id="direct_1"), make_case(id="direct_2")])
        asr = result.configs["baseline"].asr
        assert (asr.numerator, asr.denominator) == (0, 1)
        assert asr.pooled == 0.0

    def test_invalid_response_and_evaluator_failure_are_also_excluded_from_asr(self):
        records = [
            make_record("direct_1", "direct_injection", True, status="ok"),
            make_record("direct_2", "direct_injection", False, status="invalid_response"),
            make_record("direct_3", "direct_injection", False, status="evaluator_failure"),
        ]
        cases = [make_case(id=f"direct_{i}") for i in (1, 2, 3)]
        result = both_configs(records, cases)
        asr = result.configs["baseline"].asr
        assert (asr.numerator, asr.denominator) == (0, 1)

    def test_failures_do_not_lower_benign_success_rate(self):
        records = [
            make_record("benign_1", "benign", True, status="ok"),
            make_record("benign_2", "benign", False, status="invalid_response"),
        ]
        cases = [
            make_case(
                id=f"benign_{i}", category=AttackCategory.benign, protected_value=None,
                expected_keywords=["x"], expected="must_answer_correctly",
            )
            for i in (1, 2)
        ]  # fmt: skip
        result = both_configs(records, cases)
        rate = result.configs["baseline"].benign_success_rate
        assert (rate.numerator, rate.denominator) == (1, 1)
        assert rate.pooled == 1.0  # not 0.5

    def test_failure_counts_are_surfaced_explicitly(self):
        records = [
            make_record("direct_1", "direct_injection", True, status="ok"),
            make_record("direct_2", "direct_injection", False, status="timeout"),
            make_record("secret_1", "secret_extraction", False, status="model_failure"),
            make_record("tool_1", "tool_misuse", False, status="invalid_response"),
            make_record("benign_1", "benign", False, status="evaluator_failure"),
        ]
        cases = [
            make_case(id="direct_1"),
            make_case(id="direct_2"),
            make_case(id="secret_1", category=AttackCategory.secret_extraction),
            make_case(
                id="tool_1", category=AttackCategory.tool_misuse, target=TargetApp.tool_agent,
                protected_value=None, forbidden_tool="delete_user",
            ),
            make_case(
                id="benign_1", category=AttackCategory.benign, protected_value=None,
                expected_keywords=["x"], expected="must_answer_correctly",
            ),
        ]  # fmt: skip
        result = both_configs(records, cases)
        base = result.configs["baseline"]
        assert base.total_evaluations == 5
        assert base.valid_evaluations == 1
        assert base.failed_evaluations == 4
        assert base.valid_evaluation_rate == pytest.approx(0.2)
        assert base.failures_by_status == {
            "timeout": 1, "model_failure": 1, "invalid_response": 1, "evaluator_failure": 1,
        }  # fmt: skip

    def test_zero_valid_denominator_returns_null_not_zero(self):
        records = [
            make_record("direct_1", "direct_injection", False, status="timeout"),
            make_record("direct_2", "direct_injection", False, status="model_failure"),
        ]
        result = both_configs(records, [make_case(id="direct_1"), make_case(id="direct_2")])
        asr = result.configs["baseline"].asr
        assert asr.numerator == 0 and asr.denominator == 0
        assert asr.pooled is None
        assert asr.mean is None
        assert asr.std is None
        # coverage itself is still a real number (0 of 2 were valid), never null
        assert result.configs["baseline"].valid_evaluation_rate == 0.0
        assert result.configs["baseline"].total_evaluations == 2

    def test_reduction_uses_only_valid_records_even_when_counts_differ_between_configs(self):
        baseline_records = [
            make_record("direct_1", "direct_injection", False, status="ok"),  # attack succeeded
            make_record("direct_2", "direct_injection", False, status="timeout"),  # excluded
        ]
        guarded_records = [
            make_record("direct_1", "direct_injection", True, status="ok"),  # defended
            make_record("direct_2", "direct_injection", True, status="ok"),  # defended
        ]
        result = agg.aggregate_experiment(
            "exp",
            [agg.Trial(config="baseline", seed=1, records=baseline_records),
             agg.Trial(config="guarded", seed=1, records=guarded_records)],
            [make_case(id="direct_1"), make_case(id="direct_2")],
        )  # fmt: skip
        assert result.configs["baseline"].asr.denominator == 1  # only direct_1 was gradeable
        assert result.configs["guarded"].asr.denominator == 2  # no failures here
        assert result.comparison.baseline_asr == 1.0
        assert result.comparison.guarded_asr == 0.0
        assert result.comparison.absolute_asr_reduction == 1.0
        assert result.comparison.relative_asr_reduction == 1.0

    def test_a_trial_where_everything_failed_contributes_no_outcome_signal(self):
        records = [
            make_record("direct_1", "direct_injection", False, status="timeout"),
            make_record("direct_2", "direct_injection", False, status="timeout"),
        ]
        result = both_configs(records, [make_case(id="direct_1"), make_case(id="direct_2")])
        base = result.configs["baseline"]
        assert base.asr.pooled is None
        assert base.valid_evaluations == 0
        assert base.failed_evaluations == 2
        assert base.failures_by_status["timeout"] == 2


class Harness:
    """Replaces Ollama and the benchmark loop so experiments run instantly and offline."""

    def __init__(self, monkeypatch, fail_on=()):
        self.calls: list[tuple[int, bool]] = []
        self.seen_options: list[dict] = []
        self.fail_on = set(fail_on)

        async def ready():
            return None

        async def fake_run_tests(test_cases, guardrails_enabled, on_record=None, concurrency=1):
            seed = ollama_client.default_options["seed"]
            self.calls.append((seed, guardrails_enabled))
            self.seen_options.append(dict(ollama_client.default_options))
            if (seed, guardrails_enabled) in self.fail_on:
                raise OllamaError("model exploded")
            return make_records(seed, guardrails_enabled, test_cases)

        monkeypatch.setattr(experiment, "ensure_ollama_ready", ready)
        monkeypatch.setattr(experiment, "run_tests", fake_run_tests)
        monkeypatch.setattr(experiment, "git_state", lambda: ("abc123", False))


@pytest.fixture(autouse=True)
def restore_client_options():
    yield
    ollama_client.default_options = {}


def run(tmp_path, seeds=(2, 1), experiment_id="exp-test", **kwargs):
    return asyncio.run(
        experiment.run_experiment(
            dataset(), list(seeds), tmp_path, experiment_id=experiment_id, on_record=None, **kwargs
        )
    )


class TestRunExperiment:
    def test_runs_every_config_for_every_seed_in_sorted_order(self, tmp_path, monkeypatch):
        h = Harness(monkeypatch)
        manifest, result = run(tmp_path, seeds=(3, 1, 2))
        assert h.calls == [
            (1, False), (1, True), (2, False), (2, True), (3, False), (3, True),
        ]  # fmt: skip
        assert result.total_evaluations == 5 * 3 * 2
        assert manifest.seeds == [1, 2, 3]

    def test_seed_is_passed_to_the_model_and_client_is_restored(self, tmp_path, monkeypatch):
        h = Harness(monkeypatch)
        ollama_client.default_options = {"temperature": 0}
        run(tmp_path, seeds=(1,))
        assert h.seen_options == [{"seed": 1}, {"seed": 1}]
        assert ollama_client.default_options == {"temperature": 0}

    def test_persists_manifest_trials_and_aggregate(self, tmp_path, monkeypatch):
        Harness(monkeypatch)
        manifest, result = run(tmp_path, seeds=(1, 2))
        root = tmp_path / "exp-test"

        assert sorted(p.name for p in (root / "trials").iterdir()) == [
            "baseline-seed-1.json",
            "baseline-seed-2.json",
            "guarded-seed-1.json",
            "guarded-seed-2.json",
        ]
        saved = json.loads((root / "manifest.json").read_text())
        assert saved["experiment_id"] == "exp-test"
        assert saved["status"] == "complete" and saved["completed_at"]
        assert saved["seeds"] == [1, 2]
        assert saved["configurations"] == ["baseline", "guarded"]
        assert saved["test_case_count"] == 5
        assert saved["git_sha"] == "abc123"
        assert saved["model"] == ollama_client.model
        assert len(saved["dataset_sha256"]) == 64
        assert [t["file"] for t in saved["trials"]][0] == "trials/baseline-seed-1.json"

        trial = json.loads((root / "trials" / "guarded-seed-2.json").read_text())
        assert trial["status"] == "completed"
        assert trial["meta"]["seed"] == 2 and trial["meta"]["run"] == "guarded"
        assert len(trial["records"]) == 5  # per-case records are kept, not just averages
        assert trial["metrics"]["guardrails_enabled"] is True

        on_disk = json.loads((root / "aggregate.json").read_text())
        assert on_disk == json.loads(result.model_dump_json())
        assert on_disk["total_evaluations"] == 20

    def test_saved_experiment_reloads_to_the_same_aggregate(self, tmp_path, monkeypatch):
        Harness(monkeypatch)
        _, result = run(tmp_path)
        manifest, loaded = experiment.load_experiment(tmp_path / "exp-test")
        rebuilt = agg.aggregate_experiment(
            manifest.experiment_id, loaded, dataset(), planned=len(manifest.trials)
        )
        assert rebuilt.model_dump() == result.model_dump()

    def test_dataset_fingerprint_is_stable_and_order_sensitive(self):
        assert experiment.dataset_fingerprint(dataset()) == experiment.dataset_fingerprint(dataset())
        assert experiment.dataset_fingerprint(dataset()) != experiment.dataset_fingerprint(
            list(reversed(dataset()))
        )

    def test_failed_trial_does_not_stop_the_others(self, tmp_path, monkeypatch):
        h = Harness(monkeypatch, fail_on={(1, True)})
        manifest, result = run(tmp_path, seeds=(1, 2))
        assert len(h.calls) == 4  # seed 2 still ran after seed 1 guarded failed
        assert manifest.status == "partial"
        assert result.trials_failed == 1 and result.trials_completed == 3
        failed = json.loads((tmp_path / "exp-test/trials/guarded-seed-1.json").read_text())
        assert failed["status"] == "failed" and failed["error"] == "model exploded"
        assert failed["records"] == [] and "metrics" not in failed
        assert [t.status for t in manifest.trials] == [
            "completed", "failed", "completed", "completed",
        ]  # fmt: skip

    def test_refuses_to_overwrite_an_existing_experiment(self, tmp_path, monkeypatch):
        Harness(monkeypatch)
        run(tmp_path)
        with pytest.raises(experiment.ExperimentError, match="already exists"):
            run(tmp_path)

    def test_rejects_bad_seed_lists_and_empty_datasets(self, tmp_path, monkeypatch):
        Harness(monkeypatch)
        with pytest.raises(experiment.ExperimentError, match="at least one seed"):
            run(tmp_path, seeds=())
        with pytest.raises(experiment.ExperimentError, match="unique"):
            run(tmp_path, seeds=(1, 1))
        with pytest.raises(experiment.ExperimentError, match="no test cases"):
            asyncio.run(experiment.run_experiment([], [1], tmp_path))
        assert not (tmp_path / "exp-test").exists()

    def test_load_experiment_errors(self, tmp_path):
        with pytest.raises(ResultsError, match="not found"):
            experiment.load_experiment(tmp_path)
        (tmp_path / "manifest.json").write_text("{not json")
        with pytest.raises(ResultsError, match="not a readable experiment"):
            experiment.load_experiment(tmp_path)

    def test_git_state_never_raises(self, tmp_path):
        assert experiment.git_state(tmp_path / "does-not-exist") == (None, None)


class TestCli:
    def test_experiment_command_prints_report_and_writes_files(
        self, tmp_path, monkeypatch, capsys
    ):
        Harness(monkeypatch)
        monkeypatch.setattr(cli, "load_test_cases", dataset)
        cli.main(
            ["experiment", "--seeds", "1", "2", "--experiment-id", "e1",
             "--experiments-dir", str(tmp_path)]
        )  # fmt: skip
        out = capsys.readouterr().out
        assert "Experiment e1: 20 evaluations, 4/4 trials completed" in out
        assert "relative" in out and "absolute" in out
        assert (tmp_path / "e1" / "aggregate.json").is_file()

    def test_invalid_seeds_exit_with_one_line_error(self, tmp_path, monkeypatch, capsys):
        Harness(monkeypatch)
        monkeypatch.setattr(cli, "load_test_cases", dataset)
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["experiment", "--seeds", "1", "1", "--experiments-dir", str(tmp_path)])
        assert exit_info.value.code == 1
        assert "unique" in capsys.readouterr().err

    def test_seeds_default_to_five_trials(self):
        assert cli.build_parser().parse_args(["experiment"]).seeds == [1, 2, 3, 4, 5]

    def test_resume_defaults_to_none(self):
        assert cli.build_parser().parse_args(["experiment"]).resume is None

    def test_max_retries_default_and_dest(self):
        args = cli.build_parser().parse_args(["experiment", "--max-retries", "5"])
        assert args.max_retries == 5


class RecordingHarness:
    """Like Harness, but calls on_record per case (so checkpointing is actually exercised) and
    can be told to fail partway through a specific trial, after some cases already checkpointed.
    """

    def __init__(self, monkeypatch, crash_after: dict[tuple[int, bool], int] | None = None):
        self.calls: list[tuple[int, bool, int]] = []  # (seed, guarded, len(test_cases))
        self.received_concurrency: list[int] = []
        self.crash_after = crash_after or {}

        async def ready():
            return None

        async def fake_run_tests(test_cases, guardrails_enabled, on_record=None, concurrency=1):
            seed = ollama_client.default_options["seed"]
            self.calls.append((seed, guardrails_enabled, len(test_cases)))
            self.received_concurrency.append(concurrency)
            cutoff = self.crash_after.get((seed, guardrails_enabled))
            produced = []
            for i, tc in enumerate(test_cases, start=1):
                if cutoff is not None and i > cutoff:
                    raise OllamaError(f"crashed after {cutoff} cases")
                failing = FAILS.get((seed, guardrails_enabled), set())
                record = make_record(
                    tc.id, tc.category.value, tc.id not in failing,
                    guardrails_enabled=guardrails_enabled, latency_ms=1.0,
                )
                produced.append(record)
                if on_record:
                    on_record(i, len(test_cases), record)
            return produced

        monkeypatch.setattr(experiment, "ensure_ollama_ready", ready)
        monkeypatch.setattr(experiment, "run_tests", fake_run_tests)
        monkeypatch.setattr(experiment, "git_state", lambda: ("abc123", False))


class TestCheckpointingAndResume:
    def test_checkpoint_file_is_removed_once_a_trial_completes(self, tmp_path, monkeypatch):
        RecordingHarness(monkeypatch)
        run(tmp_path, seeds=(1,))
        trials_dir = tmp_path / "exp-test" / "trials"
        assert not list(trials_dir.glob("*.checkpoint.jsonl"))

    def test_interrupted_trial_leaves_a_partial_checkpoint(self, tmp_path, monkeypatch):
        RecordingHarness(monkeypatch, crash_after={(1, False): 2})
        manifest, result = run(tmp_path, seeds=(1,))
        trial_entry = next(t for t in manifest.trials if t.config == "baseline" and t.seed == 1)
        assert trial_entry.status == "failed"
        ckpt = tmp_path / "exp-test" / "trials" / "baseline-seed-1.checkpoint.jsonl"
        assert [r.test_id for r in checkpoint.load_records(ckpt)] == [
            c.id for c in dataset()[:2]
        ]
        # the other three trials (guarded seed 1's included) still ran normally
        assert result.trials_completed == 1 and result.trials_failed == 1

    def test_resume_only_reruns_the_remaining_cases_of_an_interrupted_trial(
        self, tmp_path, monkeypatch
    ):
        h = RecordingHarness(monkeypatch, crash_after={(1, False): 2})
        run(tmp_path, seeds=(1,))
        assert h.calls == [(1, False, 5), (1, True, 5)]  # baseline crashed, guarded finished

        h2 = RecordingHarness(monkeypatch)  # no crashes this time
        manifest, result = asyncio.run(
            experiment.resume_experiment(dataset(), tmp_path, "exp-test", on_record=None)
        )
        # only the incomplete baseline/seed-1 trial was re-touched, and only for the 3
        # cases that weren't already checkpointed
        assert h2.calls == [(1, False, 3)]
        assert manifest.status == "complete"
        assert result.trials_completed == 2 and result.trials_failed == 0
        assert result.total_evaluations == 10

    def test_resume_produces_no_duplicate_records_in_original_order(self, tmp_path, monkeypatch):
        RecordingHarness(monkeypatch, crash_after={(1, False): 2})
        run(tmp_path, seeds=(1,))
        RecordingHarness(monkeypatch)
        asyncio.run(experiment.resume_experiment(dataset(), tmp_path, "exp-test", on_record=None))

        trial = json.loads(
            (tmp_path / "exp-test" / "trials" / "baseline-seed-1.json").read_text()
        )
        ids = [r["test_id"] for r in trial["records"]]
        assert ids == [c.id for c in dataset()]  # original dataset order, exactly once each
        assert len(ids) == len(set(ids))
        assert trial["status"] == "completed"
        ckpt = tmp_path / "exp-test" / "trials" / "baseline-seed-1.checkpoint.jsonl"
        assert not ckpt.exists()  # cleared once the trial finished

    def test_resume_skips_trials_already_marked_completed(self, tmp_path, monkeypatch):
        RecordingHarness(monkeypatch)
        run(tmp_path, seeds=(1, 2))  # everything completes on the first pass

        h2 = RecordingHarness(monkeypatch)
        manifest, result = asyncio.run(
            experiment.resume_experiment(dataset(), tmp_path, "exp-test", on_record=None)
        )
        assert h2.calls == []  # nothing left to do
        assert result.total_evaluations == 20
        assert manifest.status == "complete"

    def test_resume_a_trial_that_failed_with_zero_progress_reruns_it_fully(
        self, tmp_path, monkeypatch
    ):
        RecordingHarness(monkeypatch, crash_after={(1, True): 0})
        run(tmp_path, seeds=(1,))

        h2 = RecordingHarness(monkeypatch)
        manifest, result = asyncio.run(
            experiment.resume_experiment(dataset(), tmp_path, "exp-test", on_record=None)
        )
        assert h2.calls == [(1, True, 5)]  # nothing had been checkpointed, so all 5 re-ran
        assert result.trials_failed == 0

    def test_resume_preserves_original_seeds_and_concurrency(self, tmp_path, monkeypatch):
        h = RecordingHarness(monkeypatch, crash_after={(1, False): 1})
        run(tmp_path, seeds=(1, 2), concurrency=3)
        assert h.received_concurrency == [3, 3, 3, 3]  # all 4 trials ran at concurrency 3

        h2 = RecordingHarness(monkeypatch)
        manifest, _ = asyncio.run(
            experiment.resume_experiment(dataset(), tmp_path, "exp-test", on_record=None)
        )
        assert manifest.seeds == [1, 2]  # unchanged, even though resume takes no --seeds
        assert manifest.concurrency == 3
        assert h2.received_concurrency == [3]  # the resumed trial also ran at the original value

    def test_resume_rejects_a_changed_dataset(self, tmp_path, monkeypatch):
        RecordingHarness(monkeypatch, crash_after={(1, False): 1})
        run(tmp_path, seeds=(1,))

        changed = [*dataset()[:-1], make_case(id="benign_1", prompt="a different prompt")]
        with pytest.raises(experiment.ExperimentError, match="doesn't match"):
            asyncio.run(experiment.resume_experiment(changed, tmp_path, "exp-test"))

    def test_resume_missing_experiment_raises(self, tmp_path):
        with pytest.raises(experiment.ExperimentError, match="No experiment found"):
            asyncio.run(experiment.resume_experiment(dataset(), tmp_path, "does-not-exist"))

    def test_resume_via_cli(self, tmp_path, monkeypatch, capsys):
        RecordingHarness(monkeypatch, crash_after={(1, False): 2})
        monkeypatch.setattr(cli, "load_test_cases", dataset)
        cli.main(["experiment", "--seeds", "1", "--experiment-id", "e1",
                   "--experiments-dir", str(tmp_path)])  # fmt: skip

        RecordingHarness(monkeypatch)
        cli.main(["experiment", "--resume", "e1", "--experiments-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "Resuming experiment e1" in out
        assert "2/2 trials completed" in out
