import asyncio
import json
import random

import pytest

from attacks.schema import AttackCategory, TargetApp
from models.ollama_client import OllamaError, ollama_client
from runner import aggregate as agg
from runner import cli, experiment, report
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
        assert result.configs["baseline"].evaluations == 10

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


class Harness:
    """Replaces Ollama and the benchmark loop so experiments run instantly and offline."""

    def __init__(self, monkeypatch, fail_on=()):
        self.calls: list[tuple[int, bool]] = []
        self.seen_options: list[dict] = []
        self.fail_on = set(fail_on)

        async def ready():
            return None

        async def fake_run_tests(test_cases, guardrails_enabled, on_record=None):
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
