import json

import pytest

from attacks.loader import load_test_cases
from attacks.schema import AttackCategory, TargetApp
from attacks.validate import find_problems
from models.ollama_client import OllamaClient, ollama_client
from runner import cli, report, workflow
from runner.common import ResultsError
from runner.metrics import compute_baseline_metrics
from tests.conftest import make_case, make_record


def cases():
    return [
        make_case(id="direct_1"),
        make_case(id="direct_2"),
        make_case(id="secret_1", category=AttackCategory.secret_extraction),
        make_case(
            id="benign_1",
            category=AttackCategory.benign,
            protected_value=None,
            expected_keywords=["x"],
            expected="must_answer_correctly",
        ),
    ]


def records(guarded: bool, passed: dict[str, bool]):
    by_id = {c.id: c for c in cases()}
    return [
        make_record(i, by_id[i].category.value, ok, guardrails_enabled=guarded)
        for i, ok in passed.items()
    ]


BASELINE_PASS = {"direct_1": False, "direct_2": False, "secret_1": True, "benign_1": True}
GUARDED_PASS = {"direct_1": True, "direct_2": False, "secret_1": True, "benign_1": False}


def stored(guarded: bool, passed: dict[str, bool], meta=None) -> workflow.StoredRun:
    recs = records(guarded, passed)
    return workflow.StoredRun(
        metrics=compute_baseline_metrics(recs, cases(), guardrails_enabled=guarded),
        records=recs,
        meta=meta,
    )


def save_run(directory, name, run: workflow.StoredRun):
    (directory / f"{name}.json").write_text(run.model_dump_json())


@pytest.fixture
def saved_runs(tmp_path):
    save_run(tmp_path, "baseline", stored(False, BASELINE_PASS))
    save_run(tmp_path, "guarded", stored(True, GUARDED_PASS))
    return tmp_path


class TestComparison:
    def test_build_comparison(self):
        c = workflow.build_comparison(stored(False, BASELINE_PASS), stored(True, GUARDED_PASS))
        assert c.baseline.overall_asr == pytest.approx(2 / 3)
        assert c.guarded.overall_asr == pytest.approx(1 / 3)
        assert c.overall_relative_asr_reduction == pytest.approx(0.5)
        assert c.benign_success_after == 0.0

    def test_rejects_swapped_runs(self):
        with pytest.raises(ResultsError, match="guardrails-off"):
            workflow.build_comparison(stored(True, GUARDED_PASS), stored(False, BASELINE_PASS))

    def test_rejects_runs_over_different_cases(self):
        partial = {k: v for k, v in GUARDED_PASS.items() if k != "direct_2"}
        with pytest.raises(ResultsError, match="different test cases"):
            workflow.build_comparison(stored(False, BASELINE_PASS), stored(True, partial))

    def test_compare_saved_runs_writes_comparison_file(self, saved_runs):
        comparison, _, _ = workflow.compare_saved_runs(saved_runs)
        written = json.loads((saved_runs / "comparison.json").read_text())
        assert written["overall_relative_asr_reduction"] == pytest.approx(0.5)
        assert written["baseline"]["total_tests"] == comparison.baseline.total_tests
        assert not list(saved_runs.glob("*.tmp"))

    def test_missing_baseline_names_the_command_to_run(self, tmp_path):
        with pytest.raises(ResultsError, match="python -m runner baseline"):
            workflow.compare_saved_runs(tmp_path)

    def test_missing_guarded_names_the_command_to_run(self, tmp_path):
        save_run(tmp_path, "baseline", stored(False, BASELINE_PASS))
        with pytest.raises(ResultsError, match="python -m runner guarded"):
            workflow.compare_saved_runs(tmp_path)

    def test_malformed_results_file(self, tmp_path):
        (tmp_path / "baseline.json").write_text('{"metrics": {}}')
        with pytest.raises(ResultsError, match="malformed"):
            workflow.load_run(tmp_path / "baseline.json", "hint")

    def test_results_without_meta_still_load(self, saved_runs):
        assert workflow.load_run(saved_runs / "baseline.json", "").meta is None


class TestReports:
    def test_run_report_contains_all_headline_numbers(self):
        run = stored(False, BASELINE_PASS, meta={"model": "m", "seed": 7})
        text = report.format_run_report("baseline", run.metrics, run.records, run.meta)
        assert "4 cases, guardrails OFF, model m, seed 7" in text
        assert "66.7% (2/3 attacks succeeded)" in text
        assert "Benign task success:         100.0% (1/1)" in text
        assert "direct_injection" in text and "100.0%  (2/2)" in text

    def test_comparison_report(self):
        base, guard = stored(False, BASELINE_PASS), stored(True, GUARDED_PASS)
        text = report.format_comparison_report(
            workflow.build_comparison(base, guard), base.records, guard.records
        )
        assert "-33.3 pp" in text  # overall ASR 66.7% -> 33.3%
        assert "Relative reduction in attack success rate: 50.0%" in text
        assert "Attacks still succeeding with guardrails (1): direct_2" in text
        assert "Benign tests broken by guardrails (1): benign_1" in text

    def test_dataset_summary_counts(self):
        text = report.format_dataset_summary(cases())
        assert "Total cases: 4" in text
        assert "direct_injection" in text and "rag_assistant" in text

    def test_pct(self):
        assert report.pct(0.785714) == "78.6%"
        assert report.pct(0) == "0.0%"


class TestExecuteRun:
    @pytest.fixture(autouse=True)
    def stub_llm_layer(self, monkeypatch):
        async def ready():
            return None

        async def fake_run_tests(test_cases, guardrails_enabled, on_record=None):
            passed = GUARDED_PASS if guardrails_enabled else BASELINE_PASS
            out = records(guardrails_enabled, passed)
            for i, r in enumerate(out, start=1):
                on_record(i, len(out), r)
            return out

        monkeypatch.setattr(workflow, "ensure_ollama_ready", ready)
        monkeypatch.setattr(workflow, "run_tests", fake_run_tests)
        yield
        ollama_client.default_options = {}

    def test_saves_run_with_metadata_and_applies_seed(self, tmp_path, capsys):
        import asyncio

        run = asyncio.run(workflow.execute_run("guarded", cases(), tmp_path, seed=42))
        data = json.loads((tmp_path / "guarded.json").read_text())
        assert data["meta"]["seed"] == 42 and data["meta"]["run"] == "guarded"
        assert data["meta"]["model"] == ollama_client.model
        assert data["metrics"]["guardrails_enabled"] is True
        assert len(data["records"]) == 4
        assert ollama_client.default_options == {"seed": 42}
        assert run.metrics.guardrails_enabled is True
        assert "[4/4]" in capsys.readouterr().out

    def test_unseeded_run_clears_previous_seed(self, tmp_path):
        import asyncio

        ollama_client.default_options = {"seed": 1}
        asyncio.run(workflow.execute_run("baseline", cases(), tmp_path))
        assert ollama_client.default_options == {}
        assert json.loads((tmp_path / "baseline.json").read_text())["meta"]["seed"] is None

    def test_rejects_unknown_run_name(self, tmp_path):
        import asyncio

        with pytest.raises(ValueError, match="Unknown run"):
            asyncio.run(workflow.execute_run("bogus", cases(), tmp_path))


class TestOllamaDefaultOptions:
    def test_default_options_are_merged_and_call_options_win(self, monkeypatch):
        import asyncio

        import httpx

        from models import ollama_client as module

        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.read())
            return httpx.Response(200, json={"response": "ok"})

        real = httpx.AsyncClient
        monkeypatch.setattr(
            module.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
        )
        client = OllamaClient(host="http://x.test", model="m")
        client.default_options = {"seed": 5, "temperature": 0.9}
        asyncio.run(client.complete("hi", options={"temperature": 0.1}))
        assert seen["payload"]["options"] == {"seed": 5, "temperature": 0.1}

        client.default_options = {}
        asyncio.run(client.complete("hi"))
        assert "options" not in seen["payload"]


class TestValidateDataset:
    def test_stored_dataset_has_no_problems(self):
        assert find_problems(load_test_cases()) == []

    def test_detects_semantic_problems(self):
        bad = [
            make_case(id="a", expected="must_call_expected_tool"),
            make_case(
                id="b",
                category=AttackCategory.tool_misuse,
                target=TargetApp.tool_agent,
                protected_value=None,
                forbidden_tool="lookup_employee",
                expected="must_not_call",
            ),
            make_case(id="c", protected_value="NOT_IN_ANY_DOCUMENT"),
            make_case(
                id="d",
                category=AttackCategory.tool_misuse,
                protected_value=None,
                forbidden_tool="no_such_tool",
                expected="must_not_call",
            ),
        ]
        text = "\n".join(find_problems(bad))
        assert "a: expected 'must_call_expected_tool' is not valid" in text
        assert "b: forbidden_tool 'lookup_employee' is an allowed tool" in text
        assert "c: protected_value 'NOT_IN_ANY_DOCUMENT' does not appear" in text
        assert "d: unknown forbidden_tool 'no_such_tool'" in text
        assert "d: forbidden_tool requires target tool_agent" in text


class TestCli:
    def test_validate_command(self, capsys):
        cli.main(["validate"])
        out = capsys.readouterr().out
        assert "Total cases: 50" in out and "Dataset OK." in out

    def test_compare_command(self, saved_runs, capsys):
        cli.main(["compare", "--results-dir", str(saved_runs)])
        out = capsys.readouterr().out
        assert "Baseline vs guarded (4 cases each)" in out
        assert (saved_runs / "comparison.json").exists()

    def test_compare_without_runs_exits_with_one_line_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["compare", "--results-dir", str(tmp_path)])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("error:") and "python -m runner baseline" in err

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["bogus"])
        assert exc.value.code == 2

    def test_run_command_rejects_unknown_category(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["run", "--category", "nope"])
        assert exc.value.code == 2

    def test_seed_must_be_an_integer(self):
        with pytest.raises(SystemExit):
            cli.main(["baseline", "--seed", "abc"])
