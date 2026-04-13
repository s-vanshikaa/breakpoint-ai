import json

import pytest
from fastapi.testclient import TestClient

from api import main as main_module
from api.main import app
from config import settings
from models.ollama_client import OllamaModelNotFoundError, OllamaUnavailableError


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    directory = tmp_path / "results"
    directory.mkdir()
    return directory


@pytest.fixture
def api():
    return TestClient(app)


def records():
    return [
        {"test_id": "a", "category": "direct_injection", "passed": True},
        {"test_id": "b", "category": "benign", "passed": False},
    ]


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_summary_returns_comparison(api, results_dir):
    (results_dir / "comparison.json").write_text(json.dumps({"baseline": {}, "guarded": {}}))
    assert api.get("/results/summary").json() == {"baseline": {}, "guarded": {}}


def test_summary_missing_file_is_404_with_hint(api, results_dir):
    response = api.get("/results/summary")
    assert response.status_code == 404
    assert "Run the benchmark" in response.json()["detail"]


def test_summary_malformed_json_is_500(api, results_dir):
    (results_dir / "comparison.json").write_text("{nope")
    response = api.get("/results/summary")
    assert response.status_code == 500
    assert "not valid JSON" in response.json()["detail"]


def test_summary_missing_keys_is_500(api, results_dir):
    (results_dir / "comparison.json").write_text("{}")
    assert "missing baseline, guarded" in api.get("/results/summary").json()["detail"]


def test_records_default_run_and_category_filter(api, results_dir):
    (results_dir / "guarded.json").write_text(json.dumps({"records": records()}))
    assert len(api.get("/results/records").json()) == 2
    filtered = api.get("/results/records", params={"category": "benign"}).json()
    assert [r["test_id"] for r in filtered] == ["b"]


def test_records_selects_run(api, results_dir):
    (results_dir / "baseline.json").write_text(json.dumps({"records": records()[:1]}))
    assert len(api.get("/results/records", params={"run": "baseline"}).json()) == 1


def test_records_rejects_unknown_run(api, results_dir):
    assert api.get("/results/records", params={"run": "bogus"}).status_code == 422


def test_records_without_records_key_is_500(api, results_dir):
    (results_dir / "guarded.json").write_text(json.dumps({"metrics": {}}))
    assert api.get("/results/records").status_code == 500


@pytest.mark.parametrize(
    "error, status",
    [(OllamaUnavailableError("down"), 503), (OllamaModelNotFoundError("no model"), 503)],
)
def test_complete_maps_ollama_errors(api, monkeypatch, error, status):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(main_module.ollama_client, "complete", fail)
    response = api.post("/complete", json={"prompt": "hi"})
    assert response.status_code == status
    assert response.json() == {"detail": str(error)}


def test_complete_success(api, monkeypatch):
    async def ok(prompt, system=None):
        return f"echo:{prompt}"

    monkeypatch.setattr(main_module.ollama_client, "complete", ok)
    assert api.post("/complete", json={"prompt": "hi"}).json() == {"response": "echo:hi"}


def test_cors_origins_are_configurable():
    from config import Settings

    assert Settings(cors_origins="http://a.test, http://b.test").cors_origin_list == [
        "http://a.test",
        "http://b.test",
    ]
