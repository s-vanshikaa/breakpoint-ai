import asyncio

import httpx
import pytest

from models import ollama_client as module
from models.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)


def use_transport(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )


def client(model="llama3.2:1b") -> OllamaClient:
    return OllamaClient(host="http://ollama.test/", model=model)


def test_host_trailing_slash_is_stripped():
    assert client().host == "http://ollama.test"


def test_complete_returns_response_text(monkeypatch):
    seen = {}

    def handler(request):
        seen["json"] = request.read()
        return httpx.Response(200, json={"response": "hello"})

    use_transport(monkeypatch, handler)
    assert asyncio.run(client().complete("hi", system="sys", options={"temperature": 0.1})) == "hello"
    assert b'"system"' in seen["json"] and b'"temperature"' in seen["json"]


def test_complete_unreachable_server(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    use_transport(monkeypatch, handler)
    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        asyncio.run(client().complete("hi"))


def test_complete_model_not_found(monkeypatch):
    use_transport(monkeypatch, lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(OllamaModelNotFoundError, match="ollama pull llama3.2:1b"):
        asyncio.run(client().complete("hi"))


def test_complete_server_error(monkeypatch):
    use_transport(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(OllamaError, match="HTTP 500"):
        asyncio.run(client().complete("hi"))


def test_complete_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow")

    use_transport(monkeypatch, handler)
    with pytest.raises(OllamaError, match="did not respond"):
        asyncio.run(client().complete("hi"))


def test_complete_malformed_body(monkeypatch):
    use_transport(monkeypatch, lambda r: httpx.Response(200, json={"unexpected": 1}))
    with pytest.raises(OllamaError, match="malformed"):
        asyncio.run(client().complete("hi"))


def tags(*names):
    return lambda r: httpx.Response(200, json={"models": [{"name": n} for n in names]})


def test_check_ready_ok(monkeypatch):
    use_transport(monkeypatch, tags("llama3.2:1b"))
    asyncio.run(client().check_ready())


def test_check_ready_matches_latest_tag_for_untagged_model(monkeypatch):
    use_transport(monkeypatch, tags("mistral:latest"))
    asyncio.run(client(model="mistral").check_ready())


def test_check_ready_model_missing(monkeypatch):
    use_transport(monkeypatch, tags("other:1b"))
    with pytest.raises(OllamaModelNotFoundError, match="llama3.2:1b"):
        asyncio.run(client().check_ready())


def test_check_ready_server_down(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    use_transport(monkeypatch, handler)
    with pytest.raises(OllamaUnavailableError):
        asyncio.run(client().check_ready())
