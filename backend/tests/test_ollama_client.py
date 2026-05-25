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


def use_counting_transport(monkeypatch, handler):
    """Like use_transport, but also returns a list that grows by one per AsyncClient built."""
    real = httpx.AsyncClient
    built: list[httpx.AsyncClient] = []

    def factory(**kw):
        c = real(transport=httpx.MockTransport(handler), **kw)
        built.append(c)
        return c

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    return built


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


class TestClientLifecycle:
    def test_without_open_every_call_gets_its_own_client(self, monkeypatch):
        built = use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def two_calls():
            c = client()
            await c.complete("one")
            await c.complete("two")

        asyncio.run(two_calls())
        assert len(built) == 2
        assert all(b.is_closed for b in built)  # each ephemeral client cleans itself up

    def test_open_shares_one_client_across_calls(self, monkeypatch):
        built = use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def three_calls():
            c = client()
            await c.open()
            await c.complete("one")
            await c.complete("two")
            await c.complete("three")
            await c.aclose()

        asyncio.run(three_calls())
        assert len(built) == 1  # one pooled client served every request

    def test_open_is_idempotent(self, monkeypatch):
        built = use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def go():
            c = client()
            await c.open()
            first = c._client
            await c.open()  # must not replace an already-open client
            assert c._client is first
            await c.complete("hi")
            await c.aclose()

        asyncio.run(go())
        assert len(built) == 1

    def test_aclose_closes_and_clears_the_client(self, monkeypatch):
        use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def go():
            c = client()
            await c.open()
            pooled = c._client
            await c.aclose()
            assert c._client is None
            assert pooled.is_closed

        asyncio.run(go())

    def test_aclose_without_open_is_a_no_op(self):
        asyncio.run(client().aclose())  # must not raise

    def test_falls_back_to_ephemeral_client_after_close(self, monkeypatch):
        built = use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def go():
            c = client()
            await c.open()
            await c.complete("one")
            await c.aclose()
            await c.complete("two")  # client is closed again, so this must not reuse it

        asyncio.run(go())
        assert len(built) == 2

    def test_async_context_manager_opens_and_closes(self, monkeypatch):
        built = use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def go():
            c = client()
            async with c as opened:
                assert opened is c
                assert c._client is not None
                await c.complete("one")
                await c.complete("two")
            assert c._client is None

        asyncio.run(go())
        assert len(built) == 1

    def test_context_manager_closes_even_if_the_body_raises(self, monkeypatch):
        use_counting_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))

        async def go():
            c = client()
            with pytest.raises(RuntimeError):
                async with c:
                    raise RuntimeError("boom")
            assert c._client is None

        asyncio.run(go())

    def test_pooled_client_is_used_for_check_ready_and_complete(self, monkeypatch):
        def handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "llama3.2:1b"}]})
            return httpx.Response(200, json={"response": "ok"})

        built = use_counting_transport(monkeypatch, handler)

        async def go():
            c = client()
            async with c:
                await c.check_ready()
                await c.complete("hi")

        asyncio.run(go())
        assert len(built) == 1  # check_ready and complete shared the same pooled client
