import asyncio

import httpx
import pytest

from models import ollama_client as module
from models.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    classify_error,
    is_retryable,
)
from models.retry import RetryPolicy


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


def fast_policy(**overrides) -> RetryPolicy:
    """A policy with no real jitter/delay, so retry tests don't sleep for real."""
    fields = {"max_attempts": 3, "base_delay_seconds": 0.0, "max_delay_seconds": 0.0, "jitter_seconds": 0.0}
    fields.update(overrides)
    return RetryPolicy(**fields)


def sequenced_handler(*responses):
    """Returns each response/exception in `responses` in order, one per call."""
    calls = iter(responses)

    def handler(request):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


class TestClassification:
    def test_timeout_and_unavailable_are_retryable(self):
        assert is_retryable(OllamaTimeoutError("x")) is True
        assert is_retryable(OllamaUnavailableError("x")) is True

    def test_model_missing_and_malformed_are_not_retryable(self):
        assert is_retryable(OllamaModelNotFoundError("x")) is False
        assert is_retryable(OllamaResponseError("x")) is False

    def test_retryable_and_permanent_http_status_codes(self):
        for code in (429, 500, 502, 503):
            e = OllamaError("x")
            e.status_code = code
            assert is_retryable(e) is True, code
        for code in (400, 401, 403):
            e = OllamaError("x")
            e.status_code = code
            assert is_retryable(e) is False, code

    def test_error_categories(self):
        assert classify_error(OllamaTimeoutError("x")) == "timeout"
        assert classify_error(OllamaResponseError("x")) == "invalid_response"
        assert classify_error(OllamaUnavailableError("x")) == "model_failure"
        assert classify_error(OllamaModelNotFoundError("x")) == "model_failure"


class TestCompleteWithRetry:
    def test_succeeds_on_the_first_try_without_delay(self, monkeypatch):
        use_transport(monkeypatch, lambda r: httpx.Response(200, json={"response": "ok"}))
        result = asyncio.run(client().complete_with_retry("hi", policy=fast_policy()))
        assert result.text == "ok" and result.attempts == 1 and result.retried is False

    def test_transient_failure_then_success_is_retried_and_reported(self, monkeypatch):
        use_transport(
            monkeypatch,
            sequenced_handler(
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"response": "recovered"}),
            ),
        )
        result = asyncio.run(client().complete_with_retry("hi", policy=fast_policy()))
        assert result.text == "recovered"
        assert result.attempts == 2
        assert result.retried is True

    def test_retries_are_capped_at_max_attempts(self, monkeypatch):
        seen = {"n": 0}

        def handler(request):
            seen["n"] += 1
            return httpx.Response(500, text="boom")

        use_transport(monkeypatch, handler)
        with pytest.raises(OllamaError, match="HTTP 500") as excinfo:
            asyncio.run(client().complete_with_retry("hi", policy=fast_policy(max_attempts=3)))
        assert seen["n"] == 3  # first attempt + 2 retries, then gave up
        assert excinfo.value.attempts == 3
        assert excinfo.value.retried is True

    def test_permanent_failure_is_not_retried(self, monkeypatch):
        seen = {"n": 0}

        def handler(request):
            seen["n"] += 1
            return httpx.Response(404, json={"error": "not found"})

        use_transport(monkeypatch, handler)
        with pytest.raises(OllamaModelNotFoundError) as excinfo:
            asyncio.run(client().complete_with_retry("hi", policy=fast_policy(max_attempts=5)))
        assert seen["n"] == 1  # no retry attempted
        assert excinfo.value.attempts == 1
        assert excinfo.value.retried is False

    def test_timeout_is_retried_and_classified(self, monkeypatch):
        use_transport(
            monkeypatch,
            sequenced_handler(httpx.ReadTimeout("slow"), httpx.Response(200, json={"response": "ok"})),
        )
        result = asyncio.run(client().complete_with_retry("hi", policy=fast_policy()))
        assert result.attempts == 2

    def test_timeout_exhausted_is_classified_as_timeout(self, monkeypatch):
        def handler(request):
            raise httpx.ReadTimeout("slow")

        use_transport(monkeypatch, handler)
        with pytest.raises(OllamaTimeoutError) as excinfo:
            asyncio.run(client().complete_with_retry("hi", policy=fast_policy(max_attempts=2)))
        assert classify_error(excinfo.value) == "timeout"
        assert excinfo.value.attempts == 2

    def test_malformed_response_is_not_retried(self, monkeypatch):
        seen = {"n": 0}

        def handler(request):
            seen["n"] += 1
            return httpx.Response(200, json={"unexpected": 1})

        use_transport(monkeypatch, handler)
        with pytest.raises(OllamaResponseError):
            asyncio.run(client().complete_with_retry("hi", policy=fast_policy(max_attempts=5)))
        assert seen["n"] == 1

    def test_default_policy_is_used_when_none_given(self, monkeypatch):
        # client().retry_policy defaults to 3 attempts; make every attempt fail permanently
        # fast so the test doesn't wait on the real default backoff.
        c = client()
        c.retry_policy = fast_policy(max_attempts=2)
        seen = {"n": 0}

        def handler(request):
            seen["n"] += 1
            return httpx.Response(500, text="boom")

        use_transport(monkeypatch, handler)
        with pytest.raises(OllamaError):
            asyncio.run(c.complete_with_retry("hi"))
        assert seen["n"] == 2

    def test_backoff_actually_sleeps_between_retries(self, monkeypatch):
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
        use_transport(
            monkeypatch,
            sequenced_handler(
                httpx.Response(500, text="boom"), httpx.Response(200, json={"response": "ok"})
            ),
        )
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=5.0, jitter_seconds=0.0)
        asyncio.run(client().complete_with_retry("hi", policy=policy))
        assert sleeps == [1.0]

    def test_plain_complete_does_not_retry(self, monkeypatch):
        seen = {"n": 0}

        def handler(request):
            seen["n"] += 1
            return httpx.Response(500, text="boom")

        use_transport(monkeypatch, handler)
        with pytest.raises(OllamaError):
            asyncio.run(client().complete("hi"))
        assert seen["n"] == 1  # complete() never retries, only complete_with_retry() does
