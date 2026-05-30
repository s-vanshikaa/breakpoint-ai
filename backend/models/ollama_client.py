import asyncio
import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from config import settings
from models.retry import RETRYABLE_STATUS_CODES, CompletionResult, RetryPolicy, backoff_delay


class OllamaError(RuntimeError):
    """Base class for errors talking to the local Ollama server.

    `status_code` and `attempts`/`retried` are set opportunistically (by _complete_once and
    complete_with_retry respectively) so callers can classify and log a failure without
    parsing the message string.
    """

    status_code: int | None = None
    attempts: int | None = None
    retried: bool | None = None


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached (connection error). Retryable."""


class OllamaModelNotFoundError(OllamaError):
    """Ollama is running but the configured model is not pulled. Not retryable."""


class OllamaTimeoutError(OllamaError):
    """The request did not complete within the configured timeout. Retryable."""


class OllamaResponseError(OllamaError):
    """Ollama returned a 200 with a body we couldn't parse. Not retryable (deterministic)."""


def is_retryable(error: OllamaError) -> bool:
    """Whether retrying `error` is likely to help, per the categories in the Commit 3 brief."""
    if isinstance(error, (OllamaTimeoutError, OllamaUnavailableError)):
        return True
    if isinstance(error, (OllamaModelNotFoundError, OllamaResponseError)):
        return False
    return error.status_code in RETRYABLE_STATUS_CODES


def classify_error(error: OllamaError) -> str:
    """Maps an OllamaError to one of the failure categories a TestRecord can carry."""
    if isinstance(error, OllamaTimeoutError):
        return "timeout"
    if isinstance(error, OllamaResponseError):
        return "invalid_response"
    return "model_failure"  # unavailable, model missing, or any other HTTP failure


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout_seconds
        # Applied to every request (e.g. {"seed": 42}); per-call options take precedence.
        self.default_options: dict = {}
        # Used by complete_with_retry() when no per-call policy is given.
        self.retry_policy = RetryPolicy(
            max_attempts=settings.ollama_retry_max_attempts,
            base_delay_seconds=settings.ollama_retry_base_delay_seconds,
            max_delay_seconds=settings.ollama_retry_max_delay_seconds,
            jitter_seconds=settings.ollama_retry_jitter_seconds,
        )
        # Set by open()/__aenter__ for the lifetime of a benchmark run, so every request
        # shares one connection pool instead of opening a fresh TCP/TLS handshake each time.
        # None (the default) falls back to a short-lived client per call, as before.
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        """Starts a pooled client shared by every request until aclose() is called."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        """Closes the pooled client, if one is open. Safe to call more than once."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaClient":
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _request(self, method: str, url: str, *, timeout: float | None = None, **kwargs):
        """Uses the pooled client if open() was called, otherwise a one-off client."""
        if self._client is not None:
            if timeout is not None:
                kwargs["timeout"] = timeout
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            return await client.request(method, url, **kwargs)

    def _unavailable(self) -> OllamaUnavailableError:
        return OllamaUnavailableError(
            f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`, "
            "or set OLLAMA_HOST to the correct address."
        )

    def _model_missing(self) -> OllamaModelNotFoundError:
        return OllamaModelNotFoundError(
            f"Ollama model '{self.model}' is not available. Run `ollama pull {self.model}`, "
            "or set OLLAMA_MODEL to a model you have installed."
        )

    async def check_ready(self) -> None:
        """Raises an OllamaError if the server is down or the model is not pulled."""
        try:
            response = await self._request("get", f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
            installed = {m["name"] for m in response.json().get("models", [])}
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise self._unavailable() from e
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise OllamaError(f"Unexpected response from Ollama at {self.host}: {e}") from e

        wanted = {self.model, f"{self.model}:latest"} if ":" not in self.model else {self.model}
        if not wanted & installed:
            raise self._model_missing()

    async def _complete_once(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        """A single attempt, with no retry. Raises a specific OllamaError subclass on failure,
        with `status_code` set on it when the failure was an HTTP response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        merged_options = {**self.default_options, **(options or {})}
        if merged_options:
            payload["options"] = merged_options

        try:
            response = await self._request("post", f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except httpx.ConnectError as e:
            raise self._unavailable() from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Ollama did not respond within {self.timeout:.0f}s (model '{self.model}')."
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise self._model_missing() from e
            error = OllamaError(
                f"Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
            error.status_code = e.response.status_code
            raise error from e
        except (ValueError, KeyError) as e:
            raise OllamaResponseError(f"Ollama returned a malformed response: {e}") from e

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        """One attempt, no retry. Kept for callers (the API server, ad hoc scripts) that want
        a plain string and don't need retry bookkeeping. Benchmark evaluations should use
        complete_with_retry() instead."""
        return await self._complete_once(prompt, system, options)

    async def complete_with_retry(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
        policy: RetryPolicy | None = None,
    ) -> CompletionResult:
        """Retries transient failures (timeouts, connection errors, 429/500/502/503) with
        exponential backoff and jitter; permanent failures (model missing, malformed response,
        other HTTP statuses) raise immediately. On final failure, the raised OllamaError carries
        `attempts` and `retried` so the caller can record them even though no result was
        returned."""
        policy = policy or self.retry_policy
        last_error: OllamaError | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                text = await self._complete_once(prompt, system, options)
                return CompletionResult(text=text, attempts=attempt, retried=attempt > 1)
            except OllamaError as e:
                e.attempts = attempt
                e.retried = attempt > 1
                last_error = e
                if not is_retryable(e) or attempt == policy.max_attempts:
                    raise
                await asyncio.sleep(backoff_delay(policy, attempt))
        raise last_error  # pragma: no cover - loop always returns or raises above


ollama_client = OllamaClient()


@asynccontextmanager
async def benchmark_session(
    client: OllamaClient,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> AsyncIterator[OllamaClient]:
    """Scopes a benchmark run: applies `timeout`/`max_retries` overrides (if given), opens a
    pooled HTTP client for the duration, and restores both settings afterwards regardless of
    how the run ends. `max_retries` is the total attempt count (1 = no retries)."""
    previous_timeout = client.timeout
    previous_policy = client.retry_policy
    if timeout is not None:
        client.timeout = timeout
    if max_retries is not None:
        client.retry_policy = dataclasses.replace(client.retry_policy, max_attempts=max_retries)
    try:
        async with client:
            yield client
    finally:
        client.timeout = previous_timeout
        client.retry_policy = previous_policy
