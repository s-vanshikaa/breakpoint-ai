import httpx

from config import settings


class OllamaError(RuntimeError):
    """Base class for errors talking to the local Ollama server."""


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached."""


class OllamaModelNotFoundError(OllamaError):
    """Ollama is running but the configured model is not pulled."""


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

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
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
            raise OllamaError(
                f"Ollama did not respond within {self.timeout:.0f}s (model '{self.model}')."
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise self._model_missing() from e
            raise OllamaError(
                f"Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except (ValueError, KeyError) as e:
            raise OllamaError(f"Ollama returned a malformed response: {e}") from e


ollama_client = OllamaClient()
