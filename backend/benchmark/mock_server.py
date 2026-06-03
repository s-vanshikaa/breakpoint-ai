"""A deterministic, Ollama-compatible mock HTTP server for benchmarking the runner itself.

Why a mock server rather than real Ollama: local Ollama inference serializes work on one
CPU-bound model instance, so client-side concurrency mostly just waits in line - it can't show
whether the runner's scheduling, pooling, retry and checkpoint code actually behave correctly
under real concurrent load. This server is a real HTTP endpoint (via uvicorn, on a real local
TCP socket) but its handler is instant/async and its behavior is fully deterministic and
configurable, so it can isolate runner-level performance and reliability from local hardware.

It is not a stand-in for real model latency or throughput - see benchmark/harness.py and the
committed results for how that distinction is labeled.
"""

import asyncio
import random
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# HTTP statuses the mock can return; matches the retryable set in models.retry plus one
# representative permanent status (400) for completeness in manual testing.
_STATUS_FAILURE_KINDS = ("429", "500", "502", "503")
ALL_FAILURE_KINDS = (*_STATUS_FAILURE_KINDS, "timeout", "malformed")


@dataclass(frozen=True)
class MockConfig:
    """Fully determines this server's behavior; the same config always produces the same
    sequence of outcomes for a given request count, via a counter-seeded RNG."""

    seed: int = 0
    model: str = "mock-model"
    response_text: str = "mock response"
    # Baseline latency shape applied to every request that isn't a failure this time.
    mode: str = "success"  # "success" | "fixed_latency" | "variable_latency"
    fixed_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    # Fraction of requests that get a failure instead of the mode above; 0 disables failures.
    failure_rate: float = 0.0
    failure_kinds: tuple[str, ...] = ALL_FAILURE_KINDS
    # How long the server sleeps before responding to a request chosen to "timeout". Must be
    # set (by the caller) comfortably longer than the client's own timeout for a real
    # httpx.TimeoutException to happen - the server doesn't know the client's timeout.
    timeout_sleep_seconds: float = 30.0


def _request_rng(config: MockConfig, request_number: int) -> random.Random:
    """Deterministic per-request RNG: same seed + same request number -> same outcome, every
    time, independent of wall-clock timing or which worker handled it."""
    return random.Random(config.seed * 1_000_003 + request_number)


def build_app(config: MockConfig) -> FastAPI:
    app = FastAPI()
    counter = {"n": 0}

    @app.get("/api/tags")
    async def tags() -> dict:
        return {"models": [{"name": config.model}]}

    @app.post("/api/generate")
    async def generate() -> JSONResponse:
        counter["n"] += 1
        rng = _request_rng(config, counter["n"])

        kind = None
        if config.failure_rate > 0 and rng.random() < config.failure_rate:
            kind = rng.choice(config.failure_kinds)
        elif config.mode in ALL_FAILURE_KINDS:
            kind = config.mode  # a single fixed mode, e.g. for exercising one failure in isolation

        if kind == "timeout":
            await asyncio.sleep(config.timeout_sleep_seconds)
            # A real client with a shorter timeout never sees this; it raises first.
            return JSONResponse({"response": config.response_text})
        if kind == "malformed":
            return JSONResponse({"unexpected_field": True})  # no "response" key
        if kind in _STATUS_FAILURE_KINDS:
            return JSONResponse({"error": f"mock {kind}"}, status_code=int(kind))

        if config.mode == "fixed_latency":
            await asyncio.sleep(config.fixed_latency_ms / 1000)
        elif config.mode == "variable_latency":
            jitter = rng.uniform(0, config.latency_jitter_ms)
            await asyncio.sleep((config.fixed_latency_ms + jitter) / 1000)

        return JSONResponse({"response": config.response_text})

    return app


@dataclass
class MockServer:
    """Runs build_app(config) on a real local TCP port for the lifetime of an `async with`
    block. In-process (a uvicorn Server task in the same event loop), so there's no subprocess
    startup cost and no risk of leaking an external process if a test fails."""

    config: MockConfig
    _server: uvicorn.Server | None = field(default=None, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _port: int | None = field(default=None, init=False, repr=False)

    @property
    def host(self) -> str:
        if self._port is None:
            raise RuntimeError("MockServer is not running (use it as an async context manager).")
        return f"http://127.0.0.1:{self._port}"

    async def start(self, startup_timeout: float = 5.0) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        self._port = sock.getsockname()[1]

        app = build_app(self.config)
        # Without this, a request that's mid-"timeout" (deliberately sleeping) makes uvicorn's
        # graceful shutdown wait for it indefinitely - stop() would hang for as long as
        # config.timeout_sleep_seconds, which defeats the point of a fast, deterministic mock.
        # "critical" (not "warning"): a request in "timeout" mode is forcibly cancelled by the
        # 1s graceful-shutdown window above, which otherwise logs a benign CancelledError
        # traceback on every such request. The client already sees a real httpx timeout either
        # way; this only silences server-side log noise, not the actual behavior being tested.
        uv_config = uvicorn.Config(app, log_level="critical", timeout_graceful_shutdown=1)
        self._server = uvicorn.Server(uv_config)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))

        waited = 0.0
        while not self._server.started:
            if self._task.done():
                self._task.result()  # surface a startup exception instead of hanging
            await asyncio.sleep(0.01)
            waited += 0.01
            if waited > startup_timeout:
                raise RuntimeError("Mock server did not start within the timeout.")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            await self._task
        self._server = None
        self._task = None
        self._port = None

    async def __aenter__(self) -> "MockServer":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()


@asynccontextmanager
async def mock_server(config: MockConfig) -> AsyncIterator[MockServer]:
    server = MockServer(config)
    async with server:
        yield server
