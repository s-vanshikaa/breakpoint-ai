import asyncio

import pytest

from benchmark.mock_server import ALL_FAILURE_KINDS, MockConfig, MockServer
from models.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFoundError,
    classify_error,
)


async def with_server(config: MockConfig, calls: int, client_timeout: float = 1.0):
    async with MockServer(config) as server:
        client = OllamaClient(host=server.host, model=config.model, timeout=client_timeout)
        outcomes = []
        for i in range(calls):
            try:
                text = await client.complete(f"job {i}")
                outcomes.append(("ok", text))
            except OllamaError as e:
                outcomes.append((classify_error(e), str(e)))
    return outcomes


class TestModes:
    def test_success_returns_configured_text(self):
        config = MockConfig(seed=1, response_text="hello")
        outcomes = asyncio.run(with_server(config, 1))
        assert outcomes == [("ok", "hello")]

    def test_check_ready_reports_the_configured_model(self):
        async def go():
            config = MockConfig(seed=1, model="my-mock")
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model="my-mock", timeout=1.0)
                await client.check_ready()  # must not raise

        asyncio.run(go())

    def test_fixed_latency_actually_delays_the_response(self):
        import time

        async def go():
            config = MockConfig(seed=1, mode="fixed_latency", fixed_latency_ms=100)
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model=config.model, timeout=2.0)
                start = time.perf_counter()
                await client.complete("hi")
                return time.perf_counter() - start

        elapsed = asyncio.run(go())
        assert elapsed >= 0.09

    def test_variable_latency_varies_but_stays_bounded(self):
        import time

        async def go():
            config = MockConfig(
                seed=1, mode="variable_latency", fixed_latency_ms=10, latency_jitter_ms=40
            )
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model=config.model, timeout=2.0)
                elapsed = []
                for i in range(8):
                    start = time.perf_counter()
                    await client.complete(f"job {i}")
                    elapsed.append(time.perf_counter() - start)
                return elapsed

        elapsed = asyncio.run(go())
        assert all(0.009 <= e <= 0.2 for e in elapsed)
        assert len(set(round(e, 3) for e in elapsed)) > 1  # not all identical

    def test_malformed_raises_response_error(self):
        config = MockConfig(seed=1, mode="malformed")
        outcomes = asyncio.run(with_server(config, 1))
        assert outcomes[0][0] == "invalid_response"

    @pytest.mark.parametrize("status", ["429", "500", "502", "503"])
    def test_http_status_modes_are_classified_as_model_failure(self, status):
        config = MockConfig(seed=1, mode=status)
        outcomes = asyncio.run(with_server(config, 1))
        assert outcomes[0][0] == "model_failure"

    def test_timeout_mode_triggers_a_real_client_timeout(self):
        config = MockConfig(seed=1, mode="timeout", timeout_sleep_seconds=1.5)
        outcomes = asyncio.run(with_server(config, 1, client_timeout=0.2))
        assert outcomes[0][0] == "timeout"

    def test_404_style_model_missing_is_not_something_the_mock_produces_by_default(self):
        # check_ready against a mismatched model name should behave like real Ollama: missing.
        async def go():
            config = MockConfig(seed=1, model="mock-model")
            async with MockServer(config) as server:
                client = OllamaClient(host=server.host, model="a-different-model", timeout=1.0)
                with pytest.raises(OllamaModelNotFoundError):
                    await client.check_ready()

        asyncio.run(go())


class TestDeterminism:
    def test_same_seed_produces_the_same_outcome_sequence(self):
        # timeout_sleep_seconds is kept short: the client times out and moves on regardless,
        # but a short sleep also lets the server finish that in-flight handler quickly when
        # MockServer.stop() waits for graceful shutdown, instead of the 30s default.
        config = MockConfig(
            seed=7, failure_rate=0.4, failure_kinds=ALL_FAILURE_KINDS, timeout_sleep_seconds=0.5
        )
        first = asyncio.run(with_server(config, 30, client_timeout=0.05))
        second = asyncio.run(with_server(config, 30, client_timeout=0.05))
        assert [kind for kind, _ in first] == [kind for kind, _ in second]

    def test_different_seeds_produce_different_sequences(self):
        config_a = MockConfig(seed=1, failure_rate=0.5, failure_kinds=("500",))
        config_b = MockConfig(seed=2, failure_rate=0.5, failure_kinds=("500",))
        a = asyncio.run(with_server(config_a, 30))
        b = asyncio.run(with_server(config_b, 30))
        assert [k for k, _ in a] != [k for k, _ in b]

    def test_failure_rate_is_approximately_respected_over_many_requests(self):
        config = MockConfig(seed=3, failure_rate=0.3, failure_kinds=("500",))
        outcomes = asyncio.run(with_server(config, 200))
        failure_fraction = sum(1 for kind, _ in outcomes if kind != "ok") / len(outcomes)
        assert 0.2 <= failure_fraction <= 0.4

    def test_zero_failure_rate_never_fails(self):
        config = MockConfig(seed=1, failure_rate=0.0)
        outcomes = asyncio.run(with_server(config, 50))
        assert all(kind == "ok" for kind, _ in outcomes)


class TestServerLifecycle:
    def test_start_stop_is_reusable_across_instances(self):
        async def go():
            for _ in range(3):
                config = MockConfig(seed=1)
                async with MockServer(config) as server:
                    client = OllamaClient(host=server.host, model=config.model, timeout=1.0)
                    await client.complete("hi")

        asyncio.run(go())  # must not raise (e.g. port reuse issues)

    def test_host_is_unavailable_before_start(self):
        server = MockServer(MockConfig(seed=1))
        with pytest.raises(RuntimeError):
            _ = server.host
