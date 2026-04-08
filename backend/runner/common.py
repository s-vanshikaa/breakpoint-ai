import asyncio
import sys
from collections.abc import Awaitable, Callable

from attacks.loader import DatasetError
from models.ollama_client import OllamaError, ollama_client


class ResultsError(RuntimeError):
    """A stored results file is missing or malformed."""


async def ensure_ollama_ready() -> None:
    """Fail fast (before any test runs) if Ollama or the model is unavailable."""
    await ollama_client.check_ready()


def run_cli(main: Callable[[], Awaitable[None]]) -> None:
    """Runs an async CLI entry point, printing expected failures as one-line errors."""
    try:
        asyncio.run(main())
    except (OllamaError, DatasetError, ResultsError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
