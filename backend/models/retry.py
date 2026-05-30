"""Retry policy for transient model-call failures: exponential backoff with jitter.

Pure logic, no I/O — models/ollama_client.py drives the actual sleeping/retrying.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # total attempts, including the first; 1 disables retries
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 5.0
    jitter_seconds: float = 0.25


@dataclass(frozen=True)
class CompletionResult:
    text: str
    attempts: int  # total attempts made, including the first (1 = succeeded on the first try)
    retried: bool  # attempts > 1


# HTTP statuses that are worth retrying: rate limiting and server-side/upstream failures.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503})


def backoff_delay(policy: RetryPolicy, attempt: int, *, rand: random.Random | None = None) -> float:
    """Delay before retrying, after `attempt` (1-indexed) has just failed.

    Exponential in the attempt number, capped at max_delay_seconds, plus a little jitter so
    concurrent retries don't all land on the server at once.
    """
    rand = rand or random
    exponential = policy.base_delay_seconds * (2 ** (attempt - 1))
    delay = min(exponential, policy.max_delay_seconds)
    return delay + rand.uniform(0, policy.jitter_seconds)
