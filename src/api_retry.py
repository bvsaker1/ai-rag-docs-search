"""
Retrying API calls: which errors to retry, how long to wait, and when to give up.

    Fatal (raise FatalAPIError at once):  400 bad request, 401 bad key, 402 out of credit,
                                          403 forbidden, 404 unknown model/endpoint
    Rate limited (429):                   wait as long as the Retry-After header says (capped),
                                          else back off
    Anything else (5xx, timeouts,         exponential backoff with jitter:
    connection/SSL errors, bad JSON):     base · 2^(attempt-1) + random(0, base), capped

Works with httpx errors (Cohere reranking) and OpenAI SDK errors (LLM calls through
instructor, which may wrap them); the status code is found anywhere in the exception chain.
The OpenAI SDK also retries 429/5xx itself (twice, honoring Retry-After) before an error
reaches call_with_retries.
"""

import email.utils
import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

FATAL_STATUSES = {400, 401, 402, 403, 404}
RATE_LIMITED = 429
MAX_WAIT_SECONDS = 60.0


class FatalAPIError(RuntimeError):
    """An API error that retrying can't fix (bad key, no credit, bad request)."""

    def __init__(self, status: int, error: Exception) -> None:
        hints = {401: "check OPENROUTER_API_KEY", 402: "add OpenRouter credit", 403: "check the key's permissions",
                 404: "check the model name"}
        hint = f" — {hints[status]}" if status in hints else ""
        super().__init__(f"HTTP {status}{hint}: {error}")
        self.status = status


def _chain(error: BaseException):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def status_code(error: BaseException) -> Optional[int]:
    """The HTTP status of an httpx or OpenAI SDK error anywhere in the exception chain."""
    for e in _chain(error):
        status = getattr(e, "status_code", None)
        response = getattr(e, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def retry_after(error: BaseException, now: Optional[float] = None) -> Optional[float]:
    """Seconds to wait per the Retry-After header (seconds or an HTTP date), if present."""
    for e in _chain(error):
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None)
        value = headers.get("retry-after") if headers is not None else None
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(value) if value else None
            if parsed is not None:
                return max(0.0, parsed.timestamp() - (now if now is not None else time.time()))
    return None


def is_rate_limited(error: BaseException) -> bool:
    return status_code(error) == RATE_LIMITED


def backoff(attempt: int, base: float, rng: random.Random = random) -> float:
    """Exponential backoff with jitter for attempt 1, 2, ..., capped at MAX_WAIT_SECONDS."""
    return min(MAX_WAIT_SECONDS, base * 2 ** (attempt - 1) + rng.uniform(0, base))


def call_with_retries(fn: Callable[[], T], max_attempts: int, base_seconds: float = 1.0,
                      on_retry: Optional[Callable[[int, Exception, float], None]] = None,
                      sleep: Callable[[float], None] = time.sleep, rng: random.Random = random) -> T:
    """Call `fn` up to `max_attempts` times. Fatal errors raise FatalAPIError immediately;
    on the last attempt the error itself is raised. `on_retry(attempt, error, wait)` runs
    before each wait."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            status = status_code(e)
            if status in FATAL_STATUSES:
                raise FatalAPIError(status, e) from e
            if attempt == max_attempts:
                raise
            hinted = retry_after(e) if status == RATE_LIMITED else None
            wait = min(MAX_WAIT_SECONDS, hinted) if hinted is not None else backoff(attempt, base_seconds, rng)
            if on_retry:
                on_retry(attempt, e, wait)
            sleep(wait)
    raise AssertionError("unreachable")
