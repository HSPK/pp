"""Bounded-retry HTTP for management requests.

Ported from ``packages/coding-agent/src/utils/management-http.ts``.

Transport-level helper for *idempotent* management requests (version checks,
catalogs, downloads). It must not be used for agent/model operations: those can
fail after the request has started and are retried by their semantic caller.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_MAX_RETRIES = 2


@dataclass
class FetchRetryOptions:
    max_retries: int | None = None
    """Additional attempts after the initial request. Defaults to two."""
    retry_on_status: bool | None = None
    """Retry transient HTTP responses as well as transport failures. Defaults to true."""
    timeout_ms: int | None = None
    """Overall time budget shared by every attempt."""


async def fetch_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    options: FetchRetryOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    """Fetch ``url``, retrying transport failures and transient statuses."""
    options = options or FetchRetryOptions()
    max_retries = (
        DEFAULT_MAX_RETRIES
        if options.max_retries is None or not math.isfinite(options.max_retries)
        else max(0, math.floor(options.max_retries))
    )
    retry_on_status = True if options.retry_on_status is None else options.retry_on_status

    # TypeScript builds one `AbortSignal.timeout(timeoutMs)` before the retry
    # loop and reuses it for every attempt, so `timeout_ms` is the *overall*
    # budget rather than a per-attempt one. httpx has no shared signal, so the
    # budget is tracked as a deadline and each attempt gets only what is left.
    deadline: float | None = None
    if options.timeout_ms is not None and options.timeout_ms > 0:
        deadline = time.monotonic() + options.timeout_ms / 1000

    def remaining_timeout() -> httpx.Timeout | None:
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            raise httpx.TimeoutException(f"Timed out after {options.timeout_ms}ms")
        return httpx.Timeout(left)

    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=remaining_timeout())
    try:
        attempt = 0
        while True:
            timeout = remaining_timeout()
            try:
                response = await http_client.request(method, url, headers=headers or {}, timeout=timeout)
                should_retry = (
                    retry_on_status and response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries
                )
                if not should_retry:
                    return response
                await response.aclose()
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= max_retries:
                    raise
            attempt += 1
    finally:
        if owns_client:
            await http_client.aclose()


__all__ = ["DEFAULT_MAX_RETRIES", "RETRYABLE_STATUS_CODES", "FetchRetryOptions", "fetch_with_retry"]
