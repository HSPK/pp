"""Python port of `packages/coding-agent/test/management-http.test.ts`."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from pi_coding_agent.utils import management_http
from pi_coding_agent.utils.management_http import FetchRetryOptions, fetch_with_retry


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


class _VirtualClock:
    """The single time source `fetch_with_retry` reads; advancing it is instant.

    The retry loop measures its shared budget with `time.monotonic()`. Burning
    that budget with a real `asyncio.sleep` would make the assertions depend on
    wall-clock timing under parallel test load; advancing this clock by hand
    makes the remaining budget exact.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _VirtualClock:
    virtual = _VirtualClock()
    # Rebinds only the `time` name inside `management_http`, so the real
    # `time.monotonic` the event loop depends on is untouched.
    monkeypatch.setattr(management_http, "time", virtual)
    return virtual


async def test_retries_a_transient_transport_failure_once():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= 2:
            raise httpx.ConnectError("fetch failed", request=request)
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        response = await fetch_with_retry("https://example.test", client=client)

    assert response.is_success
    assert len(calls) == 3


async def test_shares_the_timeout_budget_across_attempts(clock: _VirtualClock):
    """TypeScript builds one `AbortSignal.timeout(timeoutMs)` outside the retry
    loop, so the budget is spent across *all* attempts, not restarted per
    attempt. Here the same is observed through the deadline: an attempt that
    burns part of the budget leaves only the remainder for the retry."""
    timeouts: list[float | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout", {}).get("connect")
        timeouts.append(timeout)
        if len(timeouts) == 1:
            clock.advance(0.15)
            raise httpx.ConnectError("fetch failed", request=request)
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        response = await fetch_with_retry(
            "https://example.test", options=FetchRetryOptions(timeout_ms=1000), client=client
        )

    assert response.is_success
    assert len(timeouts) == 2
    # A per-attempt budget would hand the retry the full second again.
    assert timeouts[0] == pytest.approx(1.0)
    assert timeouts[1] == pytest.approx(0.85)


async def test_raises_once_the_shared_timeout_budget_is_exhausted(clock: _VirtualClock):
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        clock.advance(0.1)
        raise httpx.ConnectError("fetch failed", request=request)

    async with _client(handler) as client:
        with pytest.raises(httpx.TimeoutException, match="Timed out after 50ms"):
            await fetch_with_retry("https://example.test", options=FetchRetryOptions(timeout_ms=50), client=client)

    # The retry never starts: the budget is checked before the attempt.
    assert len(calls) == 1


async def test_retries_transient_http_responses_and_returns_the_successful_response():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        response = await fetch_with_retry("https://example.test", client=client)

    assert response.is_success
    assert len(calls) == 2


async def test_does_not_retry_caller_cancellation():
    """TypeScript's case creates an `AbortController`, calls `.abort()` on it
    *before* ever calling `fetchWithRetry`, and asserts `fetch` is never
    invoked (`fetchMock).not.toHaveBeenCalled()`). This port has no
    `signal`/`AbortController` parameter: caller cancellation is
    `asyncio.CancelledError` delivered by `asyncio.Task.cancel()` (see the
    README's async conventions). The equivalent "already cancelled before the
    call starts" state is a task cancelled before it is ever scheduled to
    run: the very first checkpoint inside `fetch_with_retry` then raises
    `CancelledError` before the transport handler is reached, matching the
    TypeScript "zero calls" assertion exactly rather than merely approximating
    it."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:

        async def run() -> httpx.Response:
            return await fetch_with_retry("https://example.test", client=client)

        task = asyncio.ensure_future(run())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(calls) == 0
