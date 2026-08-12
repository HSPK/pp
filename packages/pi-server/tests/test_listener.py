"""Port of `packages/server/test/listener.test.ts`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from conftest import wait
from pi_server.testing.server import create_test_server
from pi_server.types import PiServerOptions


class _Listener:
    def __init__(self, address: str, start_error: Exception | None = None) -> None:
        self.address: str | None = address
        self.accept: Callable[[Any], Any] | None = None
        self.start_count = 0
        self.close_count = 0
        self._start_error = start_error

    async def start(self, accept: Callable[[Any], Any]) -> None:
        self.start_count += 1
        self.accept = accept
        if self._start_error is not None:
            raise self._start_error

    async def close(self) -> None:
        self.close_count += 1
        self.address = None


async def test_starts_and_closes_every_configured_listener() -> None:
    first = _Listener("first")
    second = _Listener("second")
    harness = create_test_server(PiServerOptions(listeners=[first, second]))

    await wait(harness.server.start())
    assert harness.server.addresses == ["first", "second"]
    assert callable(first.accept)
    assert callable(second.accept)

    await wait(harness.server.close())
    assert first.close_count == 1
    assert second.close_count == 1
    assert harness.server.addresses == []


async def test_closes_previously_started_listeners_when_startup_fails() -> None:
    first = _Listener("first")
    failure = RuntimeError("listener failed")
    second = _Listener("second", start_error=failure)
    harness = create_test_server(PiServerOptions(listeners=[first, second]))

    try:
        await wait(harness.server.start())
        raise AssertionError("expected start() to raise")
    except RuntimeError as error:
        assert error is failure
    assert first.close_count == 1
    assert second.close_count == 0
