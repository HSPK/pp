"""`setInterval`/`clearInterval` equivalents built on ``asyncio``.

Extracted from ``packages/tui/src/`` usages of Node's timer API so that
components (the loader, auto-scroll, ...) can share one implementation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class IntervalHandle:
    """Emulates `setInterval`/`clearInterval` by chaining `asyncio` `call_later` calls.

    Node's `setInterval` works without an event loop being "current"; asyncio's
    does not. When no loop is running the handle is created in a cancelled-like
    state and simply never fires, which keeps component construction outside a
    loop (notably in tests) from raising.
    """

    def __init__(self, callback: Callable[[], None], interval_s: float) -> None:
        self._callback = callback
        self._interval_s = interval_s
        self._cancelled = False
        self._handle: asyncio.TimerHandle | None = None
        loop = _running_loop()
        if loop is not None:
            self._handle = loop.call_later(interval_s, self._tick)

    @property
    def scheduled(self) -> bool:
        return self._handle is not None and not self._cancelled

    def _tick(self) -> None:
        if self._cancelled:
            return
        self._callback()
        if self._cancelled:
            return
        loop = _running_loop()
        if loop is not None:
            self._handle = loop.call_later(self._interval_s, self._tick)

    def cancel(self) -> None:
        self._cancelled = True
        if self._handle is not None:
            self._handle.cancel()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def schedule_interval(callback: Callable[[], None], interval_s: float) -> IntervalHandle:
    """Emulate `setInterval` using a self-rescheduling `asyncio` timer."""
    return IntervalHandle(callback, interval_s)
