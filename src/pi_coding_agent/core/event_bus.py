"""A tiny pub/sub channel for cross-extension notifications.

Ported from ``packages/coding-agent/src/core/event-bus.ts``.

Extensions can publish arbitrary events on named channels
(``pi.events.emit(channel, data)``) and other extensions can subscribe
(``pi.events.on(channel, handler)``) without knowing about each other. A
handler that raises must never take down the emitter or block other
handlers on the same channel, so errors are caught and logged per handler.

Dispatch is synchronous, matching Node's ``EventEmitter.emit``: listeners run
before ``emit`` returns, and an async handler's body runs up to its first
``await`` too. Deferring handlers to plain tasks would reorder observable
effects, and would drop events entirely when ``emit`` is called with no
running event loop. Async handlers are therefore started eagerly
(:func:`asyncio.eager_task_factory`), so only the continuation after the first
``await`` is deferred -- exactly what the JavaScript does.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

EventBusHandler = Callable[[Any], Awaitable[None] | None]

_pending_handlers: set[asyncio.Task[None]] = set()
"""Strong references to in-flight async handlers; the loop only holds tasks weakly."""


class EventBus(Protocol):
    def emit(self, channel: str, data: Any) -> None: ...

    def on(self, channel: str, handler: EventBusHandler) -> Callable[[], None]:
        """Subscribe `handler` to `channel`; returns an unsubscribe callback."""
        ...


class EventBusController(EventBus, Protocol):
    def clear(self) -> None: ...


class _EventBusImpl:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventBusHandler]] = {}

    def emit(self, channel: str, data: Any) -> None:
        for handler in list(self._handlers.get(channel, ())):
            self._dispatch(channel, handler, data)

    def on(self, channel: str, handler: EventBusHandler) -> Callable[[], None]:
        self._handlers.setdefault(channel, []).append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers.get(channel)
            if handlers is not None and handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def clear(self) -> None:
        self._handlers.clear()

    def _dispatch(self, channel: str, handler: EventBusHandler, data: Any) -> None:
        try:
            result = handler(data)
        except Exception as err:
            self._report(channel, err)
            return
        if not inspect.isawaitable(result):
            return
        try:
            loop = asyncio.get_running_loop()
            # Eager start reproduces JS: an async handler runs synchronously
            # up to its first `await`, and only the tail is scheduled.
            task = asyncio.eager_task_factory(loop, self._await_result(channel, result))
            _pending_handlers.add(task)
            task.add_done_callback(_pending_handlers.discard)
        except RuntimeError:
            # No running loop, so the continuation can never run. Close the
            # coroutine rather than leaking an "never awaited" warning.
            close = getattr(result, "close", None)
            if close is not None:
                close()
            self._report(channel, RuntimeError("no running event loop for async handler"))

    async def _await_result(self, channel: str, result: Awaitable[None]) -> None:
        try:
            await result
        except Exception as err:
            self._report(channel, err)

    def _report(self, channel: str, err: BaseException) -> None:
        print(f"Event handler error ({channel}): {err}", file=sys.stderr)


def create_event_bus() -> EventBusController:
    return _EventBusImpl()
