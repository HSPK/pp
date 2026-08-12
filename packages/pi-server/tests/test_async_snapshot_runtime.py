"""Covers the asynchronous half of `PiSessionRuntime.snapshot`.

There is no upstream counterpart to this file, and that absence is the point.
`packages/server/src/types.ts` declares::

    snapshot(): SessionSnapshot | Promise<SessionSnapshot>

and `packages/server/src/sessions.ts` consumes it with a plain ``await``, which
in TypeScript resolves both branches, so upstream needs no test to tell them
apart. Python has no such luck: ``await`` on a plain ``dict`` raises, so the
port has to branch on :func:`inspect.isawaitable` (``_await_maybe``). Drop that
branch in the *other* direction -- write ``runtime.snapshot()`` with no await at
all -- and every existing test still passes, because the only runtime shipped in
this repo (`pi_server.testing.service.TestSessionRuntime`) returns a ``dict``
synchronously. A real coroutine-returning runtime would then blow up with
``'coroutine' object is not subscriptable``.

So these tests keep the whole real object graph -- real `PiServer`, real Unix
transport, real client -- and vary only the axis the fake cannot reach: they
subclass the shipped runtime so that `snapshot` is a genuine coroutine function.
"""

from __future__ import annotations

import inspect
from typing import Any

from conftest import attach, wait
from pi_server.testing.service import TestServerService, TestSessionRuntime


class AsyncSnapshotRuntime(TestSessionRuntime):
    """The shipped runtime, but `snapshot` returns a coroutine like the protocol allows."""

    __test__ = False

    async def snapshot(self) -> dict[str, Any]:  # type: ignore[override]
        return super().snapshot()


class AsyncSnapshotService(TestServerService):
    """The shipped service, handing out runtimes whose `snapshot` must be awaited."""

    __test__ = False

    def _acquire(self, id: str) -> TestSessionRuntime:
        stored = self.sessions.get(id)
        if stored is None:
            raise RuntimeError(f"Unknown session: {id}")
        self.locked.add(id)
        runtime = AsyncSnapshotRuntime(stored, lambda: self.locked.discard(id))
        self.runtimes.setdefault(id, []).append(runtime)
        return runtime


def test_the_double_really_is_async_where_the_shipped_one_is_not() -> None:
    """Guards the guard: if this fake ever went sync the tests below would prove nothing."""
    assert inspect.iscoroutinefunction(AsyncSnapshotRuntime.snapshot)
    assert not inspect.iscoroutinefunction(TestSessionRuntime.snapshot)
    # Only `snapshot` is overridden; everything else stays the shipped implementation.
    for name in ("prompt", "steer", "abort", "set_model", "set_thinking", "subscribe", "dispose", "get_phase"):
        assert getattr(AsyncSnapshotRuntime, name) is getattr(TestSessionRuntime, name)


async def test_create_awaits_an_asynchronous_runtime_snapshot(harness: Any) -> None:
    """`_create` validates `snapshot["id"]`, which needs a resolved dict, not a coroutine."""
    service = AsyncSnapshotService()
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())

    response = await wait(client.request({"command": "create", "name": "async-snap"}))

    assert response["ok"] is True
    session = response["result"]["session"]
    assert session["id"] == service.last_created_id
    assert session["name"] == "async-snap"
    assert session["phase"] == "idle"
    assert session["attached"] is True
    assert session["locked"] is True


async def test_attach_awaits_an_asynchronous_runtime_snapshot(harness: Any) -> None:
    """The second `snapshot()` call site, reached through `attach` on an existing session."""
    service = AsyncSnapshotService()
    service.seed("session-1")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())

    session = await attach(client, "session-1")

    assert session["id"] == "session-1"
    assert session["phase"] == "idle"
    assert session["locked"] is True
    assert session["transcript"] == []


async def test_broadcast_snapshots_resolve_for_an_asynchronous_runtime(harness: Any) -> None:
    """Runtime-driven snapshot events go through the same await path."""
    service = AsyncSnapshotService()
    service.seed("session-1")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    await attach(client, "session-1")
    message_index = len(client.messages)

    runtime = service.latest_runtime("session-1")
    runtime.set_phase("turn")
    runtime.emit_snapshot()

    message = await wait(
        client.next_from(
            message_index,
            lambda message: message["type"] == "event" and message["event"]["type"] == "session_snapshot",
        )
    )
    assert message["event"]["snapshot"]["phase"] == "turn"
    assert message["event"]["snapshot"]["id"] == "session-1"
