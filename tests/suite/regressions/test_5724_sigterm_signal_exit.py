"""Python port of `packages/coding-agent/test/suite/regressions/5724-sigterm-signal-exit.test.ts`.

The TypeScript regression is about Node's `signal-exit` (pulled in by
`proper-lockfile`) re-sending SIGTERM/SIGHUP when it sees no other process
listeners during the same dispatch, so `shutdown` must not unregister its
handlers while cleanup is still pending. Python has no `signal-exit`, but the
ordering is still observable and still matters: with the handlers already gone,
a second signal arriving mid-teardown is handled by the default disposition and
kills the process before cleanup finishes.

As in the 5080 port this drives a **real** `InteractiveMode` rather than a
hand-built `this`: `shutdown` suppresses every exception it raises, so a
stand-in whose `dispose`/`abort`/`drain_input` were sync where production has
coroutine functions would silently satisfy the ordering assertions. The
wrappers below delegate to the real bound methods and record *after* they
return, so a sync-where-async regression drops the step from `order`.

`runtime_host.dispose()` is the same call on both sides (here it emits
`session_shutdown` via `AgentSessionRuntime.dispose`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from interactive_harness import make_interactive_mode

from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


async def _make_mode_with_blocking_dispose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: asyncio.Future[None]
) -> tuple[InteractiveMode, list[str]]:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    order: list[str] = []

    real_dispose = mode.runtime_host.dispose
    real_drain = mode.renderer.terminal.drain_input
    real_stop = mode.renderer.stop
    real_unregister = mode._unregister_signal_handlers

    async def dispose() -> None:
        await real_dispose()
        order.append("dispose")
        # Hold cleanup open so the test can observe the state of the signal
        # handlers while `shutdown` is still mid-teardown.
        await gate

    async def drain_input(max_ms: float = 1000, idle_ms: float = 50) -> None:
        await real_drain(max_ms, idle_ms)
        order.append("drainInput")

    def stop(*args: object, **kwargs: object) -> None:
        real_stop(*args, **kwargs)
        order.append("stop")

    def unregister() -> None:
        real_unregister()
        order.append("unregister")

    monkeypatch.setattr(mode.runtime_host, "dispose", dispose)
    monkeypatch.setattr(mode.renderer.terminal, "drain_input", drain_input)
    monkeypatch.setattr(mode.renderer, "stop", stop)
    monkeypatch.setattr(mode, "_unregister_signal_handlers", unregister)
    return mode, order


async def test_keeps_signal_handlers_registered_while_signal_triggered_cleanup_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    mode, order = await _make_mode_with_blocking_dispose(tmp_path, monkeypatch, gate)

    shutdown = asyncio.ensure_future(mode.shutdown(from_signal=True))
    for _ in range(10):
        await asyncio.sleep(0)

    assert order == ["dispose"]
    assert "unregister" not in order

    gate.set_result(None)
    await shutdown

    assert order == ["dispose", "drainInput", "stop", "unregister"]
