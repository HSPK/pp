"""Python port of `packages/coding-agent/test/suite/regressions/5080-signal-shutdown-extension-cleanup.test.ts`.

The TypeScript test invokes `InteractiveMode.prototype.shutdown` with a
hand-built `this`. This port drives a **real** `InteractiveMode` (real
`AgentSessionRuntime`, real `AgentSession`, real `SessionManager`, pi-tui's
`FakeTerminal`) instead, because `shutdown` is exactly the kind of code a
stand-in object cannot police: every step runs under
`contextlib.suppress(Exception)`, so a stub whose method is sync where
production is a coroutine function -- or missing entirely -- would be swallowed
and the ordering assertions would still pass. The recording wrappers below call
through to the real methods, so the real async-ness is what is exercised.

`runtime_host.dispose()` is the call the regression is about on both sides:
here as there, it is what emits the `session_shutdown` extension event (see
`AgentSessionRuntime.dispose`, which emits it with `reason="quit"`).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from harness import assistant_msg, user_msg
from interactive_harness import make_interactive_mode

from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime
from pi_coding_agent.core.config import APP_NAME
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


async def _make_mode_recording_shutdown_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[InteractiveMode, list[str]]:
    """Real interactive mode whose teardown steps append to a shared list.

    Each wrapper delegates to the real bound method, so this records ordering
    without replacing behaviour or changing shape.
    """
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    order: list[str] = []

    real_dispose = mode.runtime_host.dispose
    real_drain = mode.renderer.terminal.drain_input
    real_stop = mode.renderer.stop

    # Each step is recorded *after* the real call returns, not before. That
    # ordering matters: `shutdown` runs every step under
    # `contextlib.suppress(Exception)`, so a collaborator that stopped being a
    # coroutine function would make `await` raise, be suppressed, and -- with
    # record-first wrappers -- still leave a complete-looking order list.
    async def dispose() -> None:
        await real_dispose()
        order.append("dispose")

    async def drain_input(max_ms: float = 1000, idle_ms: float = 50) -> None:
        await real_drain(max_ms, idle_ms)
        order.append("drainInput")

    def stop(*args: object, **kwargs: object) -> None:
        real_stop(*args, **kwargs)
        order.append("stop")

    monkeypatch.setattr(mode.runtime_host, "dispose", dispose)
    monkeypatch.setattr(mode.renderer.terminal, "drain_input", drain_input)
    monkeypatch.setattr(mode.renderer, "stop", stop)
    return mode, order


def test_shutdown_collaborators_are_coroutine_functions_in_production() -> None:
    """Guard the shape `shutdown` assumes, since it suppresses every failure.

    `shutdown` does `await self.runtime_host.dispose()`, `await
    self.session.abort()` and `await self.ui.terminal.drain_input(...)` inside
    `contextlib.suppress(Exception)`. If any of those stopped being a coroutine
    function the `await` would raise `TypeError`, be suppressed, and the step
    would silently not happen -- with no test failure anywhere. This pins them.
    """
    assert inspect.iscoroutinefunction(AgentSessionRuntime.dispose)
    assert inspect.iscoroutinefunction(AgentSession.abort)
    assert not inspect.iscoroutinefunction(InteractiveMode._unregister_signal_handlers)


async def test_signal_triggered_shutdown_disposes_before_terminal_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode, order = await _make_mode_recording_shutdown_order(tmp_path, monkeypatch)

    await mode.shutdown(from_signal=True)

    assert order == ["dispose", "drainInput", "stop"]
    assert mode.is_shutting_down is True


async def test_interactive_quit_stops_the_tui_before_disposing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mode, order = await _make_mode_recording_shutdown_order(tmp_path, monkeypatch)

    await mode.shutdown()

    assert order == ["drainInput", "stop", "dispose"]


async def test_interactive_quit_prints_a_resume_hint_for_persisted_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mode, order = await _make_mode_recording_shutdown_order(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    session_manager = mode.session_manager
    # The session file is written lazily -- `_persist_entry` defers until the
    # transcript has an assistant message -- and `format_resume_command`
    # requires it to exist on disk.
    session_manager.append_message(user_msg("hello"))
    session_manager.append_message(assistant_msg("hi"))
    session_file = session_manager.get_session_file()
    assert session_file is not None and Path(session_file).exists()
    assert session_manager.is_persisted()
    session_id = session_manager.get_session_id()

    await mode.shutdown()

    assert order == ["drainInput", "stop", "dispose"]
    assert capsys.readouterr().out == f"To resume this session: {APP_NAME} --session {session_id}\n"


async def test_interactive_quit_prints_no_resume_hint_for_an_unwritten_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """TypeScript's default fake session manager has no session file.

    Rather than fake the manager, leave the real session empty: nothing was
    appended, so no file was written and the same `format_resume_command`
    branch (`not os.path.exists(session_file)`) declines to print a hint.
    """
    mode, _order = await _make_mode_recording_shutdown_order(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    session_file = mode.session_manager.get_session_file()
    assert session_file is not None and not Path(session_file).exists()

    await mode.shutdown()

    assert "To resume this session:" not in capsys.readouterr().out


async def test_signal_triggered_shutdown_does_not_print_a_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mode, _order = await _make_mode_recording_shutdown_order(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    mode.session_manager.append_message(user_msg("hello"))
    mode.session_manager.append_message(assistant_msg("hi"))
    session_file = mode.session_manager.get_session_file()
    assert session_file is not None and Path(session_file).exists()

    await mode.shutdown(from_signal=True)

    assert "To resume this session:" not in capsys.readouterr().out


async def test_reentrant_shutdown_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mode, order = await _make_mode_recording_shutdown_order(tmp_path, monkeypatch)
    mode.is_shutting_down = True

    await mode.shutdown(from_signal=True)

    assert order == []
