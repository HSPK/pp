"""Python port of `packages/coding-agent/test/interactive-mode-suspend.test.ts`.

Like the TypeScript file, this calls `InteractiveMode._handle_ctrl_z` against a
hand-built stand-in for `self`.

TypeScript also asserts `setInterval(fn, 2 ** 30)` is opened for the duration of
the suspend. That keep-alive exists because Node's `process.kill` returns
immediately and the process exits once no ref'd handles remain; CPython's
`os.kill(pid, SIGTSTP)` blocks inside the stopped process, so this port has no
keep-alive handle and the corresponding assertions have no counterpart. The
SIGINT-ignoring handler and the ordering of `ui.stop()` / `ui.start()` /
`ui.request_render(True)` are asserted exactly as TypeScript does.
"""

from __future__ import annotations

import os
import signal
from typing import Any

import pytest

from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


class _FakeUi:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.render_forced: list[bool] = []

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def request_render(self, force: bool = False) -> None:
        self.calls.append("request_render")
        self.render_forced.append(force)


class _CtrlZContext:
    _handle_ctrl_z = InteractiveMode._handle_ctrl_z

    def __init__(self) -> None:
        self.ui = _FakeUi()
        self.statuses: list[str] = []

    def show_status(self, message: str) -> None:
        self.statuses.append(message)


def test_shows_a_status_message_and_skips_suspend_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _CtrlZContext()
    kills: list[tuple[int, int]] = []
    signal_calls: list[tuple[int, Any]] = []

    monkeypatch.setattr("pi_coding_agent.modes.interactive.interactive_mode.sys.platform", "win32")
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(signal, "signal", lambda sig, handler: signal_calls.append((sig, handler)))

    context._handle_ctrl_z()

    assert context.statuses == ["Suspend to background is not supported on Windows"]
    assert context.ui.calls == []
    assert signal_calls == []
    assert kills == []


def test_keeps_sigint_ignored_while_suspended_and_restores_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _CtrlZContext()
    kills: list[tuple[int, int]] = []
    installed: list[Any] = []
    sentinel_previous = object()

    def fake_signal(sig: int, handler: Any) -> Any:
        assert sig == signal.SIGINT
        installed.append(handler)
        return sentinel_previous

    monkeypatch.setattr("pi_coding_agent.modes.interactive.interactive_mode.sys.platform", "linux")
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(signal, "signal", fake_signal)

    context._handle_ctrl_z()

    # pid=0 targets the whole process group, matching TypeScript's
    # `process.kill(0, "SIGTSTP")`.
    assert kills == [(0, signal.SIGTSTP)]
    # TS additionally asserts `process.once("SIGCONT", ...)` registered a
    # handler and then invokes it by hand, because Node's `process.kill`
    # returns immediately and the restore work has to happen from a listener.
    # `os.kill(os.getpid(), SIGTSTP)` stops the calling process and only
    # returns once SIGCONT arrives, so the restore is the code that follows it
    # -- there is no handler to register or fire, and the assertions below are
    # made directly against the post-resume effects TS reaches through it.
    # SIGINT ignored before suspending, previous handler restored on resume.
    assert installed == [signal.SIG_IGN, sentinel_previous]
    assert context.ui.calls == ["stop", "start", "request_render"]
    assert context.ui.render_forced == [True]


def test_cleans_up_the_temporary_handlers_if_suspension_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _CtrlZContext()
    installed: list[Any] = []
    sentinel_previous = object()
    suspend_error = OSError("suspend failed")

    def fake_signal(sig: int, handler: Any) -> Any:
        installed.append(handler)
        return sentinel_previous

    def failing_kill(pid: int, sig: int) -> None:
        raise suspend_error

    monkeypatch.setattr("pi_coding_agent.modes.interactive.interactive_mode.sys.platform", "linux")
    monkeypatch.setattr(os, "kill", failing_kill)
    monkeypatch.setattr(signal, "signal", fake_signal)

    with pytest.raises(OSError) as excinfo:
        context._handle_ctrl_z()

    assert excinfo.value is suspend_error
    assert installed == [signal.SIG_IGN, sentinel_previous]
    assert context.ui.calls == ["stop"]
