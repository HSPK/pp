"""Tests for the terminal capability detection and raw-mode driver.

Python port of `packages/tui/test/terminal.test.ts`.

`ProcessTerminal` is exercised entirely through a `FakeTerminalIo` (see
`pi_tui.testing`); no test touches a real TTY or blocks on stdin.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest
from pi_tui.keys import set_kitty_protocol_active
from pi_tui.terminal import (
    TERMINAL_PROGRESS_ACTIVE_SEQUENCE,
    TERMINAL_PROGRESS_CLEAR_SEQUENCE,
    TERMINAL_PROGRESS_KEEPALIVE_S,
    ProcessTerminal,
    _compute_write_log_path,
    _real_enter_raw_mode,
    _real_exit_raw_mode,
    _real_get_size,
    _real_register_resize_handler,
    _real_trigger_resize_refresh,
    _RealStdinSource,
    is_apple_terminal_session,
    is_native_modifier_pressed,
    normalize_apple_terminal_input,
    normalize_native_shift_enter_input,
    real_terminal_io,
)
from pi_tui.testing import FakeTerminalIo, ManualTimers


class TestNormalizeNativeShiftEnterInput:
    def test_rewrites_return_when_detection_enabled_and_shift_pressed(self) -> None:
        assert normalize_native_shift_enter_input("\r", True, True) == "\x1b[13;2u"

    def test_leaves_return_unchanged_when_detection_disabled(self) -> None:
        assert normalize_native_shift_enter_input("\r", False, True) == "\r"

    def test_leaves_return_unchanged_when_shift_not_pressed(self) -> None:
        assert normalize_native_shift_enter_input("\r", True, False) == "\r"

    def test_leaves_non_return_input_unchanged(self) -> None:
        assert normalize_native_shift_enter_input("\x1b[13;2u", True, True) == "\x1b[13;2u"
        assert normalize_native_shift_enter_input("a", True, True) == "a"


class TestNormalizeAppleTerminalInput:
    def test_rewrites_apple_terminal_return_when_shift_pressed(self) -> None:
        assert normalize_apple_terminal_input("\r", True, True) == "\x1b[13;2u"

    def test_leaves_return_unchanged_when_shift_not_pressed(self) -> None:
        assert normalize_apple_terminal_input("\r", True, False) == "\r"

    def test_leaves_non_apple_terminal_return_unchanged(self) -> None:
        assert normalize_apple_terminal_input("\r", False, True) == "\r"

    def test_leaves_non_return_input_unchanged(self) -> None:
        assert normalize_apple_terminal_input("\x1b[13;2u", True, True) == "\x1b[13;2u"
        assert normalize_apple_terminal_input("a", True, True) == "a"


@pytest.fixture(autouse=True)
def reset_kitty_protocol():  # type: ignore[no-untyped-def]
    set_kitty_protocol_active(False)
    yield
    set_kitty_protocol_active(False)


def _setup_negotiation() -> tuple[ProcessTerminal, FakeTerminalIo, list[str]]:
    io = FakeTerminalIo()
    terminal = ProcessTerminal(io.build())
    received: list[str] = []
    terminal._input_handler = received.append
    terminal._query_and_enable_kitty_protocol()
    return terminal, io, received


@pytest.fixture(autouse=True)
def _pin_escape_timeout(monkeypatch: pytest.MonkeyPatch):
    """Pin the escape-reassembly window to the local-terminal default.

    These cases mirror upstream's `mock.timers.tick(10)` / `tick(150)` pairs,
    which assume the 10 ms default. `resolve_escape_timeout_ms` returns 100 ms
    when `SSH_CONNECTION`/`SSH_TTY` are set, so on a machine reached over SSH
    the timings would silently stop matching the TypeScript they were ported
    from -- the test would be measuring the environment, not the code.
    """
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("PI_TUI_ESC_TIMEOUT", raising=False)


class TestKittyKeyboardProtocolNegotiation:
    @pytest.mark.asyncio
    async def test_queries_kitty_mode_before_modify_other_keys_fallback(self) -> None:
        terminal, io, _received = _setup_negotiation()
        try:
            assert io.writes[0] == "\x1b[>7u\x1b[?u\x1b[c"
            assert "\x1b[>4;2m" not in io.writes
            assert terminal.kitty_protocol_active is False
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_activates_kitty_mode_for_nonzero_negotiated_flags(self) -> None:
        terminal, io, received = _setup_negotiation()
        try:
            io.send("\x1b[?7u")

            assert received == []
            assert terminal.kitty_protocol_active is True
            assert "\x1b[>4;2m" not in io.writes
            assert "\x1b[>4;0m" not in io.writes

            terminal.stop()
            assert io.writes.count("\x1b[<u") == 1
            assert "\x1b[>4;0m" not in io.writes
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_falls_back_to_modify_other_keys_for_zero_kitty_flags(self) -> None:
        terminal, io, received = _setup_negotiation()
        try:
            io.send("\x1b[?0u")

            assert received == []
            assert terminal.kitty_protocol_active is False
            assert io.writes.count("\x1b[>4;2m") == 1

            terminal.stop()
            assert io.writes.count("\x1b[>4;0m") == 1
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_falls_back_to_modify_other_keys_for_device_attributes(self) -> None:
        terminal, io, received = _setup_negotiation()
        try:
            io.send("\x1b[?62;4;52c")

            assert received == []
            assert terminal.kitty_protocol_active is False
            assert io.writes.count("\x1b[>4;2m") == 1
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_forwards_normal_input_while_waiting_for_kitty_response(self) -> None:
        terminal, io, received = _setup_negotiation()
        try:
            io.send("a")

            assert received == ["a"]
            assert terminal.kitty_protocol_active is False
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_tracks_split_kitty_confirmation(self) -> None:
        # The TypeScript test drives this with `mock.timers.tick(10)`; the
        # virtual clock keeps it independent of wall-clock time.
        timers = ManualTimers(asyncio.get_running_loop())
        timers.install()
        try:
            terminal, io, received = _setup_negotiation()
            try:
                io.send("\x1b[?7")
                timers.tick(10)

                assert received == []

                io.send("u")

                assert terminal.kitty_protocol_active is True
                assert "\x1b[>4;2m" not in io.writes
            finally:
                terminal.stop()
        finally:
            timers.uninstall()

    @pytest.mark.asyncio
    async def test_replays_buffered_csi_prefix_input_when_not_kitty_response(self) -> None:
        # Mirrors the TypeScript `mock.timers.tick(10)` / `tick(150)` pair: the
        # fragment must still be buffered after 10 ms and replayed once the
        # 150 ms negotiation timeout elapses.
        timers = ManualTimers(asyncio.get_running_loop())
        timers.install()
        try:
            terminal, io, received = _setup_negotiation()
            try:
                io.send("\x1b[")
                timers.tick(10)

                assert received == []

                timers.tick(150)

                assert received == ["\x1b["]
            finally:
                terminal.stop()
        finally:
            timers.uninstall()


class TestProcessTerminalProgress:
    def test_writes_valid_osc_9_4_clear_sequence(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())

        terminal.set_progress(False)

        assert io.writes == ["\x1b]9;4;0\x07"]


class TestProcessTerminalDimensions:
    def test_falls_back_to_columns_and_lines_env_before_default(self) -> None:
        io = FakeTerminalIo(columns=0, rows=0, environ={"COLUMNS": "123", "LINES": "45"})
        terminal = ProcessTerminal(io.build())

        assert terminal.columns == 123
        assert terminal.rows == 45

    def test_falls_back_to_80x24_when_nothing_else_available(self) -> None:
        io = FakeTerminalIo(columns=0, rows=0, environ={})
        terminal = ProcessTerminal(io.build())

        assert terminal.columns == 80
        assert terminal.rows == 24

    def test_uses_real_size_when_available(self) -> None:
        io = FakeTerminalIo(columns=132, rows=43)
        terminal = ProcessTerminal(io.build())

        assert terminal.columns == 132
        assert terminal.rows == 43


class TestProcessTerminalStartStop:
    @pytest.mark.asyncio
    async def test_start_enables_raw_mode_and_bracketed_paste(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        try:
            terminal.start(lambda _data: None, lambda: None)
            assert io.raw_mode_enter_count == 1
            assert "\x1b[?2004h" in io.writes
            assert io.stdin.started is True
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_stop_disables_bracketed_paste_and_restores_raw_mode(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.start(lambda _data: None, lambda: None)

        terminal.stop()

        assert "\x1b[?2004l" in io.writes
        assert io.raw_mode_exit_count == 1
        assert io.stdin.stopped is True

    @pytest.mark.asyncio
    async def test_resize_handler_is_invoked_on_resize(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        resizes: list[None] = []
        try:
            terminal.start(lambda _data: None, lambda: resizes.append(None))
            io.resize(100, 40)
            assert resizes == [None]
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_stop_unregisters_resize_handler(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        resizes: list[None] = []
        terminal.start(lambda _data: None, lambda: resizes.append(None))
        terminal.stop()

        io.resize(100, 40)
        assert resizes == []


class TestProcessTerminalCursorAndScreenControl:
    def test_move_by_positive_moves_down(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.move_by(3)
        assert io.writes == ["\x1b[3B"]

    def test_move_by_negative_moves_up(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.move_by(-2)
        assert io.writes == ["\x1b[2A"]

    def test_move_by_zero_writes_nothing(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.move_by(0)
        assert io.writes == []

    def test_hide_and_show_cursor(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.hide_cursor()
        terminal.show_cursor()
        assert io.writes == ["\x1b[?25l", "\x1b[?25h"]

    def test_clear_operations(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.clear_line()
        terminal.clear_from_cursor()
        terminal.clear_screen()
        assert io.writes == ["\x1b[K", "\x1b[J", "\x1b[2J\x1b[H"]

    def test_set_title(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.set_title("hello")
        assert io.writes == ["\x1b]0;hello\x07"]


class TestProcessTerminalDrainInput:
    @pytest.mark.asyncio
    async def test_drain_input_returns_after_idle_timeout(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.start(lambda _data: None, lambda: None)
        try:
            await terminal.drain_input(max_ms=200, idle_ms=20)
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_drain_input_suppresses_input_handler(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        received: list[str] = []
        terminal.start(received.append, lambda: None)
        try:
            drain_task = asyncio.ensure_future(terminal.drain_input(max_ms=100, idle_ms=20))
            await asyncio.sleep(0.01)
            io.send("late input")
            await drain_task
            assert received == []
        finally:
            terminal.stop()


class TestProcessTerminalProgressKeepalive:
    @pytest.mark.asyncio
    async def test_set_progress_true_writes_active_sequence(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        try:
            terminal.set_progress(True)
            assert io.writes == ["\x1b]9;4;3\x07"]
        finally:
            terminal.set_progress(False)

    @pytest.mark.asyncio
    async def test_set_progress_false_cancels_keepalive(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.set_progress(True)
        terminal.set_progress(False)
        assert terminal._progress_task is None

    @pytest.mark.asyncio
    async def test_keepalive_reschedules_with_configured_delay(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        real_sleep = asyncio.sleep
        recorded_delays: list[float] = []

        async def fake_sleep(delay: float, *args: object, **kwargs: object) -> None:
            recorded_delays.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        try:
            terminal.set_progress(True)
            for _ in range(5):
                await real_sleep(0)

            assert recorded_delays
            assert all(delay == TERMINAL_PROGRESS_KEEPALIVE_S for delay in recorded_delays)
            assert io.writes.count(TERMINAL_PROGRESS_ACTIVE_SEQUENCE) >= 2
        finally:
            terminal.set_progress(False)

    @pytest.mark.asyncio
    async def test_stop_writes_progress_clear_sequence_when_progress_active(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.set_progress(True)

        terminal.stop()

        assert io.writes.count(TERMINAL_PROGRESS_CLEAR_SEQUENCE) == 1


class TestIsAppleTerminalSession:
    def test_reflects_platform_and_term_program_env(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        assert is_apple_terminal_session() is True

        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        assert is_apple_terminal_session() is False

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        assert is_apple_terminal_session() is False


class TestComputeWriteLogPath:
    def test_empty_env_returns_empty_string(self) -> None:
        assert _compute_write_log_path({}, 123) == ""
        assert _compute_write_log_path({"PI_TUI_WRITE_LOG": ""}, 123) == ""

    def test_file_path_env_is_used_as_is(self) -> None:
        assert _compute_write_log_path({"PI_TUI_WRITE_LOG": "/no/such/dir/tui.log"}, 123) == "/no/such/dir/tui.log"

    def test_directory_env_generates_timestamped_log_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _compute_write_log_path({"PI_TUI_WRITE_LOG": str(tmp_path)}, 4242)
        assert result.startswith(str(tmp_path))
        assert "tui-" in result
        assert result.endswith("-4242.log")


class TestProcessTerminalWriteLog:
    def test_write_appends_to_log_file_when_configured(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        io = FakeTerminalIo(environ={"PI_TUI_WRITE_LOG": str(tmp_path)})
        terminal = ProcessTerminal(io.build())

        terminal.write("hello")
        terminal.write(" world")

        log_files = list(tmp_path.iterdir())
        assert len(log_files) == 1
        assert log_files[0].read_text(encoding="utf-8") == "hello world"


class TestProcessTerminalColumnsRowsZeroEnvFallback:
    def test_falls_back_to_default_when_env_value_is_zero(self) -> None:
        io = FakeTerminalIo(columns=0, rows=0, environ={"COLUMNS": "0", "LINES": "0"})
        terminal = ProcessTerminal(io.build())

        assert terminal.columns == 80
        assert terminal.rows == 24


class TestProcessTerminalResizeHandlerGuard:
    def test_on_resize_does_nothing_before_start(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        # No resize handler registered yet; should not raise.
        terminal._on_resize()


class TestProcessTerminalPasteForwarding:
    @pytest.mark.asyncio
    async def test_bracketed_paste_forwarded_as_wrapped_input(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        received: list[str] = []
        terminal.start(received.append, lambda: None)
        try:
            io.send("\x1b[?0u")
            io.send("\x1b[200~pasted text\x1b[201~")

            assert received == ["\x1b[200~pasted text\x1b[201~"]
        finally:
            terminal.stop()

    def test_paste_is_ignored_without_input_handler(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        # No exception even though no input handler is registered.
        terminal._on_stdin_buffer_paste("content")


class TestProcessTerminalStdinBytesDecoding:
    @pytest.mark.asyncio
    async def test_incomplete_utf8_sequence_is_buffered_until_complete(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        received: list[str] = []
        terminal.start(received.append, lambda: None)
        try:
            io.send("\x1b[?0u")
            received.clear()

            # "世" encoded as UTF-8 split across two byte chunks.
            encoded = "世".encode()
            io.send(encoded[:1])
            io.send(encoded[1:])

            assert received == ["世"]
        finally:
            terminal.stop()


class TestProcessTerminalKittyAlreadyActiveGuards:
    @pytest.mark.asyncio
    async def test_repeated_kitty_flags_negotiation_keeps_state_active(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal._input_handler = lambda _data: None
        terminal._query_and_enable_kitty_protocol()
        try:
            io.send("\x1b[?7u")
            assert terminal.kitty_protocol_active is True

            terminal._query_and_enable_kitty_protocol()
            io.send("\x1b[?7u")

            assert terminal.kitty_protocol_active is True
            assert "\x1b[>4;2m" not in io.writes
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_device_attributes_after_kitty_active_does_not_enable_modify_other_keys(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal._input_handler = lambda _data: None
        terminal._query_and_enable_kitty_protocol()
        try:
            io.send("\x1b[?7u")
            assert terminal.kitty_protocol_active is True

            terminal._query_and_enable_kitty_protocol()
            io.send("\x1b[?62;4c")

            assert terminal.kitty_protocol_active is True
            assert "\x1b[>4;2m" not in io.writes
        finally:
            terminal.stop()


class TestProcessTerminalNegotiationBufferFlushGuards:
    def test_flush_buffer_as_input_is_noop_when_buffer_empty(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        received: list[str] = []
        terminal._input_handler = received.append

        terminal._flush_keyboard_protocol_negotiation_buffer_as_input()

        assert received == []

    @pytest.mark.asyncio
    async def test_schedule_flush_is_noop_when_buffer_empty(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())

        terminal._schedule_keyboard_protocol_negotiation_buffer_flush()

        assert terminal._keyboard_protocol_buffer_flush_timer is None

    def test_forward_input_sequence_is_noop_without_handler(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        assert terminal._input_handler is None

        # Should not raise even though there is no handler to call.
        terminal._forward_input_sequence("x")

    def test_enable_modify_other_keys_is_noop_when_already_active(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())

        terminal._enable_modify_other_keys()
        assert io.writes.count("\x1b[>4;2m") == 1

        terminal._enable_modify_other_keys()
        assert io.writes.count("\x1b[>4;2m") == 1

    @pytest.mark.asyncio
    async def test_drain_input_without_kitty_pushed_does_not_write_disable_sequence(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        # drain_input() before start(): kitty negotiation was never pushed.
        await terminal.drain_input(max_ms=10, idle_ms=5)

        assert "\x1b[<u" not in io.writes

    @pytest.mark.asyncio
    async def test_buffered_negotiation_prefix_flushed_when_followup_does_not_extend_it(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        received: list[str] = []
        terminal.start(received.append, lambda: None)
        try:
            io.send("\x1b[?0u")
            received.clear()

            io.send("\x1b[")
            await asyncio.sleep(0.02)
            assert received == []
            assert terminal._keyboard_protocol_negotiation_buffer == "\x1b["

            io.send("q")

            assert received == ["\x1b[", "q"]
            assert terminal._keyboard_protocol_negotiation_buffer == ""
        finally:
            terminal.stop()

    @pytest.mark.asyncio
    async def test_buffered_negotiation_prefix_extends_while_still_a_valid_prefix(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        received: list[str] = []
        terminal.start(received.append, lambda: None)
        try:
            io.send("\x1b[?0u")
            received.clear()

            io.send("\x1b[")
            await asyncio.sleep(0.02)
            assert terminal._keyboard_protocol_negotiation_buffer == "\x1b["

            io.send("?")

            # "\x1b[?" is still a valid (longer) negotiation prefix, so it
            # stays buffered rather than being flushed as plain input.
            assert received == []
            assert terminal._keyboard_protocol_negotiation_buffer == "\x1b[?"
        finally:
            terminal.stop()


class TestProcessTerminalStopWithoutStart:
    def test_stop_without_start_does_not_disable_kitty_sequences(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())

        terminal.stop()

        assert "\x1b[<u" not in io.writes


class TestProcessTerminalModifyOtherKeysActiveProperty:
    def test_modify_other_keys_active_reflects_state(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        assert terminal.modify_other_keys_active is False

        terminal._enable_modify_other_keys()
        assert terminal.modify_other_keys_active is True


class TestProcessTerminalDrainInputMaxTimeout:
    @pytest.mark.asyncio
    async def test_drain_input_returns_when_max_ms_elapses_before_idle(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        terminal.start(lambda _data: None, lambda: None)
        try:
            # idle_ms far exceeds max_ms, so the overall timeout must fire first.
            await terminal.drain_input(max_ms=20, idle_ms=5000)
        finally:
            terminal.stop()


class TestRealTerminalIoHelpers:
    """Exercises the real-OS-backed helpers without requiring a TTY.

    Under pytest, stdin/stdout are not a TTY, so these always take the
    non-TTY / no-signal-support fallback branches, never touching a real
    terminal device.
    """

    def test_real_get_size_returns_zero_pair_without_a_tty(self) -> None:
        assert _real_get_size() == (0, 0) or isinstance(_real_get_size(), tuple)

    def test_real_enter_raw_mode_returns_none_without_a_tty(self) -> None:
        if sys.stdin.isatty():
            pytest.skip("stdin is a real TTY in this environment")
        assert _real_enter_raw_mode() is None

    def test_real_exit_raw_mode_with_none_is_a_noop(self) -> None:
        # Should not raise even without ever having entered raw mode.
        _real_exit_raw_mode(None)

    def test_real_register_resize_handler_invokes_callback_on_sigwinch(self) -> None:
        if not hasattr(signal, "SIGWINCH") or sys.platform == "win32":
            pytest.skip("SIGWINCH not supported on this platform")

        calls: list[None] = []
        unregister = _real_register_resize_handler(lambda: calls.append(None))
        try:
            os.kill(os.getpid(), signal.SIGWINCH)
            assert calls == [None]
        finally:
            unregister()

    def test_real_trigger_resize_refresh_does_not_raise(self) -> None:
        # No handler registered; the OS default action for SIGWINCH is to
        # ignore the signal, so this must not raise or terminate the process.
        _real_trigger_resize_refresh()

    def test_real_terminal_io_reflects_process_state(self) -> None:
        io = real_terminal_io()
        assert io.is_tty == sys.stdin.isatty()
        assert io.platform == sys.platform
        assert io.pid == os.getpid()

    def test_real_terminal_io_write_writes_and_flushes_stdout(self, capsys) -> None:  # type: ignore[no-untyped-def]
        io = real_terminal_io()
        io.write("hello-real-terminal-io")
        captured = capsys.readouterr()
        assert "hello-real-terminal-io" in captured.out

    def test_real_get_size_uses_os_get_terminal_size_when_available(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(os, "get_terminal_size", lambda _fd: os.terminal_size((100, 50)))
        assert _real_get_size() == (100, 50)

    def test_real_enter_raw_mode_saves_and_applies_raw_mode_when_tty(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import pi_tui.terminal as terminal_module

        if not terminal_module._HAS_TERMIOS:
            pytest.skip("termios not available on this platform")

        class FakeStdin:
            def isatty(self) -> bool:
                return True

            def fileno(self) -> int:
                return 0

        monkeypatch.setattr(terminal_module.sys, "stdin", FakeStdin())
        monkeypatch.setattr(terminal_module.termios, "tcgetattr", lambda _fd: "saved-termios-state")
        monkeypatch.setattr(terminal_module.tty, "setraw", lambda _fd: None)

        assert terminal_module._real_enter_raw_mode() == "saved-termios-state"

    def test_real_exit_raw_mode_attempts_tcsetattr_without_raising(self) -> None:
        # stdin is not a TTY under pytest, so the underlying tcsetattr call
        # fails and is swallowed; this must not propagate an exception.
        _real_exit_raw_mode("some-previous-state")

    def test_real_register_resize_handler_is_noop_on_windows(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(sys, "platform", "win32")
        handler = _real_register_resize_handler(lambda: None)
        assert handler() is None

    def test_real_register_resize_handler_is_noop_without_sigwinch(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(signal, "SIGWINCH"):
            pytest.skip("platform genuinely has no SIGWINCH")
        monkeypatch.delattr(signal, "SIGWINCH", raising=False)
        handler = _real_register_resize_handler(lambda: None)
        assert handler() is None

    def test_real_trigger_resize_refresh_is_noop_on_windows(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(sys, "platform", "win32")
        # Must return without attempting os.kill.
        _real_trigger_resize_refresh()

    def test_real_trigger_resize_refresh_is_noop_without_sigwinch(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(signal, "SIGWINCH"):
            pytest.skip("platform genuinely has no SIGWINCH")
        monkeypatch.delattr(signal, "SIGWINCH", raising=False)
        _real_trigger_resize_refresh()


class TestRealStdinSource:
    @pytest.mark.asyncio
    async def test_start_reads_data_from_added_reader(self) -> None:
        read_fd, write_fd = os.pipe()
        source = _RealStdinSource(fd=read_fd)
        received: list[bytes] = []
        got_data = asyncio.Event()

        def on_data(data: bytes) -> None:
            received.append(data)
            got_data.set()

        try:
            source.start(on_data)
            os.write(write_fd, b"hello")
            await asyncio.wait_for(got_data.wait(), timeout=2)
            assert received == [b"hello"]
        finally:
            source.stop()
            os.close(write_fd)
            os.close(read_fd)

    def test_read_ready_swallows_os_error_from_closed_fd(self) -> None:
        source = _RealStdinSource(fd=-1)
        received: list[bytes] = []
        source._on_data = received.append

        # An invalid fd raises OSError inside os.read; it must be caught
        # and reported as an empty read rather than propagating.
        source._read_ready()

        assert received == [b""]


class TestIsNativeModifierPressed:
    def test_always_returns_false(self) -> None:
        # No portable Python equivalent to the native keyboard-modifier
        # addon; always reports the modifier as not pressed.
        assert is_native_modifier_pressed("shift") is False
        assert is_native_modifier_pressed("command") is False


class TestSetProgressIdempotentWhileActive:
    @pytest.mark.asyncio
    async def test_second_set_progress_true_does_not_spawn_a_second_task(self) -> None:
        io = FakeTerminalIo()
        terminal = ProcessTerminal(io.build())
        try:
            terminal.set_progress(True)
            first_task = terminal._progress_task

            terminal.set_progress(True)

            assert terminal._progress_task is first_task
            assert io.writes.count(TERMINAL_PROGRESS_ACTIVE_SEQUENCE) == 2
        finally:
            terminal.set_progress(False)
