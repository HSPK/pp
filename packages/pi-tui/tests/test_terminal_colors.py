"""Tests for OSC 11 background-color and color-scheme report parsing.

Python port of `packages/tui/test/terminal-colors.test.ts`.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui.terminal_colors import RgbColor, parse_osc11_background_color, parse_terminal_color_scheme_report
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen


class InputRecorder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def render(self, _width: int) -> list[str]:
        return []

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)

    def invalidate(self) -> None:
        return None


class TestParseOsc11BackgroundColor:
    def test_parses_16bit_osc11_rgb_responses(self) -> None:
        assert parse_osc11_background_color("\x1b]11;rgb:0000/8000/ffff\x07") == RgbColor(r=0, g=128, b=255)

    def test_parses_osc11_hex_responses(self) -> None:
        assert parse_osc11_background_color("\x1b]11;#ffffff\x1b\\") == RgbColor(r=255, g=255, b=255)
        assert parse_osc11_background_color("\x1b]11;#000000\x07") == RgbColor(r=0, g=0, b=0)

    def test_rejects_non_strict_osc11_responses(self) -> None:
        assert parse_osc11_background_color("x\x1b]11;#ffffff\x07") is None
        assert parse_osc11_background_color("\x1b]10;#ffffff\x07") is None
        assert parse_osc11_background_color("\x1b]11;#ffffff\x07x") is None

    def test_parses_48bit_hex_osc11_responses(self) -> None:
        assert parse_osc11_background_color("\x1b]11;#aaaabbbbcccc\x07") == RgbColor(r=170, g=187, b=204)

    def test_rejects_malformed_hex_osc11_responses(self) -> None:
        assert parse_osc11_background_color("\x1b]11;#abc\x07") is None
        assert parse_osc11_background_color("\x1b]11;#zzzzzz\x07") is None

    def test_rejects_malformed_rgb_osc11_responses(self) -> None:
        assert parse_osc11_background_color("\x1b]11;rgb:zz/00/00\x07") is None
        assert parse_osc11_background_color("\x1b]11;rgb:00/00\x07") is None

    def test_is_osc11_background_color_response(self) -> None:
        from pi_tui.terminal_colors import is_osc11_background_color_response

        assert is_osc11_background_color_response("\x1b]11;#ffffff\x07") is True
        assert is_osc11_background_color_response("not it") is False


class TestParseTerminalColorSchemeReport:
    def test_parses_color_scheme_reports(self) -> None:
        assert parse_terminal_color_scheme_report("\x1b[?997;1n") == "dark"
        assert parse_terminal_color_scheme_report("\x1b[?997;2n") == "light"
        assert parse_terminal_color_scheme_report("\x1b[?997;2n\x1b[?997;1n\x1b[?997;1n") == "dark"
        assert parse_terminal_color_scheme_report("\x1b[?997;1n\x1b[?997;2n\x1b[?997;2n") == "light"
        assert parse_terminal_color_scheme_report("\x1b[?997;3n") is None
        assert parse_terminal_color_scheme_report("\x1b[?996n") is None
        assert parse_terminal_color_scheme_report("x\x1b[?997;1n") is None


class TestQueryTerminalBackgroundColor:
    @pytest.mark.asyncio
    async def test_writes_osc11_query_and_resolves_parsed_rgb_reply(self) -> None:
        terminal = FakeTerminal()
        tui = TuiMainScreen(terminal)
        tui.start()
        try:
            query = asyncio.ensure_future(tui.query_terminal_background_color(1000))
            await asyncio.sleep(0)
            assert "\x1b]11;?\x07" in terminal.writes

            terminal.send_input("\x1b]11;#ffffff\x07")

            assert await query == RgbColor(r=255, g=255, b=255)
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_consumes_osc11_replies_before_listeners_and_focused_dispatch(self) -> None:
        terminal = FakeTerminal()
        tui = TuiMainScreen(terminal)
        component = InputRecorder()
        listener_inputs: list[str] = []
        tui.add_child(component)
        tui.set_focus(component)
        tui.add_input_listener(lambda data: listener_inputs.append(data) or None)
        tui.start()
        try:
            query = asyncio.ensure_future(tui.query_terminal_background_color(1000))
            await asyncio.sleep(0)

            terminal.send_input("\x1b]11;#000000\x07")

            assert await query == RgbColor(r=0, g=0, b=0)
            assert listener_inputs == []
            assert component.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_consumes_unparseable_strict_osc11_replies_and_resolves_none(self) -> None:
        terminal = FakeTerminal()
        tui = TuiMainScreen(terminal)
        component = InputRecorder()
        listener_inputs: list[str] = []
        tui.add_child(component)
        tui.set_focus(component)
        tui.add_input_listener(lambda data: listener_inputs.append(data) or None)
        tui.start()
        try:
            query = asyncio.ensure_future(tui.query_terminal_background_color(1000))
            await asyncio.sleep(0)

            terminal.send_input("\x1b]11;not-a-color\x07")

            assert await query is None
            assert listener_inputs == []
            assert component.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_dispatches_non_matching_input_normally_while_waiting(self) -> None:
        terminal = FakeTerminal()
        tui = TuiMainScreen(terminal)
        component = InputRecorder()
        listener_inputs: list[str] = []
        tui.add_child(component)
        tui.set_focus(component)
        tui.add_input_listener(lambda data: listener_inputs.append(data) or None)
        tui.start()
        try:
            settled = False

            async def _query() -> RgbColor | None:
                nonlocal settled
                result = await tui.query_terminal_background_color(1000)
                settled = True
                return result

            query = asyncio.ensure_future(_query())
            await asyncio.sleep(0)

            terminal.send_input("x")
            await asyncio.sleep(0)

            assert settled is False
            assert listener_inputs == ["x"]
            assert component.inputs == ["x"]

            terminal.send_input("\x1b]11;#ffffff\x07")
            assert await query == RgbColor(r=255, g=255, b=255)
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_keeps_consuming_late_osc11_reply_after_timeout(self) -> None:
        terminal = FakeTerminal()
        tui = TuiMainScreen(terminal)
        component = InputRecorder()
        listener_inputs: list[str] = []
        tui.add_child(component)
        tui.set_focus(component)
        tui.add_input_listener(lambda data: listener_inputs.append(data) or None)
        tui.start()
        try:
            query = asyncio.ensure_future(tui.query_terminal_background_color(1))

            # Awaiting the query is itself the wait for its 1 ms timeout; an
            # extra fixed sleep only added wall-clock time.
            assert await query is None

            terminal.send_input("\x1b]11;#ffffff\x07")

            assert listener_inputs == []
            assert component.inputs == []
        finally:
            tui.stop()
