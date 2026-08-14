"""Python port of `packages/tui/test/tui-overlay-style-leak.test.ts`.

The TypeScript suite reads the italic attribute off xterm.js cells; this port
reads it off `MiniTerminalModel` (see `pi_tui.testing`), which now tracks the
italic SGR attribute per cell for exactly this purpose.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui import OverlayOptions
from pi_tui.tui_main_screen import TuiMainScreen


class _StaticLines:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, _width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        return None


class _StaticOverlay:
    def __init__(self, line: str) -> None:
        self.line = line

    def render(self, _width: int) -> list[str]:
        return [self.line]

    def invalidate(self) -> None:
        return None


async def _render_and_flush(tui: TuiMainScreen, terminal: FakeTerminal, model: MiniTerminalModel) -> None:
    tui.request_render(True)
    await asyncio.sleep(0.03)
    model.feed("".join(terminal.writes))
    terminal.writes.clear()


@pytest.mark.asyncio
async def test_does_not_leak_styles_when_a_trailing_reset_sits_beyond_the_last_visible_column():
    width = 20
    base_line = f"\x1b[3m{'X' * width}\x1b[23m"

    terminal = FakeTerminal(width, 6)
    model = MiniTerminalModel(width, 6)
    tui = TuiMainScreen(terminal)
    tui.add_child(_StaticLines([base_line, "INPUT"]))
    tui.start()
    await _render_and_flush(tui, terminal, model)

    assert model.cell_italic(1, 0) is False
    tui.stop()


@pytest.mark.asyncio
async def test_does_not_leak_styles_when_overlay_slicing_drops_trailing_sgr_resets():
    width = 20
    base_line = f"\x1b[3m{'X' * width}\x1b[23m"

    terminal = FakeTerminal(width, 6)
    model = MiniTerminalModel(width, 6)
    tui = TuiMainScreen(terminal)
    tui.add_child(_StaticLines([base_line, "INPUT"]))

    tui.show_overlay(_StaticOverlay("OVR"), OverlayOptions(row=0, col=5, width=3))
    tui.start()
    await _render_and_flush(tui, terminal, model)

    assert model.cell_italic(1, 0) is False
    tui.stop()
