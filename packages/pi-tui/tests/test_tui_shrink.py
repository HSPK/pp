"""Python port of `packages/tui/test/tui-shrink.test.ts`.

The TypeScript suite drives an xterm.js-backed `VirtualTerminal`; this port uses
`FakeTerminal` plus `MiniTerminalModel` from `pi_tui.testing`, which models the
subset of escape sequences `TuiMainScreen` emits (see `tests/test_tui_main_screen.py`).
"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui_main_screen import TuiMainScreen


class _Lines:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, _width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        return None


async def _wait_for_render() -> None:
    await asyncio.sleep(0.03)


@pytest.mark.asyncio
async def test_clears_all_rendered_lines_when_content_shrinks_to_zero():
    terminal = FakeTerminal(40, 10)
    model = MiniTerminalModel(40, 10)
    tui = TuiMainScreen(terminal)
    content = _Lines(["first", "second", "third"])
    tui.add_child(content)
    tui.start()
    await _wait_for_render()
    fed = len(terminal.writes)
    model.feed("".join(terminal.writes[:fed]))

    viewport = model.viewport()
    assert any("first" in line for line in viewport)
    assert any("second" in line for line in viewport)
    assert any("third" in line for line in viewport)

    tui.clear()
    tui.request_render()
    await _wait_for_render()
    model.feed("".join(terminal.writes[fed:]))

    viewport = model.viewport()
    assert not any("first" in line for line in viewport), "first line should be cleared"
    assert not any("second" in line for line in viewport), "second line should be cleared"
    assert not any("third" in line for line in viewport), "third line should be cleared"

    tui.stop()
