"""Python port of `packages/tui/test/tui-cell-size-input.test.ts`.

`TuiBase.start` asks an image-capable terminal for its cell size in pixels
(`CSI 16 t`). The reply arrives on the same input channel as keystrokes, so the
TUI must consume exactly the reply (`CSI 6 ; height ; width t`) and forward
everything else -- including a bare ESC, which shares the reply's first byte.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pi_tui.terminal_image import (
    CellDimensions,
    get_cell_dimensions,
    reset_capabilities_cache,
    set_cell_dimensions,
)
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen


class _InputRecorder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def render(self, _width: int) -> list[str]:
        return [""]

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)

    def invalidate(self) -> None:
        return None


@contextmanager
def _image_terminal() -> Iterator[None]:
    saved: dict[str, str | None] = {
        name: os.environ.get(name)
        for name in ("TERM_PROGRAM", "TERM", "GHOSTTY_RESOURCES_DIR", "TMUX", "KITTY_WINDOW_ID")
    }
    os.environ["TERM_PROGRAM"] = "ghostty"
    for name in ("TERM", "GHOSTTY_RESOURCES_DIR", "TMUX", "KITTY_WINDOW_ID"):
        os.environ.pop(name, None)
    reset_capabilities_cache()
    saved_dimensions = get_cell_dimensions()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_capabilities_cache()
        set_cell_dimensions(saved_dimensions)


@pytest.mark.asyncio
async def test_forwards_bare_escape_even_when_a_cell_size_query_was_sent_at_startup():
    with _image_terminal():
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        recorder = _InputRecorder()

        tui.set_focus(recorder)
        tui.start()

        # The query is what makes the bare-ESC case a real regression: without
        # it the TUI would never be waiting for a `CSI 6 ; h ; w t` reply.
        assert "\x1b[16t" in terminal.writes

        terminal.send_input("\x1b")

        assert recorder.inputs == ["\x1b"]
        tui.stop()


@pytest.mark.asyncio
async def test_consumes_cell_size_responses_and_still_forwards_later_user_input():
    with _image_terminal():
        set_cell_dimensions(CellDimensions(width_px=9, height_px=18))

        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        recorder = _InputRecorder()

        tui.set_focus(recorder)
        tui.start()

        terminal.send_input("\x1b[6;20;10t")
        assert recorder.inputs == []
        assert get_cell_dimensions() == CellDimensions(width_px=10, height_px=20)

        terminal.send_input("q")
        assert recorder.inputs == ["q"]
        tui.stop()


@pytest.mark.asyncio
async def test_does_not_query_cell_size_when_the_terminal_has_no_image_support():
    previous = {name: os.environ.get(name) for name in ("TERM_PROGRAM", "TERM", "TMUX", "KITTY_WINDOW_ID")}
    os.environ["TERM_PROGRAM"] = "alacritty"
    for name in ("TERM", "TMUX", "KITTY_WINDOW_ID"):
        os.environ.pop(name, None)
    reset_capabilities_cache()
    try:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.start()
        assert "\x1b[16t" not in terminal.writes
        tui.stop()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_capabilities_cache()


@pytest.mark.asyncio
async def test_ignores_a_cell_size_response_with_non_positive_dimensions():
    with _image_terminal():
        set_cell_dimensions(CellDimensions(width_px=9, height_px=18))

        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        recorder = _InputRecorder()
        tui.set_focus(recorder)
        tui.start()

        terminal.send_input("\x1b[6;0;10t")

        # Consumed (never forwarded) but the stale dimensions are kept.
        assert recorder.inputs == []
        assert get_cell_dimensions() == CellDimensions(width_px=9, height_px=18)
        tui.stop()
