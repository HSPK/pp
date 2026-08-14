"""Python port of the `TUI Kitty image cleanup` block of
`packages/tui/test/tui-render.test.ts`.

The rest of that suite lives in `tests/test_tui_main_screen.py`; these cases are
separated because they all need image capabilities and fixed cell dimensions
forced on for the duration of the test.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from pi_tui.components.image import Image, ImageOptions, ImageTheme
from pi_tui.terminal_image import (
    CellDimensions,
    ImageDimensions,
    TerminalCapabilities,
    encode_kitty,
    reset_capabilities_cache,
    set_capabilities,
    set_cell_dimensions,
)
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen, delete_kitty_image


class _TestComponent:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or []

    def render(self, _width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        return None


async def wait_render() -> None:
    await asyncio.sleep(0.03)


@pytest.fixture
def kitty_terminal() -> Iterator[None]:
    set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
    set_cell_dimensions(CellDimensions(width_px=10, height_px=10))
    try:
        yield
    finally:
        reset_capabilities_cache()
        set_cell_dimensions(CellDimensions(width_px=9, height_px=18))


def _image(max_width_cells: int, size_px: int) -> Image:
    return Image(
        "AAAA",
        "image/png",
        ImageTheme(fallback_color=lambda value: value),
        ImageOptions(max_width_cells=max_width_cells),
        ImageDimensions(width_px=size_px, height_px=size_px),
    )


class TestKittyImageCleanup:
    @pytest.mark.asyncio
    async def test_clears_reserved_kitty_image_rows_before_drawing_appended_placements(
        self, kitty_terminal: None
    ) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["before"])
        tui.add_child(component)
        tui.start()
        await wait_render()
        terminal.writes.clear()

        image_lines = _image(2, 20).render(40)
        image_sequence = image_lines[0]
        component.lines = ["before", *image_lines, "after"]
        tui.request_render()
        await wait_render()

        writes = "".join(terminal.writes)
        assert f"\x1b[2K\r\n\x1b[2K\x1b[1A{image_sequence}\x1b[1B" in writes, (
            "reserved rows should be cleared before the image placement is drawn"
        )
        assert f"{image_sequence}\r\n\x1b[2K" not in writes, (
            "reserved row clears must not run after the image placement is drawn"
        )
        tui.stop()

    @pytest.mark.asyncio
    async def test_falls_back_to_full_redraw_when_kitty_image_pre_clear_would_scroll(
        self, kitty_terminal: None
    ) -> None:
        terminal = FakeTerminal(40, 2)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["before"])
        tui.add_child(component)
        tui.start()
        await wait_render()
        redraws_before_image = tui.full_redraws
        terminal.writes.clear()

        component.lines = ["before", *_image(3, 30).render(40), "after"]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws > redraws_before_image, "unsafe image pre-clear should force a full redraw"
        assert "\x1b[2J" in "".join(terminal.writes), "fallback should clear and fully redraw"
        tui.stop()

    @pytest.mark.asyncio
    async def test_reserves_kitty_image_rows_before_drawing_during_full_redraw_fallbacks(
        self, kitty_terminal: None
    ) -> None:
        terminal = FakeTerminal(40, 5)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["l0", "l1", "l2", "l3", "l4"])
        tui.add_child(component)
        tui.start()
        await wait_render()
        redraws_before_image = tui.full_redraws
        terminal.writes.clear()

        image_lines = _image(3, 30).render(40)
        image_sequence = image_lines[0]
        component.lines = ["l0", "l1", "l2", "l3", "l4", *image_lines, "after"]
        tui.request_render()
        await wait_render()

        writes = "".join(terminal.writes)
        assert tui.full_redraws > redraws_before_image, "scrolling image append should force a full redraw"
        assert f"\r\n\r\n\x1b[2A{image_sequence}\x1b[2B" in writes, (
            "full redraw should reserve visible image rows before drawing the placement"
        )
        assert f"{image_sequence}\r\n\x1b[0m" not in writes, (
            "full redraw must not write reserved padding rows after drawing the placement"
        )
        tui.stop()

    @pytest.mark.asyncio
    async def test_does_not_use_cursor_up_placement_for_images_taller_than_the_viewport(
        self, kitty_terminal: None
    ) -> None:
        terminal = FakeTerminal(40, 5)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["before"])
        tui.add_child(component)
        tui.start()
        await wait_render()
        terminal.writes.clear()

        image_lines = _image(6, 60).render(40)
        image_sequence = image_lines[0]
        assert len(image_lines) > terminal.rows, "test image should exceed the viewport height"

        component.lines = ["before", *image_lines, "after"]
        tui.request_render(force=True)
        await wait_render()

        writes = "".join(terminal.writes)
        assert image_sequence in writes, "image placement should be drawn"
        assert f"\x1b[{len(image_lines) - 1}A{image_sequence}" not in writes, (
            "taller-than-viewport images must keep the #4461 first-row placement path"
        )
        tui.stop()

    @pytest.mark.asyncio
    async def test_deletes_changed_image_ids_before_drawing_moved_placements(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent()
        tui.add_child(component)

        old_image = encode_kitty("AAAA", columns=2, rows=2, image_id=42, move_cursor=False)
        component.lines = ["top", old_image]
        tui.start()
        await wait_render()
        terminal.writes.clear()

        new_image = encode_kitty("BBBB", columns=2, rows=1, image_id=42, move_cursor=False)
        component.lines = [new_image, ""]
        tui.request_render()
        await wait_render()

        writes = "".join(terminal.writes)
        delete_index = writes.find(delete_kitty_image(42))
        draw_index = writes.find(new_image)
        assert delete_index >= 0, "changed old image should be deleted"
        assert draw_index >= 0, "new image should be drawn"
        assert delete_index < draw_index, "old image must be deleted before the new placement is drawn"
        tui.stop()

    @pytest.mark.asyncio
    async def test_redraws_image_lines_when_an_earlier_reserved_image_row_changes(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent()
        tui.add_child(component)

        image = encode_kitty("AAAA", columns=2, rows=2, image_id=88, move_cursor=False)
        component.lines = ["", image]
        tui.start()
        await wait_render()
        terminal.writes.clear()

        component.lines = ["covered", image]
        tui.request_render()
        await wait_render()

        writes = "".join(terminal.writes)
        delete_index = writes.find(delete_kitty_image(88))
        draw_index = writes.find(image)
        assert delete_index >= 0, "image should be deleted when a reserved row changes"
        assert draw_index >= 0, "unchanged image line should be redrawn after deleting the placement"
        assert delete_index < draw_index, "old placement must be deleted before the image line is redrawn"
        assert "\x1b[2J" not in writes, "reserved row changes should not force a full redraw"
        tui.stop()

    @pytest.mark.asyncio
    async def test_deletes_previously_rendered_image_ids_during_full_redraws(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent()
        tui.add_child(component)

        component.lines = [encode_kitty("AAAA", columns=2, rows=2, image_id=77, move_cursor=False)]
        tui.start()
        await wait_render()
        terminal.writes.clear()

        component.lines = ["plain text"]
        tui.request_render(force=True)
        await wait_render()

        writes = "".join(terminal.writes)
        delete_index = writes.find(delete_kitty_image(77))
        clear_index = writes.find("\x1b[2J")
        assert delete_index >= 0, "previous image should be deleted during full redraw"
        assert clear_index >= 0, "full redraw should clear the screen"
        assert delete_index < clear_index, "old image should be deleted before the screen is cleared"
        tui.stop()
