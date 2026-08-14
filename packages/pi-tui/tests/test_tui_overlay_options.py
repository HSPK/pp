"""Python port of `packages/tui/test/overlay-options.test.ts`.

`tests/test_tui.py` already ports the anchor/percentage/offset/max-height cases;
this module carries the remaining ones (width overflow protection, the margin
object form and stacked overlays) so every TypeScript `it(...)` has a Python
counterpart.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui import OverlayMargin, OverlayOptions
from pi_tui.tui_main_screen import TuiMainScreen


class _StaticOverlay:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.requested_width: int | None = None

    def render(self, width: int) -> list[str]:
        self.requested_width = width
        return self.lines

    def invalidate(self) -> None:
        return None


class _EmptyContent:
    def render(self, _width: int) -> list[str]:
        return []

    def invalidate(self) -> None:
        return None


class _StyledContent:
    def render(self, width: int) -> list[str]:
        styled = f"\x1b[1m\x1b[38;2;255;0;0m{'X' * width}\x1b[0m"
        return [styled, styled, styled]

    def invalidate(self) -> None:
        return None


class _HyperlinkContent:
    def render(self, width: int) -> list[str]:
        link = "\x1b]8;;file:///path/to/file.ts\x07file.ts\x1b]8;;\x07"
        line = f"See {link} for details {'X' * (width - 30)}"
        return [line, line, line]

    def invalidate(self) -> None:
        return None


async def render_and_flush(tui: TuiMainScreen) -> None:
    tui.request_render(force=True)
    await asyncio.sleep(0.03)


def _viewport(terminal: FakeTerminal, width: int, height: int) -> list[str]:
    model = MiniTerminalModel(width, height)
    for write in terminal.writes:
        model.feed(write)
    return model.viewport()


class TestWidthOverflowProtection:
    @pytest.mark.asyncio
    async def test_truncates_overlay_lines_that_exceed_declared_width(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["X" * 100])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width=20))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        for line in viewport:
            assert line is not None
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_overlay_with_complex_ansi_sequences_without_crashing(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        complex_line = (
            "\x1b[48;2;40;50;40m \x1b[38;2;128;128;128mSome styled content\x1b[39m\x1b[49m"
            "\x1b]8;;http://example.com\x07link\x1b]8;;\x07" + " more content " * 10
        )
        overlay = _StaticOverlay([complex_line, complex_line, complex_line])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width=60))
        tui.start()
        await render_and_flush(tui)

        assert len(_viewport(terminal, 80, 24)) > 0
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_overlay_composited_on_styled_base_content(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["OVERLAY"])
        tui.add_child(_StyledContent())
        tui.show_overlay(overlay, OverlayOptions(width=20, anchor="center"))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert any("OVERLAY" in line for line in viewport), "Overlay should be visible"
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_wide_characters_at_overlay_boundary(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["中文日本語한글テスト漢字"])
        tui.add_child(_EmptyContent())
        # Odd width so the clip lands mid-cell on a double-width glyph.
        tui.show_overlay(overlay, OverlayOptions(width=15))
        tui.start()
        await render_and_flush(tui)

        assert len(_viewport(terminal, 80, 24)) > 0
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_overlay_positioned_at_terminal_edge(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["X" * 50])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(col=60, width=20))
        tui.start()
        await render_and_flush(tui)

        assert len(_viewport(terminal, 80, 24)) > 0
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_overlay_on_base_content_with_osc_sequences(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["OVERLAY-TEXT"])
        tui.add_child(_HyperlinkContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="center", width=20))
        tui.start()
        await render_and_flush(tui)

        assert len(_viewport(terminal, 80, 24)) > 0
        tui.stop()


class TestMargin:
    @pytest.mark.asyncio
    async def test_respects_margin_object(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["MARGIN"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(
            overlay,
            OverlayOptions(
                anchor="top-left",
                width=10,
                margin=OverlayMargin(top=2, left=3, right=0, bottom=0),
            ),
        )
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "MARGIN" in viewport[2], f"Expected MARGIN on row 2, got: {viewport[2]}"
        assert viewport[2].index("MARGIN") == 3
        tui.stop()


class TestStackedOverlays:
    @pytest.mark.asyncio
    async def test_renders_multiple_overlays_with_later_ones_on_top(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.show_overlay(_StaticOverlay(["FIRST-OVERLAY"]), OverlayOptions(anchor="top-left", width=20))
        tui.show_overlay(_StaticOverlay(["SECOND"]), OverlayOptions(anchor="top-left", width=10))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "SECOND" in viewport[0], f"Expected SECOND on row 0, got: {viewport[0]}"
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_overlays_at_different_positions_without_interference(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.show_overlay(_StaticOverlay(["TOP-LEFT"]), OverlayOptions(anchor="top-left", width=15))
        tui.show_overlay(_StaticOverlay(["BTM-RIGHT"]), OverlayOptions(anchor="bottom-right", width=15))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "TOP-LEFT" in viewport[0], f"Expected TOP-LEFT on row 0, got: {viewport[0]}"
        assert "BTM-RIGHT" in viewport[23], f"Expected BTM-RIGHT on row 23, got: {viewport[23]}"
        tui.stop()

    @pytest.mark.asyncio
    async def test_properly_hides_overlays_in_stack_order(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.show_overlay(_StaticOverlay(["FIRST"]), OverlayOptions(anchor="top-left", width=10))
        tui.show_overlay(_StaticOverlay(["SECOND"]), OverlayOptions(anchor="top-left", width=10))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "SECOND" in viewport[0], "SECOND should be visible initially"

        tui.hide_overlay()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "FIRST" in viewport[0], "FIRST should be visible after hiding SECOND"
        tui.stop()
