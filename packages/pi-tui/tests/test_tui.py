"""Tests for `TuiBase`: overlay stack, focus management, render scheduling.

Python port of representative cases from `packages/tui/test/overlay-options.test.ts`,
`overlay-non-capturing.test.ts`, and `overlay-short-content.test.ts`. The cases
not covered here live in `tests/test_tui_overlay_non_capturing.py` and
`tests/test_tui_overlay_options.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pi_tui.component import Container
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui import (
    OverlayMargin,
    OverlayOptions,
    OverlayUnfocusOptions,
    TuiBase,
    TuiInputListenerResult,
    TuiStopOptions,
    composite_tui_line,
    is_viewport_tui,
)
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


class _FocusableOverlay:
    def __init__(self, lines: list[str]) -> None:
        self.focused = False
        self.inputs: list[str] = []
        self.lines = lines
        #: Optional hook invoked with each input char, for scripting focus
        #: transitions from within `handle_input` (mirrors the TS test
        #: helpers that override `handleInput` per-test).
        self.on_input: Callable[[str], None] | None = None

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)
        if self.on_input is not None:
            self.on_input(data)

    def render(self, _width: int) -> list[str]:
        return self.lines

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


class TestCompositeTuiLine:
    def test_composites_overlay_over_shorter_base_line(self) -> None:
        result = composite_tui_line("hello", "OVERLAY", 2, 7, 20)
        assert "OVERLAY" in result

    def test_pads_and_truncates_to_total_width(self) -> None:
        result = composite_tui_line("", "abc", 0, 3, 5)
        # visible width must never exceed total_width
        from pi_tui.utils import visible_width

        assert visible_width(result) <= 5

    def test_truncates_final_result_when_it_exceeds_a_degenerate_total_width(self) -> None:
        # A pathological (non-positive) total_width can never be satisfied by the
        # padding math above, forcing the final `sliceByColumn` clamp.
        from pi_tui.utils import visible_width

        result = composite_tui_line("abc", "X", 1, 1, 0)
        assert visible_width(result) <= 0
        assert result == ""


class TestOverlayWidthOptions:
    @pytest.mark.asyncio
    async def test_truncates_overlay_lines_exceeding_declared_width(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["X" * 100])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width=20))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert len(viewport) == 24
        tui.stop()

    @pytest.mark.asyncio
    async def test_renders_at_percentage_of_terminal_width(self) -> None:
        terminal = FakeTerminal(100, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["test"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width="50%"))
        tui.start()
        await render_and_flush(tui)

        assert overlay.requested_width == 50
        tui.stop()

    @pytest.mark.asyncio
    async def test_respects_min_width_over_smaller_percentage(self) -> None:
        terminal = FakeTerminal(100, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["test"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width="10%", min_width=30))
        tui.start()
        await render_and_flush(tui)

        assert overlay.requested_width == 30
        tui.stop()


class TestOverlayAnchorPositioning:
    @pytest.mark.asyncio
    async def test_positions_overlay_at_top_left(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["TOP-LEFT"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="top-left", width=10))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert viewport[0].startswith("TOP-LEFT")
        tui.stop()

    @pytest.mark.asyncio
    async def test_positions_overlay_at_bottom_right(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["BTM-RIGHT"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="bottom-right", width=10))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        last_row = viewport[23]
        assert "BTM-RIGHT" in last_row
        assert last_row.rstrip().endswith("BTM-RIGHT")
        tui.stop()

    @pytest.mark.asyncio
    async def test_positions_overlay_at_top_center(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["CENTERED"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="top-center", width=10))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        first_row = viewport[0]
        assert "CENTERED" in first_row
        col_index = first_row.index("CENTERED")
        assert 30 <= col_index <= 40
        tui.stop()


class TestOverlayLayoutOptions:
    @pytest.mark.asyncio
    async def test_applies_offset_x_and_offset_y_from_anchor_position(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["OFFSET"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="top-left", width=10, offset_x=10, offset_y=5))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "OFFSET" in viewport[5]
        assert viewport[5].index("OFFSET") == 10
        tui.stop()

    @pytest.mark.asyncio
    async def test_row_percent_zero_positions_at_top(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["TOP"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width=10, row="0%"))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "TOP" in viewport[0]
        tui.stop()

    @pytest.mark.asyncio
    async def test_row_percent_hundred_positions_at_bottom(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["BOTTOM"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width=10, row="100%"))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "BOTTOM" in viewport[23]
        tui.stop()

    @pytest.mark.asyncio
    async def test_col_percent_positions_at_percentage_of_width(self) -> None:
        terminal = FakeTerminal(100, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["PCT"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(width=10, row="50%", col="50%"))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 100, 24)
        found_row = next((i for i, line in enumerate(viewport) if "PCT" in line), -1)
        assert 10 <= found_row <= 13
        tui.stop()

    @pytest.mark.asyncio
    async def test_row_and_col_override_anchor(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["ABSOLUTE"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="bottom-right", row=3, col=5, width=10))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "ABSOLUTE" in viewport[3]
        assert viewport[3].index("ABSOLUTE") == 5
        tui.stop()

    @pytest.mark.asyncio
    async def test_truncates_overlay_to_max_height(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(max_height=3))
        tui.start()
        await render_and_flush(tui)

        content = "\n".join(_viewport(terminal, 80, 24))
        assert "Line 1" in content
        assert "Line 3" in content
        assert "Line 4" not in content
        assert "Line 5" not in content
        tui.stop()

    @pytest.mark.asyncio
    async def test_truncates_overlay_to_max_height_percent(self) -> None:
        terminal = FakeTerminal(80, 10)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay([f"L{i}" for i in range(1, 11)])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(max_height="50%"))
        tui.start()
        await render_and_flush(tui)

        content = "\n".join(_viewport(terminal, 80, 10))
        assert "L1" in content
        assert "L5" in content
        assert "L6" not in content
        tui.stop()


class TestOverlayMargin:
    @pytest.mark.asyncio
    async def test_clamps_negative_margins_to_zero(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["NEG-MARGIN"])

        tui.add_child(_EmptyContent())
        tui.show_overlay(
            overlay,
            OverlayOptions(anchor="top-left", width=12, margin=OverlayMargin(top=-5, left=-10, right=0, bottom=0)),
        )
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert viewport[0].startswith("NEG-MARGIN")
        tui.stop()

    @pytest.mark.asyncio
    async def test_respects_margin_as_number(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _StaticOverlay(["MARGIN"])
        tui.add_child(_EmptyContent())
        tui.show_overlay(overlay, OverlayOptions(anchor="top-left", width=10, margin=5))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert "MARGIN" not in viewport[0]
        assert "MARGIN" not in viewport[4]
        assert "MARGIN" in viewport[5]
        assert viewport[5].index("MARGIN") == 5
        tui.stop()


class TestOverlayShortContent:
    @pytest.mark.asyncio
    async def test_renders_overlay_when_content_shorter_than_terminal_height(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_StaticOverlay(["Line 1", "Line 2", "Line 3"]))
        tui.show_overlay(_StaticOverlay(["OVERLAY_TOP", "OVERLAY_MID", "OVERLAY_BOT"]))
        tui.start()
        await asyncio.sleep(0.03)

        viewport = _viewport(terminal, 80, 24)
        assert any("OVERLAY" in line for line in viewport)
        tui.stop()


class TestOverlayNonCapturingFocus:
    @pytest.mark.asyncio
    async def test_non_capturing_overlay_preserves_focus_on_creation(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            await render_and_flush(tui)
            assert editor.focused is True
            assert overlay.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_focus_transfers_focus_to_overlay(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.focus()
            await render_and_flush(tui)
            assert editor.focused is False
            assert overlay.focused is True
            assert handle.is_focused() is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_unfocus_restores_previous_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.focus()
            handle.unfocus(None)
            await render_and_flush(tui)
            assert editor.focused is True
            assert overlay.focused is False
            assert handle.is_focused() is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_set_hidden_false_does_not_auto_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.set_hidden(True)
            handle.set_hidden(False)
            await render_and_flush(tui)
            assert editor.focused is True
            assert overlay.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_hide_when_not_focused_does_not_change_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.hide()
            await render_and_flush(tui)
            assert editor.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_hide_when_focused_restores_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.focus()
            handle.hide()
            await render_and_flush(tui)
            assert editor.focused is True
            assert overlay.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_capturing_overlay_removed_with_noncapturing_below_restores_editor(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        non_capturing = _FocusableOverlay(["NC"])
        capturing = _FocusableOverlay(["CAP"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
            handle = tui.show_overlay(capturing)
            assert capturing.focused is True
            handle.hide()
            await render_and_flush(tui)
            assert editor.focused is True
            assert non_capturing.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_sub_overlay_cleanup_then_hide_overlay_restores_focus_and_input(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        timer = _FocusableOverlay(["TIMER"])
        controller = _FocusableOverlay(["CTRL"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))
            tui.show_overlay(controller)
            assert controller.focused is True
            assert editor.focused is False
            timer_handle.hide()
            tui.hide_overlay()
            await render_and_flush(tui)
            assert editor.focused is True
            assert controller.focused is False
            assert timer.focused is False
            terminal.send_input("x")
            await render_and_flush(tui)
            assert editor.inputs == ["x"]
            assert controller.inputs == []
            assert timer.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_removed_focused_child_overlay_not_parent_fallback(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        child = _FocusableOverlay(["CHILD"])
        parent = _FocusableOverlay(["PARENT"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            child_handle = tui.show_overlay(child, OverlayOptions(non_capturing=True))
            child_handle.focus()
            parent_handle = tui.show_overlay(parent)
            assert parent.focused is True

            child_handle.hide()
            parent_handle.hide()
            terminal.send_input("x")
            await render_and_flush(tui)

            assert editor.inputs == ["x"]
            assert child.inputs == []
            assert parent.inputs == []
            assert editor.focused is True
        finally:
            tui.stop()


class TestOverlayHandleAndStackGuards:
    @pytest.mark.asyncio
    async def test_hide_called_twice_is_a_noop_the_second_time(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        handle = tui.show_overlay(_StaticOverlay(["OVERLAY"]))
        handle.hide()
        # Second call: entry already removed from the stack, must not raise or
        # attempt to remove it again.
        handle.hide()
        assert tui.has_overlay() is False

    @pytest.mark.asyncio
    async def test_set_hidden_to_current_value_is_a_noop(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        handle = tui.show_overlay(_StaticOverlay(["OVERLAY"]))
        assert handle.is_hidden() is False
        # Already visible (hidden=False); setting to False again is a no-op.
        handle.set_hidden(False)
        assert handle.is_hidden() is False
        handle.set_hidden(True)
        assert handle.is_hidden() is True
        # Already hidden; setting to True again is a no-op.
        handle.set_hidden(True)
        assert handle.is_hidden() is True

    def test_hide_overlay_on_empty_stack_is_a_noop(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        assert tui.has_overlay() is False
        # No overlay has ever been shown; must not raise (empty-stack guard).
        tui.hide_overlay()
        assert tui.has_overlay() is False


class TestOverlayNoOpGuards:
    @pytest.mark.asyncio
    async def test_focus_on_hidden_overlay_is_noop(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.set_hidden(True)
            handle.focus()
            await render_and_flush(tui)
            assert editor.focused is True
            assert handle.is_focused() is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_focus_after_hide_is_noop(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.hide()
            handle.focus()
            await render_and_flush(tui)
            assert editor.focused is True
            assert handle.is_focused() is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_unfocus_when_overlay_lacks_focus_is_noop(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.unfocus(None)
            await render_and_flush(tui)
            assert editor.focused is True
            assert overlay.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_unfocus_with_null_prefocus_clears_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            handle = tui.show_overlay(overlay)
            assert overlay.focused is True
            # No unfocus options: fall back to the overlay's preFocus, which
            # is null here because nothing was focused before showOverlay.
            handle.unfocus(None)
            assert overlay.focused is False
            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == []
            assert handle.is_focused() is False
        finally:
            tui.stop()


class TestOverlayFocusCyclePrevention:
    @pytest.mark.asyncio
    async def test_explicit_unfocus_target_cycles_between_three_overlays_and_editor(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        a = _FocusableOverlay(["A"])
        b = _FocusableOverlay(["B"])
        c = _FocusableOverlay(["C"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            a_handle = tui.show_overlay(a)
            b_handle = tui.show_overlay(b)
            c_handle = tui.show_overlay(c)

            a_handle.focus()
            terminal.send_input("a")
            await render_and_flush(tui)
            b_handle.focus()
            terminal.send_input("b")
            await render_and_flush(tui)
            c_handle.focus()
            terminal.send_input("c")
            await render_and_flush(tui)
            c_handle.unfocus(OverlayUnfocusOptions(target=editor))
            terminal.send_input("e")
            await render_and_flush(tui)
            a_handle.focus()
            terminal.send_input("A")
            await render_and_flush(tui)
            a_handle.unfocus(OverlayUnfocusOptions(target=editor))
            terminal.send_input("E")
            await render_and_flush(tui)

            assert a.inputs == ["a", "A"]
            assert b.inputs == ["b"]
            assert c.inputs == ["c"]
            assert editor.inputs == ["e", "E"]
            assert editor.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_hiding_focused_overlay_falls_back_to_next_frontmost(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        a = _FocusableOverlay(["A"])
        b = _FocusableOverlay(["B"])
        c = _FocusableOverlay(["C"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            a_handle = tui.show_overlay(a)
            b_handle = tui.show_overlay(b)
            tui.show_overlay(c)
            a_handle.focus()
            b_handle.focus()
            b_handle.set_hidden(True)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert a.inputs == ["x"]
            assert c.inputs == []
            assert a.focused is True
        finally:
            tui.stop()


class TestBlockedOverlayFocusRestore:
    """Python port of the "blocked" focus-restore scenarios from
    `overlay-non-capturing.test.ts`: when focus moves away from a capturing
    overlay to a component elsewhere in the tree, the overlay's restore
    eligibility is "blocked" by that component rather than cleared outright,
    so the overlay regains focus once the blocking chain is abandoned.
    """

    @pytest.mark.asyncio
    async def test_blocked_replacement_can_move_focus_internally_before_overlay_restore(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        base = Container()
        editor = _FocusableOverlay(["EDITOR"])
        first_replacement = _FocusableOverlay(["FIRST"])
        second_replacement = _FocusableOverlay(["SECOND"])
        overlay = _FocusableOverlay(["OVERLAY"])

        def _overlay_input(data: str) -> None:
            if data == "b":
                tui.set_focus(first_replacement)

        def _first_input(data: str) -> None:
            if data == "n":
                tui.set_focus(second_replacement)

        def _second_input(data: str) -> None:
            if data == "\r":
                # Remove `second_replacement` (the current `blocked_by`) from
                # the mount tree while redirecting focus elsewhere; since it
                # is no longer mounted, focus should resume to the overlay
                # instead of `editor`.
                base.clear()
                base.add_child(editor)
                tui.set_focus(editor)

        overlay.on_input = _overlay_input
        first_replacement.on_input = _first_input
        second_replacement.on_input = _second_input
        base.add_child(editor)
        base.add_child(first_replacement)
        base.add_child(second_replacement)
        tui.add_child(base)
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(overlay)
            terminal.send_input("b")
            await render_and_flush(tui)
            terminal.send_input("n")
            await render_and_flush(tui)
            terminal.send_input("2")
            terminal.send_input("\r")
            await render_and_flush(tui)

            assert overlay.inputs == ["b"]
            assert first_replacement.inputs == ["n"]
            assert second_replacement.inputs == ["2", "\r"]
            assert overlay.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_removed_replacement_restores_overlay_even_when_prefocus_differs(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        base = Container()
        editor = _FocusableOverlay(["EDITOR"])
        palette = _FocusableOverlay(["PALETTE"])
        replacement = _FocusableOverlay(["REPLACEMENT"])
        overlay = _FocusableOverlay(["OVERLAY"])

        def _overlay_input(data: str) -> None:
            if data == "b":
                tui.set_focus(replacement)

        def _replacement_input(data: str) -> None:
            if data == "\r":
                base.clear()
                base.add_child(editor)
                tui.set_focus(editor)

        overlay.on_input = _overlay_input
        replacement.on_input = _replacement_input
        base.add_child(editor)
        base.add_child(palette)
        base.add_child(replacement)
        tui.add_child(base)
        tui.set_focus(palette)
        tui.start()
        try:
            tui.show_overlay(overlay)
            terminal.send_input("b")
            await render_and_flush(tui)
            terminal.send_input("\r")
            terminal.send_input("x")
            await render_and_flush(tui)

            assert overlay.inputs == ["b", "x"]
            assert replacement.inputs == ["\r"]
            assert editor.inputs == []
            assert overlay.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_set_focus_none_resumes_visible_overlay_when_blocked(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        replacement = _FocusableOverlay(["REPLACEMENT"])
        overlay = _FocusableOverlay(["OVERLAY"])

        def _replacement_input(data: str) -> None:
            if data == "\r":
                tui.set_focus(None)

        def _overlay_input(data: str) -> None:
            if data == "b":
                tui.set_focus(replacement)

        replacement.on_input = _replacement_input
        overlay.on_input = _overlay_input
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            tui.show_overlay(overlay)
            terminal.send_input("b")
            await render_and_flush(tui)
            terminal.send_input("\r")
            terminal.send_input("x")
            await render_and_flush(tui)

            assert replacement.inputs == ["\r"]
            assert overlay.inputs == ["b", "x"]
            assert overlay.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_unfocus_target_releases_blocked_overlay_while_replacement_remains_focused(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        fallback = _FocusableOverlay(["FALLBACK"])
        target = _FocusableOverlay(["TARGET"])
        replacement = _FocusableOverlay(["REPLACEMENT"])
        overlay = _FocusableOverlay(["OVERLAY"])

        def _replacement_input(data: str) -> None:
            if data == "\r":
                tui.set_focus(fallback)

        replacement.on_input = _replacement_input
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            overlay_handle = tui.show_overlay(overlay)

            def _overlay_input(data: str) -> None:
                if data == "b":
                    tui.set_focus(replacement)
                    overlay_handle.unfocus(OverlayUnfocusOptions(target=target))

            overlay.on_input = _overlay_input

            terminal.send_input("b")
            await render_and_flush(tui)
            assert replacement.focused is True
            terminal.send_input("\r")
            terminal.send_input("x")
            await render_and_flush(tui)

            assert overlay.inputs == ["b"]
            assert replacement.inputs == ["\r"]
            assert fallback.inputs == []
            assert target.inputs == ["x"]
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_handle_input_restores_focus_to_eligible_overlay_after_base_focus_steal(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        replacement = _FocusableOverlay(["REPLACEMENT"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(overlay)
            assert overlay.focused is True
            tui.set_focus(replacement)
            tui.set_focus(editor)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == ["x"]
            assert editor.inputs == []
            assert overlay.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_handle_input_redirects_away_from_invisible_focused_overlay(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        fallback_capturing = _FocusableOverlay(["FALLBACK"])
        non_capturing = _FocusableOverlay(["NC"])
        primary = _FocusableOverlay(["PRIMARY"])
        is_visible = True
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(fallback_capturing)
            tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
            tui.show_overlay(primary, OverlayOptions(visible=lambda _w, _h: is_visible))
            assert primary.focused is True
            is_visible = False
            terminal.send_input("x")
            await render_and_flush(tui)
            assert primary.inputs == []
            assert non_capturing.inputs == []
            assert fallback_capturing.inputs == ["x"]
            assert fallback_capturing.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_temporarily_invisible_focused_overlay_falls_back_without_losing_eligibility(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        is_visible = True
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(overlay, OverlayOptions(visible=lambda _w, _h: is_visible))
            tui.set_focus(editor)
            is_visible = False
            terminal.send_input("x")
            await render_and_flush(tui)
            assert editor.inputs == ["x"]
            assert overlay.inputs == []

            is_visible = True
            terminal.send_input("y")
            await render_and_flush(tui)
            assert editor.inputs == ["x"]
            assert overlay.inputs == ["y"]
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_temporarily_invisible_focused_overlay_with_null_prefocus_restores_when_visible(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _FocusableOverlay(["OVERLAY"])
        is_visible = True
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            tui.show_overlay(overlay, OverlayOptions(visible=lambda _w, _h: is_visible))
            is_visible = False
            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == []

            is_visible = True
            terminal.send_input("y")
            await render_and_flush(tui)
            assert overlay.inputs == ["y"]
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_cyclic_overlay_prefocus_ancestry_does_not_hang_focus_changes(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(overlay)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.focus()
            tui.set_focus(editor)
            terminal.send_input("x")
            # Wrapped in wait_for defensively: a real ancestry cycle would hang
            # this call forever if the visited-set guard regressed.
            await asyncio.wait_for(render_and_flush(tui), timeout=2)
            assert editor.inputs == ["x"]
            assert overlay.inputs == []
        finally:
            tui.stop()


class TestOverlayRenderingOrder:
    @pytest.mark.asyncio
    async def test_default_rendering_order_follows_creation_order(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.show_overlay(_StaticOverlay(["AAAA"]), OverlayOptions(anchor="top-left", width=4))
        tui.show_overlay(_StaticOverlay(["BB"]), OverlayOptions(anchor="top-left", width=4))
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        # B was shown after A, so B (on top) overwrites the first two columns.
        assert viewport[0].startswith("BB")
        tui.stop()

    @pytest.mark.asyncio
    async def test_focus_on_lower_overlay_renders_it_on_top(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        a_handle = tui.show_overlay(_StaticOverlay(["AAAA"]), OverlayOptions(anchor="top-left", width=4))
        tui.show_overlay(_StaticOverlay(["BB"]), OverlayOptions(anchor="top-left", width=4))
        a_handle.focus()
        tui.start()
        await render_and_flush(tui)

        viewport = _viewport(terminal, 80, 24)
        assert viewport[0].startswith("AAAA")
        tui.stop()


class TestHasOverlay:
    @pytest.mark.asyncio
    async def test_has_overlay_reflects_visibility(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        assert tui.has_overlay() is False
        handle = tui.show_overlay(_StaticOverlay(["X"]))
        assert tui.has_overlay() is True
        handle.set_hidden(True)
        assert tui.has_overlay() is False
        handle.set_hidden(False)
        assert tui.has_overlay() is True
        handle.hide()
        assert tui.has_overlay() is False


class TestExtractCursorPosition:
    def test_finds_and_strips_cursor_marker(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        from pi_tui.component import CURSOR_MARKER

        lines = [f"hello{CURSOR_MARKER} world"]
        position = tui.extract_cursor_position(lines, 24)
        assert position == (0, 5)
        assert lines[0] == "hello world"

    def test_returns_none_when_marker_absent(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        assert tui.extract_cursor_position(["no marker here"], 24) is None


class TestInputListeners:
    @pytest.mark.asyncio
    async def test_listener_can_consume_input(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        tui.add_input_listener(lambda _data: TuiInputListenerResult(consume=True))
        tui.start()
        try:
            terminal.send_input("z")
            await asyncio.sleep(0.01)
            assert component.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_listener_can_rewrite_input(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        tui.add_input_listener(lambda data: TuiInputListenerResult(data="rewritten") if data == "z" else None)
        tui.start()
        try:
            terminal.send_input("z")
            await asyncio.sleep(0.01)
            assert component.inputs == ["rewritten"]
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_remove_input_listener_stops_receiving_input(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        seen: list[str] = []

        def _listener(data: str) -> TuiInputListenerResult | None:
            seen.append(data)
            return None

        unregister = tui.add_input_listener(_listener)
        tui.start()
        try:
            terminal.send_input("a")
            await asyncio.sleep(0.01)
            unregister()
            terminal.send_input("b")
            await asyncio.sleep(0.01)
            assert seen == ["a"]
        finally:
            tui.stop()


class TestDebugKey:
    @pytest.mark.asyncio
    async def test_shift_ctrl_d_invokes_on_debug_before_focused_component(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        calls: list[bool] = []
        tui.on_debug = lambda: calls.append(True)
        tui.start()
        try:
            terminal.send_input("\x1b[100;6u")  # shift+ctrl+d (Kitty protocol encoding)
            await asyncio.sleep(0.01)
            assert calls == [True]
            assert component.inputs == []
        finally:
            tui.stop()


class TestRenderScheduling:
    @pytest.mark.asyncio
    async def test_request_render_is_idempotent_before_flush(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _StaticOverlay(["a"])
        tui.add_child(component)
        tui.start()
        await asyncio.sleep(0.03)
        redraws_before = tui.full_redraws
        terminal.writes.clear()

        tui.request_render()
        tui.request_render()
        tui.request_render()
        await asyncio.sleep(0.03)

        # Multiple coalesced requests should not trigger a full redraw storm.
        assert tui.full_redraws == redraws_before
        tui.stop()

    @pytest.mark.asyncio
    async def test_render_now_renders_synchronously(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _StaticOverlay(["hello"])
        tui.add_child(component)
        tui.start()
        tui.render_now()
        assert any("hello" in w for w in terminal.writes)
        tui.stop()


class _MinimalTui(TuiBase):
    """Minimal concrete `TuiBase` subclass that leaves every optional
    lifecycle hook at its base-class default, to exercise those defaults
    directly (subclasses like `TuiMainScreen`/`TuiAltScreen` override most of
    them, so the base implementations are otherwise never reached).
    """

    mode = "regular"

    def do_render(self) -> None:
        return None


class TestBaseHookDefaults:
    def test_default_lifecycle_hooks_are_inert_and_mounted_roots_returns_children(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = _MinimalTui(terminal)
        component = _EmptyContent()
        tui.add_child(component)

        assert tui.reset_render_state() is None
        assert tui.before_terminal_start() is None
        assert tui.after_terminal_start() is None
        assert tui.before_terminal_stop(TuiStopOptions()) is None
        assert tui.after_terminal_stop(TuiStopOptions()) is None
        assert tui.get_mounted_roots() == [component]

    def test_is_viewport_tui_reflects_the_viewport_flag(self) -> None:
        from pi_tui.tui_alt_screen import TuiAltScreen

        terminal = FakeTerminal(80, 24)
        assert is_viewport_tui(TuiMainScreen(terminal)) is False
        assert is_viewport_tui(TuiAltScreen(terminal)) is True


class TestSimpleSettings:
    def test_get_focused_component_returns_the_current_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        assert tui.get_focused_component() is None
        tui.set_focus(component)
        assert tui.get_focused_component() is component

    @pytest.mark.asyncio
    async def test_set_show_hardware_cursor_is_a_noop_when_unchanged_then_applies_a_real_change(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        assert tui.get_show_hardware_cursor() is False

        # Setting to the current value (False) is a no-op guard.
        tui.set_show_hardware_cursor(False)
        assert tui.get_show_hardware_cursor() is False

        tui.set_show_hardware_cursor(True)
        assert tui.get_show_hardware_cursor() is True

        tui.start()
        try:
            terminal.writes.clear()
            # A real change back to False must hide the hardware cursor.
            tui.set_show_hardware_cursor(False)
            assert tui.get_show_hardware_cursor() is False
            assert "\x1b[?25l" in terminal.writes
        finally:
            tui.stop()


class TestTerminalColorSchemeNotifications:
    @pytest.mark.asyncio
    async def test_enables_notifications_sequence_on_start_when_already_enabled(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.set_terminal_color_scheme_notifications(True)
        tui.start()
        try:
            assert "\x1b[?2031h" in terminal.writes
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_disabling_while_running_writes_the_disable_sequence(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            tui.set_terminal_color_scheme_notifications(True)
            terminal.writes.clear()
            tui.set_terminal_color_scheme_notifications(False)
            assert "\x1b[?2031l" in terminal.writes
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_consumes_color_scheme_reports_and_notifies_listeners(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        received: list[str] = []
        tui.on_terminal_color_scheme_change(received.append)
        tui.start()
        try:
            terminal.send_input("\x1b[?997;1n")
            await asyncio.sleep(0.01)
            assert received == ["dark"]
            # A color-scheme report must not be forwarded to the focused component.
            assert component.inputs == []
        finally:
            tui.stop()


class TestCellSizeResponseConsumption:
    @pytest.mark.asyncio
    async def test_consumes_valid_cell_size_response_and_triggers_invalidate_and_render(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        tui.start()
        try:
            terminal.writes.clear()
            terminal.send_input("\x1b[6;20;10t")
            await asyncio.sleep(0.03)
            # Must not be forwarded to the focused component as regular input.
            assert component.inputs == []
            # request_render() following the consumed response should have
            # produced at least one further write to the terminal.
            assert len(terminal.writes) > 0
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_ignores_cell_size_response_with_non_positive_dimensions(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        tui.start()
        try:
            terminal.send_input("\x1b[6;0;0t")
            await asyncio.sleep(0.01)
            # Still consumed (not forwarded as regular input) despite the
            # zero dimensions being discarded.
            assert component.inputs == []
        finally:
            tui.stop()


class TestKeyReleaseGuard:
    @pytest.mark.asyncio
    async def test_key_release_is_dropped_when_component_does_not_want_it(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        tui.add_child(component)
        tui.set_focus(component)
        tui.start()
        try:
            # Kitty protocol key-release encoding for "a" (":3u" release marker).
            terminal.send_input("\x1b[97:3u")
            await asyncio.sleep(0.01)
            assert component.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_key_release_is_forwarded_when_component_wants_it(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        component = _FocusableOverlay(["X"])
        component.wants_key_release = True
        tui.add_child(component)
        tui.set_focus(component)
        tui.start()
        try:
            terminal.send_input("\x1b[97:3u")
            await asyncio.sleep(0.01)
            assert component.inputs == ["\x1b[97:3u"]
        finally:
            tui.stop()


class TestRenderSchedulingEdgeCases:
    @pytest.mark.asyncio
    async def test_synchronous_render_now_preempts_a_pending_immediate_render(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        tui.add_child(_StaticOverlay(["a"]))
        # request_render(force=True) schedules a deferred `_run` callback via
        # `call_soon` but does not render synchronously.
        tui.request_render(force=True)
        # A synchronous render_now() call before that callback fires clears
        # `_render_requested`, so when the deferred callback does run it must
        # find nothing left to do (the early-return guard).
        tui.render_now()
        # Let the previously scheduled callback actually run.
        await asyncio.sleep(0)
        assert tui._render_requested is False

    @pytest.mark.asyncio
    async def test_render_requested_again_during_fire_reschedules_another_render(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        render_calls: list[int] = []

        class _SelfRequestingComponent:
            def __init__(self) -> None:
                self.triggered = False

            def render(self, _width: int) -> list[str]:
                render_calls.append(1)
                if not self.triggered:
                    self.triggered = True
                    # Request another render while this one is still being
                    # produced, mirroring a component that discovers it needs
                    # a follow-up frame mid-render (e.g. an animation tick).
                    tui.request_render()
                return ["X"]

            def invalidate(self) -> None:
                return None

        tui.add_child(_SelfRequestingComponent())
        # Force the min-render-interval throttle to zero so the recursive
        # reschedule's timer fires on the next tick instead of requiring a
        # real sleep for MIN_RENDER_INTERVAL_S to elapse.
        tui.MIN_RENDER_INTERVAL_S = 0.0
        tui.request_render()
        # Drive the event loop through: call_soon -> _schedule_render (sets a
        # zero-delay timer) -> the timer firing -> do_render() (which
        # re-requests) -> the recursive _schedule_render() call -> that timer
        # firing too. No real time is spent; every await is a zero-delay yield.
        for _ in range(10):
            await asyncio.sleep(0)

        assert len(render_calls) >= 2
