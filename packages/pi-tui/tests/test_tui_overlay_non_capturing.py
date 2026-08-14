"""Python port of `packages/tui/test/overlay-non-capturing.test.ts`.

`tests/test_tui.py` already ports a subset of this suite; this module carries the
remaining cases so that every TypeScript `it(...)` has a Python counterpart.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui import OverlayOptions, OverlayUnfocusOptions
from pi_tui.tui_main_screen import TuiMainScreen


class _StaticOverlay:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, _width: int) -> list[str]:
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


def _first_char(terminal: FakeTerminal, width: int, height: int) -> str:
    model = MiniTerminalModel(width, height)
    for write in terminal.writes:
        model.feed(write)
    viewport = model.viewport()
    return viewport[0][0:1] if viewport else ""


class TestFocusManagement:
    @pytest.mark.asyncio
    async def test_deferred_sub_overlay_pattern_restores_focus(self) -> None:
        """Port of "microtask-deferred sub-overlay pattern ... restores focus".

        The TypeScript version defers `showOverlay(controller)` onto a microtask
        via `Promise.resolve().then(...)`; the Python equivalent is a task
        scheduled on the running loop and awaited with `asyncio.sleep(0)`.
        """
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

            async def show_controller() -> None:
                tui.show_overlay(controller)

            task = asyncio.ensure_future(show_controller())
            await task
            await render_and_flush(tui)

            assert controller.focused is True
            assert editor.focused is False

            # Simulate Esc: cleanup + close.
            timer_handle.hide()
            tui.hide_overlay()
            await render_and_flush(tui)

            assert editor.focused is True, "editor should regain focus"
            assert controller.focused is False
            assert timer.focused is False

            terminal.send_input("x")
            await render_and_flush(tui)
            assert editor.inputs == ["x"], "editor should receive input after close"
            assert controller.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_active_base_focus_replacement_receives_close_input_before_overlay_restore(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        replacement = _FocusableOverlay(["REPLACEMENT"])
        overlay = _FocusableOverlay(["OVERLAY"])
        overlay.on_input = lambda data: tui.set_focus(replacement) if data == "b" else None
        replacement.on_input = lambda data: tui.set_focus(editor) if data == "\r" else None
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(overlay)
            assert overlay.focused is True
            terminal.send_input("b")
            await render_and_flush(tui)
            assert replacement.focused is True

            terminal.send_input("\r")
            await render_and_flush(tui)
            assert replacement.inputs == ["\r"]
            assert overlay.inputs == ["b"]
            assert overlay.focused is True

            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == ["b", "x"]
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_active_replacement_still_receives_input_when_it_is_another_overlay_prefocus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        replacement = _FocusableOverlay(["REPLACEMENT"])
        passive = _FocusableOverlay(["PASSIVE"])
        overlay = _FocusableOverlay(["OVERLAY"])
        overlay.on_input = lambda data: tui.set_focus(replacement) if data == "b" else None
        replacement.on_input = lambda data: tui.set_focus(editor) if data == "\r" else None
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.set_focus(replacement)
            tui.show_overlay(passive, OverlayOptions(non_capturing=True))
            tui.set_focus(editor)
            tui.show_overlay(overlay)
            terminal.send_input("b")
            await render_and_flush(tui)
            assert replacement.focused is True

            terminal.send_input("1")
            terminal.send_input("\r")
            await render_and_flush(tui)
            assert replacement.inputs == ["1", "\r"]
            assert overlay.inputs == ["b"]
            assert overlay.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_handle_input_restores_focus_to_explicitly_focused_raw_sub_overlay(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        controller = _FocusableOverlay(["CONTROLLER"])
        sub_overlay = _FocusableOverlay(["SUB"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(controller)
            sub_handle = tui.show_overlay(sub_overlay, OverlayOptions(non_capturing=True))
            sub_handle.focus()
            tui.set_focus(editor)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert sub_overlay.inputs == ["x"]
            assert controller.inputs == []
            assert editor.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_passive_non_capturing_overlay_does_not_regain_input_after_base_focus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        passive = _FocusableOverlay(["PASSIVE"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(passive, OverlayOptions(non_capturing=True))
            terminal.send_input("x")
            await render_and_flush(tui)
            assert editor.inputs == ["x"]
            assert passive.inputs == []
            assert editor.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_explicitly_focused_non_capturing_overlay_regains_input_after_base_focus_steal(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["NC"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
            handle.focus()
            tui.set_focus(editor)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == ["x"]
            assert editor.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_unfocus_prevents_visible_overlay_from_regaining_input(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(overlay)
            handle.unfocus(None)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert editor.inputs == ["x"]
            assert overlay.inputs == []
            assert editor.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_set_focus_none_explicitly_clears_visible_overlay_restore(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            tui.show_overlay(overlay)
            tui.set_focus(None)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == []
            assert overlay.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_handle_input_restores_the_focus_order_top_overlay_after_base_focus_steal(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        lower = _FocusableOverlay(["LOWER"])
        upper = _FocusableOverlay(["UPPER"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            lower_handle = tui.show_overlay(lower)
            tui.show_overlay(upper)
            lower_handle.focus()
            tui.set_focus(editor)
            terminal.send_input("x")
            await render_and_flush(tui)
            assert lower.inputs == ["x"]
            assert upper.inputs == []
            assert editor.inputs == []
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_hide_overlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        capturing = _FocusableOverlay(["CAP"])
        non_capturing = _FocusableOverlay(["NC"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            tui.show_overlay(capturing)
            tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
            assert capturing.focused is True
            tui.hide_overlay()
            await render_and_flush(tui)
            assert capturing.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_multiple_capturing_and_non_capturing_overlays_restore_focus_through_removals(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        c1 = _FocusableOverlay(["C1"])
        n1 = _FocusableOverlay(["N1"])
        c2 = _FocusableOverlay(["C2"])
        n2 = _FocusableOverlay(["N2"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            c1_handle = tui.show_overlay(c1)
            tui.show_overlay(n1, OverlayOptions(non_capturing=True))
            c2_handle = tui.show_overlay(c2)
            tui.show_overlay(n2, OverlayOptions(non_capturing=True))
            assert c2.focused is True
            c2_handle.hide()
            await render_and_flush(tui)
            assert c1.focused is True
            c1_handle.hide()
            await render_and_flush(tui)
            assert editor.focused is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_capturing_overlay_unfocus_on_topmost_capturing_overlay_falls_back_to_prefocus(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        capturing = _FocusableOverlay(["CAP"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            handle = tui.show_overlay(capturing)
            assert capturing.focused is True
            handle.unfocus(None)
            await render_and_flush(tui)
            assert editor.focused is True
            assert capturing.focused is False
        finally:
            tui.stop()


class TestFocusCyclePrevention:
    @pytest.mark.asyncio
    async def test_toggle_focus_between_non_capturing_overlays_then_unfocus_returns_to_editor(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        a = _FocusableOverlay(["A"])
        b = _FocusableOverlay(["B"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            a_handle = tui.show_overlay(a, OverlayOptions(non_capturing=True))
            b_handle = tui.show_overlay(b, OverlayOptions(non_capturing=True))
            a_handle.focus()
            b_handle.focus()
            a_handle.focus()
            a_handle.unfocus(None)
            await render_and_flush(tui)
            assert editor.focused is True
            assert a.focused is False
            assert b.focused is False
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_explicit_null_unfocus_target_clears_focus_without_restoring_overlays(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiMainScreen(terminal)
        overlay = _FocusableOverlay(["OVERLAY"])
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            handle = tui.show_overlay(overlay)
            handle.unfocus(OverlayUnfocusOptions(target=None))
            terminal.send_input("x")
            await render_and_flush(tui)
            assert overlay.inputs == []
            assert handle.is_focused() is False
        finally:
            tui.stop()


class TestRenderingOrder:
    @pytest.mark.asyncio
    async def test_focus_on_already_focused_overlay_bumps_visual_order(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            options = OverlayOptions(row=0, col=0, width=1, non_capturing=True)
            a_handle = tui.show_overlay(_StaticOverlay(["A"]), options)
            tui.show_overlay(_StaticOverlay(["B"]), options)
            a_handle.focus()
            tui.show_overlay(_StaticOverlay(["C"]), options)
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "C"
            a_handle.focus()
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "A"
            assert a_handle.is_focused() is True
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_focusing_middle_overlay_places_it_on_top_preserving_relative_order(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            options = OverlayOptions(row=0, col=0, width=1, non_capturing=True)
            tui.show_overlay(_StaticOverlay(["A"]), options)
            middle = tui.show_overlay(_StaticOverlay(["B"]), options)
            top = tui.show_overlay(_StaticOverlay(["C"]), options)
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "C"
            middle.focus()
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "B"
            middle.hide()
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "C"
            top.hide()
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "A"
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_capturing_overlay_hidden_and_shown_again_renders_on_top_after_unhide(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        tui.add_child(_EmptyContent())
        tui.start()
        try:
            tui.show_overlay(
                _StaticOverlay(["A"]),
                OverlayOptions(row=0, col=0, width=1, non_capturing=True),
            )
            capturing = tui.show_overlay(_StaticOverlay(["B"]), OverlayOptions(row=0, col=0, width=1))
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "B"
            capturing.set_hidden(True)
            tui.show_overlay(
                _StaticOverlay(["C"]),
                OverlayOptions(row=0, col=0, width=1, non_capturing=True),
            )
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "C"
            capturing.set_hidden(False)
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "B"
        finally:
            tui.stop()

    @pytest.mark.asyncio
    async def test_unfocus_does_not_change_visual_order_until_another_overlay_is_focused(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        editor = _FocusableOverlay(["EDITOR"])
        tui.add_child(_EmptyContent())
        tui.set_focus(editor)
        tui.start()
        try:
            options = OverlayOptions(row=0, col=0, width=1, non_capturing=True)
            a = tui.show_overlay(_StaticOverlay(["A"]), options)
            b = tui.show_overlay(_StaticOverlay(["B"]), options)
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "B"
            a.focus()
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "A"
            a.unfocus(None)
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "A"
            b.focus()
            await render_and_flush(tui)
            assert _first_char(terminal, 20, 6) == "B"
        finally:
            tui.stop()
