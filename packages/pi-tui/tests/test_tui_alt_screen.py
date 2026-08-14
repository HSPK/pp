"""Tests for `TuiAltScreen`: viewport scrolling, mouse selection, scrollbar, flashes.

Python port of a representative cross-section of
`packages/tui/test/tui-alt-screen.test.ts` (1067 lines). The remaining cases
(fixed dock, scrollbar-thumb dragging, Kitty/iTerm2 image placement and
grapheme-boundary selection) live in `tests/test_tui_alt_screen_extra.py`.
Uses `MiniAltScreenModel` (a fixed-grid content model, see `fakes.py`) in
place of the TypeScript suite's xterm.js-backed `VirtualTerminal`.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from dataclasses import replace

import pytest
from pi_tui.component import Component
from pi_tui.components.scroll_view import ScrollView, ScrollViewOptions
from pi_tui.components.stack import Stack, StackEntry
from pi_tui.keybindings import TUI_KEYBINDINGS, KeybindingsManager, get_keybindings, set_keybindings
from pi_tui.layout import get_scroll_view_box, get_scrollbar_geometry
from pi_tui.terminal_image import hyperlink
from pi_tui.testing import FakeTerminal, MiniAltScreenModel, wait_until
from pi_tui.tui_alt_screen import (
    SelectionPoint,
    SelectionRange,
    TuiAltScreen,
    TuiAltScreenOptions,
    _SgrMouseEvent,
)


class _Text(Component):
    """Minimal fixed-line text component (no wrapping/padding) for tests."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def set_text(self, text: str) -> None:
        self.text = text

    def render(self, _width: int) -> list[str]:
        if not self.text or self.text.strip() == "":
            return []
        return self.text.split("\n")

    def invalidate(self) -> None:
        return None


class _VStack(Stack):
    layout_type = "vstack"


class _HStack(Stack):
    layout_type = "hstack"


class _FocusableStub(Component):
    def __init__(self) -> None:
        self.focused = False
        self.inputs: list[str] = []

    def render(self, _width: int) -> list[str]:
        return ["editor"]

    def invalidate(self) -> None:
        return None

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)


def _model(terminal: FakeTerminal, width: int, height: int) -> MiniAltScreenModel:
    model = MiniAltScreenModel(width, height)
    for write in terminal.writes:
        model.feed(write)
    return model


def _viewport(terminal: FakeTerminal, width: int, height: int) -> list[str]:
    return [line.rstrip() for line in _model(terminal, width, height).screen()]


async def wait_render() -> None:
    await asyncio.sleep(0.03)


def _all_writes(terminal: FakeTerminal) -> str:
    return "".join(terminal.writes)


class TestViewportScrolling:
    @pytest.mark.asyncio
    async def test_renders_terminal_height_viewport_and_preserves_scroll(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        text = _Text("\n".join(f"line {i + 1}" for i in range(10)))
        tui.add_child(text)
        tui.start()
        await wait_render()

        assert _viewport(terminal, 20, 4) == ["line 7", "line 8", "line 9", "line 10"]
        assert tui.is_following_output is True

        terminal.send_input("\x1b[<64;1;1M")
        await wait_render()
        assert _viewport(terminal, 20, 4) == ["line 6", "line 7", "line 8", "line 9"]
        assert tui.viewport_top == 5
        assert tui.is_following_output is False

        text.set_text("\n".join(f"line {i + 1}" for i in range(12)))
        tui.request_render()
        await wait_render()
        assert _viewport(terminal, 20, 4) == ["line 6", "line 7", "line 8", "line 9"]
        tui.stop()

    @pytest.mark.asyncio
    async def test_invalidates_overlays_with_explicit_layout_root(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiAltScreen(terminal)
        overlay = _Text("overlay")
        invalidated = []
        overlay.invalidate = lambda: invalidated.append(True)  # type: ignore[method-assign]
        tui.set_layout_root(_Text("root"))
        tui.show_overlay(overlay)

        tui.invalidate()

        assert invalidated == [True]

    @pytest.mark.asyncio
    async def test_routes_wheel_input_to_scroll_view_under_pointer(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        left = ScrollView(
            _Text("\n".join(f"a{i + 1}" for i in range(7))), ScrollViewOptions(follow="end", primary=True)
        )
        right = ScrollView(_Text("\n".join(f"b{i + 1}" for i in range(7))), ScrollViewOptions(follow="end"))
        tui.set_layout_root(
            _HStack(
                [
                    StackEntry(component=left, basis=10, shrink=0),
                    StackEntry(component=right, basis=10, shrink=0),
                ]
            )
        )
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<64;15;1M")
        await wait_render()
        assert left.scroll_top == 3
        assert right.scroll_top == 2
        assert _viewport(terminal, 20, 4) == ["a4        b3", "a5        b4", "a6        b5", "a7        b6"]
        tui.stop()

    @pytest.mark.asyncio
    async def test_ignores_horizontal_trackpad_wheel_events(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(8))))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<66;1;1M")
        terminal.send_input("\x1b[<67;1;1M")
        await wait_render()
        assert tui.viewport_top == 4
        assert _viewport(terminal, 20, 4) == ["line 5", "line 6", "line 7", "line 8"]
        tui.stop()


class TestButtonMotionTracking:
    @pytest.mark.asyncio
    async def test_uses_all_motion_tracking_outside_multiplexers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("TMUX", "ZELLIJ", "STY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        terminal = FakeTerminal(80, 24)
        tui = TuiAltScreen(terminal)
        tui.start()
        assert "\x1b[?1003h" in _all_writes(terminal)
        tui.stop()

    @pytest.mark.parametrize(
        "env",
        [
            {"TMUX": "/tmp/tmux/default,1,0"},
            {"TERM": "tmux-256color"},
            {"ZELLIJ": "0"},
            {"STY": "123.session"},
            {"TERM": "screen-256color"},
        ],
    )
    @pytest.mark.asyncio
    async def test_uses_button_motion_tracking_inside_multiplexers(
        self, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("TMUX", "ZELLIJ", "STY", "TERM"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        terminal = FakeTerminal(80, 24)
        tui = TuiAltScreen(terminal)
        tui.start()
        writes = _all_writes(terminal)
        assert "\x1b[?1002h" in writes
        assert "\x1b[?1003h" not in writes
        assert "\x1b[?1006h" in writes
        tui.stop()


class TestRightClickPaste:
    @pytest.mark.asyncio
    async def test_invokes_right_click_paste_handler_only_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        terminal = FakeTerminal(80, 24)
        paste_count = 0

        def _on_paste() -> None:
            nonlocal paste_count
            paste_count += 1

        tui = TuiAltScreen(terminal, options=TuiAltScreenOptions(on_right_click_paste=_on_paste))
        monkeypatch.setattr(sys, "platform", "win32")
        tui.start()
        terminal.send_input("\x1b[<2;1;1M")
        terminal.send_input("\x1b[<2;1;1m")
        assert paste_count == 1

        monkeypatch.setattr(sys, "platform", "linux")
        terminal.send_input("\x1b[<2;1;1M")
        assert paste_count == 1
        tui.stop()


class TestKeyboardViewportNavigation:
    @pytest.mark.asyncio
    async def test_page_up_and_page_down_with_overlap(self) -> None:
        first_page = ["line 1", "line 2", "line 3", "line 4", "line 5", "line 6", "line 7", "line 8"]
        last_page = ["line 5", "line 6", "line 7", "line 8", "line 9", "line 10", "line 11", "line 12"]

        terminal = FakeTerminal(20, 8)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(12))))
        tui.start()
        await wait_render()

        # Kitty keypad pageUp press plus its release event: only the press
        # scrolls, so a single page (with 4 rows of overlap) is traversed.
        terminal.send_input("\x1b[57421u")
        terminal.send_input("\x1b[57421;1:3u")
        await wait_render()
        assert _viewport(terminal, 20, 8) == first_page

        terminal.send_input("\x1b[57422u")
        terminal.send_input("\x1b[57422;1:3u")
        await wait_render()
        assert _viewport(terminal, 20, 8) == last_page

        terminal.send_input("\x1bOH")  # home
        await wait_render()
        assert _viewport(terminal, 20, 8) == first_page

        terminal.send_input("\x1bOF")  # end
        await wait_render()
        assert _viewport(terminal, 20, 8) == last_page
        tui.stop()

    @pytest.mark.asyncio
    async def test_legacy_page_up_and_page_down_and_top_bottom_positions(self) -> None:
        terminal = FakeTerminal(20, 8)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(12))))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[5~")  # legacy pageUp
        await wait_render()
        assert tui.viewport_top == 0

        terminal.send_input("\x1b[6~")  # legacy pageDown
        await wait_render()
        assert tui.viewport_top == 4

        terminal.send_input("\x1bOH")  # home
        await wait_render()
        assert tui.viewport_top == 0

        terminal.send_input("\x1bOF")  # end
        await wait_render()
        assert tui.viewport_top == 4
        tui.stop()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_single_line_scroll_with_custom_bindings(self) -> None:
        """`tui.altScreen.lineUp`/`lineDown`, unbound by default upstream.

        They exist so a user can bind single-line scrolling; no default key
        occupies them, so the only way to exercise them is a custom binding.
        """
        original = get_keybindings()
        terminal = FakeTerminal(20, 10)
        tui = TuiAltScreen(terminal)
        set_keybindings(
            KeybindingsManager(
                TUI_KEYBINDINGS,
                {"tui.altScreen.lineUp": "ctrl+u", "tui.altScreen.lineDown": "ctrl+d"},
            )
        )
        try:
            tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(30))))
            tui.start()
            await wait_render()
            assert tui.viewport_top == 20

            terminal.send_input("\x15")  # ctrl+u
            await wait_render()
            assert tui.viewport_top == 19

            terminal.send_input("\x04")  # ctrl+d
            await wait_render()
            assert tui.viewport_top == 20
        finally:
            tui.stop()
            set_keybindings(original)

    async def test_half_page_scroll_with_custom_bindings(self) -> None:
        original = get_keybindings()
        terminal = FakeTerminal(20, 10)
        tui = TuiAltScreen(terminal)
        set_keybindings(
            KeybindingsManager(
                TUI_KEYBINDINGS,
                {"tui.altScreen.halfPageUp": "ctrl+u", "tui.altScreen.halfPageDown": "ctrl+d"},
            )
        )
        try:
            tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(30))))
            tui.start()
            await wait_render()
            assert tui.viewport_top == 20

            terminal.send_input("\x15")  # ctrl+u
            await wait_render()
            assert tui.viewport_top == 15

            terminal.send_input("\x04")  # ctrl+d
            await wait_render()
            assert tui.viewport_top == 20
        finally:
            tui.stop()
            set_keybindings(original)

    @pytest.mark.asyncio
    async def test_routes_ctrl_modified_navigation_to_focused_component(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        transcript = ScrollView(
            _Text("\n".join(f"line {i + 1}" for i in range(12))), ScrollViewOptions(follow="end", primary=True)
        )
        editor = _FocusableStub()
        tui.set_layout_root(
            _VStack(
                [
                    StackEntry(component=transcript, basis=0, grow=1, min_size=1),
                    StackEntry(component=editor, basis=1, shrink=0),
                ]
            )
        )
        tui.set_focus(editor)
        tui.start()
        await wait_render()

        terminal.send_input("\x1bOH")  # unmodified home -> viewport nav
        await wait_render()
        assert transcript.scroll_top == 0
        assert editor.inputs == []

        modified_inputs = ["\x1b[1;5H", "\x1b[1;5F", "\x1b[5;5~", "\x1b[6;5~", "\x1b[57423;5u"]
        for data in modified_inputs:
            terminal.send_input(data)
        # The matching Kitty release event must be swallowed, not forwarded.
        terminal.send_input("\x1b[57423;5:3u")
        await wait_render()
        assert transcript.scroll_top == 0
        assert editor.inputs == modified_inputs

        terminal.send_input("\x1b[6~")  # unmodified pageDown -> viewport nav
        await wait_render()
        assert transcript.scroll_top == 1
        assert editor.inputs == modified_inputs
        tui.stop()

    @pytest.mark.asyncio
    async def test_jumps_between_osc133_prompt_markers(self) -> None:
        terminal = FakeTerminal(20, 3)
        tui = TuiAltScreen(terminal)
        lines = []
        for message in (1, 2, 3, 4):
            lines.append(f"\x1b]133;A\x07message {message}")
            lines.append("detail")
        tui.add_child(_Text("\n".join(lines)))
        tui.start()
        await wait_render()
        assert tui.viewport_top == 5

        # Kitty ctrl+shift+up (previousPrompt) plus its release event.
        terminal.send_input("\x1b[57419;6u")
        terminal.send_input("\x1b[57419;6:3u")
        await wait_render()
        assert tui.viewport_top == 4
        assert _viewport(terminal, 20, 3)[0] == "message 3"

        terminal.send_input("\x1b[1;6A")  # legacy ctrl+shift+up
        await wait_render()
        assert tui.viewport_top == 2
        assert _viewport(terminal, 20, 3)[0] == "message 2"

        terminal.send_input("\x1b[57420;6u")  # Kitty ctrl+shift+down
        terminal.send_input("\x1b[57420;6:3u")
        await wait_render()
        assert tui.viewport_top == 4
        assert _viewport(terminal, 20, 3)[0] == "message 3"

        terminal.send_input("\x1b[1;6B")  # legacy ctrl+shift+down
        await wait_render()
        assert tui.viewport_top == 5
        assert _viewport(terminal, 20, 3)[1] == "message 4"
        assert tui.is_following_output is True
        tui.stop()


class TestHyperlinksAndSelection:
    @pytest.mark.asyncio
    async def test_opens_hyperlink_on_click_but_not_drag(self) -> None:
        terminal = FakeTerminal(20, 3)
        opened: list[str] = []
        tui = TuiAltScreen(terminal, options=TuiAltScreenOptions(open_url=lambda url: opened.append(url)))
        url = "https://example.com/path?q=1"
        bel_url = "https://example.com/bel"
        emoji_url = "https://example.com/emoji"
        tui.add_child(
            _Text(f"{hyperlink('link', url)}\n\x1b]8;;{bel_url}\x07link\x1b]8;;\x07\n{hyperlink('🙂', emoji_url)}")
        )
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<0;2;1M")
        terminal.send_input("\x1b[<0;2;1m")
        await wait_render()
        assert opened == [url]

        # BEL-terminated OSC 8 must be recognised too.
        terminal.send_input("\x1b[<0;2;2M")
        terminal.send_input("\x1b[<0;2;2m")
        await wait_render()
        assert opened == [url, bel_url]

        # Clicking the second cell of a wide emoji grapheme still hits the link.
        terminal.send_input("\x1b[<0;2;3M")
        terminal.send_input("\x1b[<0;2;3m")
        await wait_render()
        assert opened == [url, bel_url, emoji_url]

        # A drag (motion bit set) before release should not open the link again.
        terminal.send_input("\x1b[<0;2;1M")
        terminal.send_input("\x1b[<32;4;1M")
        terminal.send_input("\x1b[<0;4;1m")
        await wait_render()
        assert opened == [url, bel_url, emoji_url]
        tui.stop()

    @pytest.mark.asyncio
    async def test_selects_text_with_mouse_and_copies_with_osc52(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\x1b[1mal\x1b[0mpha\nbeta\ngamma\ndelta"))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<32;4;2M")
        terminal.send_input("\x1b[<0;4;2m")
        await wait_render()

        expected = base64.b64encode(b"alpha\nbeta").decode("ascii")
        writes = _all_writes(terminal)
        assert f"\x1b]52;c;{expected}\x07" in writes
        assert "\x1b[7m" in writes
        assert "al\x1b[0m\x1b[7mpha" in writes
        assert any("Copied!" in line for line in _viewport(terminal, 20, 4))
        tui.stop()

    @pytest.mark.asyncio
    async def test_double_click_selects_word_triple_click_selects_line(self) -> None:
        terminal = FakeTerminal(20, 2)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("zero alpha beta\ngamma delta"))
        tui.start()
        await wait_render()

        # Double-click selects the whole word even though the two clicks land
        # on different characters within it.
        terminal.send_input("\x1b[<0;6;1M")
        terminal.send_input("\x1b[<0;6;1m")
        terminal.send_input("\x1b[<0;10;1M")
        terminal.send_input("\x1b[<0;10;1m")
        await wait_render()
        alpha = base64.b64encode(b"alpha").decode("ascii")
        assert f"\x1b]52;c;{alpha}\x07" in _all_writes(terminal)

        # A double-click drag includes each word touched, not partial words.
        terminal.send_input("\x1b[<0;12;1M")
        terminal.send_input("\x1b[<0;12;1m")
        terminal.send_input("\x1b[<0;14;1M")
        terminal.send_input("\x1b[<32;3;2M")
        terminal.send_input("\x1b[<0;3;2m")
        await wait_render()
        words = base64.b64encode(b"beta\ngamma").decode("ascii")
        assert f"\x1b]52;c;{words}\x07" in _all_writes(terminal)

        # Triple click (on the second line) selects the whole line.
        terminal.send_input("\x1b[<0;7;2M")
        terminal.send_input("\x1b[<0;7;2m")
        terminal.send_input("\x1b[<0;9;2M")
        terminal.send_input("\x1b[<0;9;2m")
        terminal.send_input("\x1b[<0;11;2M")
        terminal.send_input("\x1b[<0;11;2m")
        await wait_render()
        line = base64.b64encode(b"gamma delta").decode("ascii")
        assert f"\x1b]52;c;{line}\x07" in _all_writes(terminal)
        tui.stop()

    @pytest.mark.asyncio
    async def test_focus_loss_cancels_active_selection(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha\nbeta\ngamma\ndelta"))
        tui.start()
        await wait_render()

        def clipboard_write_count() -> int:
            return _all_writes(terminal).count("\x1b]52;c;")

        # A completed click leaves a zero-width anchor; later orphaned
        # drag/release events must not extend it into a real selection.
        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<0;1;1m")
        terminal.send_input("\x1b[<32;4;2M")
        terminal.send_input("\x1b[<0;4;2m")
        await wait_render()
        assert clipboard_write_count() == 0

        # A press whose matching release never arrives is cancelled by a focus loss.
        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[O")  # FOCUS_OUT
        terminal.send_input("\x1b[I")  # FOCUS_IN
        terminal.send_input("\x1b[<32;4;2M")
        terminal.send_input("\x1b[<0;4;2m")
        await wait_render()
        assert clipboard_write_count() == 0
        assert "\x1b[?1004h" in _all_writes(terminal)
        tui.stop()
        assert "\x1b[?1004l" in _all_writes(terminal)

    @pytest.mark.asyncio
    async def test_does_not_repaint_idle_or_zero_width_selections_on_focus_loss(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha\nbeta\ngamma\ndelta"))
        tui.start()
        await wait_render()

        idle_write_count = len(terminal.writes)
        terminal.send_input("\x1b[O")  # FOCUS_OUT
        terminal.send_input("\x1b[I")  # FOCUS_IN
        await wait_render()
        assert len(terminal.writes) == idle_write_count

        # Losing focus after a press without a drag cancels the press without repainting.
        terminal.send_input("\x1b[<0;1;3M")
        await wait_render()
        pressed_write_count = len(terminal.writes)
        terminal.send_input("\x1b[O")
        terminal.send_input("\x1b[I")
        await wait_render()
        assert len(terminal.writes) == pressed_write_count
        tui.stop()

    @pytest.mark.asyncio
    async def test_clears_an_active_visible_selection_on_focus_loss(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha\nbeta\ngamma\ndelta"))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<32;4;2M")
        await wait_render()
        focus_loss_write_index = len(terminal.writes)
        terminal.send_input("\x1b[O")
        terminal.send_input("\x1b[I")
        await wait_render()
        focus_loss_writes = "".join(terminal.writes[focus_loss_write_index:])
        assert "alpha" in focus_loss_writes
        assert "beta" in focus_loss_writes
        assert "\x1b[7m" not in focus_loss_writes

        terminal.send_input("\x1b[<32;4;2M")
        terminal.send_input("\x1b[<0;4;2m")
        await wait_render()
        assert "\x1b]52;c;" not in _all_writes(terminal)
        tui.stop()

    @pytest.mark.asyncio
    async def test_retains_a_completed_visible_selection_across_focus_changes(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha\nbeta\ngamma\ndelta"))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<32;4;2M")
        terminal.send_input("\x1b[<0;4;2m")
        await wait_render()
        completed_write_count = len(terminal.writes)
        terminal.send_input("\x1b[O")
        terminal.send_input("\x1b[I")
        await wait_render()
        assert len(terminal.writes) == completed_write_count

        redraw_write_index = len(terminal.writes)
        tui.render_now(True)
        redraw_writes = "".join(terminal.writes[redraw_write_index:])
        assert "alpha" in redraw_writes
        assert "beta" in redraw_writes
        assert "\x1b[7m" in redraw_writes
        tui.stop()


class TestFlashMessages:
    @pytest.mark.asyncio
    async def test_stacks_and_expires_flash_messages(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("one\ntwo\nthree\nfour"))
        tui.start()
        await wait_render()

        tui.flash("First", 80)
        tui.flash("Second", 500)
        await wait_render()
        screen = _model(terminal, 20, 4).screen()
        assert screen[0].endswith(" First ")
        assert screen[1].endswith(" Second ")

        # The 80 ms flash expires on a real timer (as it does in TypeScript),
        # so wait for the outcome rather than for a fixed duration: under
        # `-n auto` a fixed sleep is a wall-clock race.
        await wait_until(
            lambda: not any("First" in line for line in _model(terminal, 20, 4).screen()),
            message="the 80ms flash never expired",
        )
        await wait_render()
        screen = _model(terminal, 20, 4).screen()
        assert screen[0].endswith(" Second ")
        assert not any("First" in line for line in screen)
        tui.stop()

    @pytest.mark.asyncio
    async def test_auto_scrolls_and_extends_a_drag_selection_held_at_the_viewport_edge(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(10))))
        tui.start()
        await wait_render()
        assert tui.viewport_top == 6

        terminal.send_input("\x1b[<0;1;3M")
        terminal.send_input("\x1b[<32;1;1M")
        # Auto-scroll repeats on a real timer; wait for it to move the viewport
        # instead of assuming a fixed number of repeats fit in a fixed sleep.
        await wait_until(
            lambda: tui.viewport_top < 6,
            message="drag at the viewport edge never auto-scrolled",
        )
        await wait_render()

        selection_top = tui.viewport_top
        assert selection_top < 6, f"expected auto-scroll above row 6, got {selection_top}"
        terminal.send_input("\x1b[<0;1;1m")
        await wait_render()

        selected_lines = [f"line {selection_top + index + 1}" for index in range(8 - selection_top)]
        selected_lines.append("l")
        expected = base64.b64encode("\n".join(selected_lines).encode()).decode("ascii")
        assert f"\x1b]52;c;{expected}\x07" in _all_writes(terminal)
        tui.stop()


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_restores_keyboard_state_and_prints_full_document_on_stop(self) -> None:
        terminal = FakeTerminal(20, 3)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("first\nsecond\nthird\nfourth\nfifth\nsixth"))
        tui.start()
        await wait_render()
        tui.stop()

        writes = terminal.writes
        alt_enter_index = next(i for i, w in enumerate(writes) if "\x1b[?1049h" in w)
        mouse_disable_index = next(i for i, w in enumerate(writes) if "\x1b[?1006l" in w)
        main_restore_index = next(i for i, w in enumerate(writes) if "\x1b[?1049l" in w)

        # The TypeScript test also pins `altScreenEnterIndex < startIndex` and
        # `mouseDisableIndex < stopIndex` against a terminal double that records
        # `start`/`stop` calls interleaved with writes. `FakeTerminal` records
        # writes only, so the equivalent claim is expressed purely as the write
        # ordering below.
        assert alt_enter_index < mouse_disable_index < main_restore_index

        restore_write = writes[main_restore_index]
        for word in ("first", "second", "third", "fourth", "fifth", "sixth"):
            assert word in restore_write
        assert restore_write.index("first") < restore_write.index("sixth")


class TestScrollFallbacksBeforeFirstRender:
    @pytest.mark.asyncio
    async def test_set_layout_root_to_the_same_component_is_a_noop(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiAltScreen(terminal)
        root = _Text("root")
        tui.set_layout_root(root)
        # A second call with the identical component must not reset the
        # cached layout (it would force an unnecessary full re-layout).
        tui._current_layout = "sentinel"  # type: ignore[assignment]
        tui.set_layout_root(root)
        assert tui._current_layout == "sentinel"

    def test_viewport_properties_fall_back_to_the_implicit_scroll_view_before_any_render(self) -> None:
        terminal = FakeTerminal(80, 24)
        tui = TuiAltScreen(terminal)
        # No start()/render_now() has happened yet, so `_current_layout` is
        # still None and these properties must fall back to the implicit
        # scroll view rather than raising.
        assert tui.viewport_top == 0
        assert tui.is_following_output is True


class TestScrollToPromptEdgeCases:
    @pytest.mark.asyncio
    async def test_previous_prompt_before_first_render_is_a_noop(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_input_listener(lambda data: None)
        # `_current_layout` is None because no render has happened.
        tui._scroll_to_prompt(-1)
        assert tui.viewport_top == 0

    @pytest.mark.asyncio
    async def test_prompt_navigation_on_empty_document_is_a_noop(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.start()
        await wait_render()
        # An empty document has no scroll content lines at all.
        terminal.send_input("\x1b[1;6A")  # ctrl+shift+up: previousPrompt
        await wait_render()
        assert tui.viewport_top == 0
        tui.stop()

    @pytest.mark.asyncio
    async def test_prompt_navigation_with_no_markers_scans_to_the_end_without_moving(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(10))))
        tui.start()
        await wait_render()
        top_before = tui.viewport_top
        # No OSC133 prompt markers anywhere: the scan must exhaust every row
        # without finding a match and leave the viewport unchanged.
        terminal.send_input("\x1b[1;6A")  # previousPrompt
        await wait_render()
        assert tui.viewport_top == top_before
        tui.stop()


class TestFocusOutWithoutActiveSelection:
    @pytest.mark.asyncio
    async def test_focus_out_without_a_pending_selection_press_does_not_clear_selection_state(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha\nbeta\ngamma\ndelta"))
        tui.start()
        await wait_render()
        # No mouse press happened, so `_selection_press_active` is False:
        # FOCUS_OUT must still be consumed but skip clearing the selection
        # anchor/focus (there is nothing to clear).
        terminal.send_input("\x1b[O")
        await wait_render()
        assert tui._selection_anchor is None
        assert tui._selection_focus is None
        tui.stop()


class TestLegacyX10MouseSequences:
    @pytest.mark.asyncio
    async def test_legacy_x10_wheel_event_scrolls_the_viewport(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(8))))
        tui.start()
        await wait_render()
        assert tui.viewport_top == 4

        # Legacy X10 mouse-report wheel-up: "\x1b[M" + button + x + y, all
        # byte-offset by 32 (button 96 = 32 (offset) + 64 (wheel bit)).
        terminal.send_input("\x1b[M" + chr(96) + chr(33) + chr(33))
        await wait_render()
        assert tui.viewport_top == 3
        tui.stop()

    @pytest.mark.asyncio
    async def test_legacy_x10_non_wheel_click_is_consumed_without_side_effects(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(8))))
        tui.start()
        await wait_render()
        top_before = tui.viewport_top

        # Legacy X10 primary-button press (button 0, no wheel bit set): not a
        # wheel event and not parseable as an SGR mouse report, but still a
        # recognized (legacy) mouse sequence that must be swallowed rather
        # than leaking through to keybinding/text handling.
        terminal.send_input("\x1b[M" + chr(32) + chr(33) + chr(33))
        await wait_render()
        assert tui.viewport_top == top_before
        tui.stop()


class TestScrollbarDrag:
    @pytest.mark.asyncio
    async def test_pressing_and_dragging_the_scrollbar_thumb_scrolls_the_view(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(
            _Text("\n".join(f"line {i + 1}" for i in range(50))),
            ScrollViewOptions(primary=True, scrollbar="always"),
        )
        tui.set_layout_root(scroll_view)
        tui.start()
        await wait_render()

        box = get_scroll_view_box(tui._current_layout, scroll_view)
        geometry = get_scrollbar_geometry(box)
        assert geometry is not None
        column = geometry.column + 1  # SGR mouse reports are 1-indexed.
        thumb_row = geometry.thumb_top + 1

        terminal.send_input(f"\x1b[<0;{column};{thumb_row}M")
        await wait_render()
        assert tui._scrollbar_drag is not None

        # Drag the thumb toward the bottom of the track without releasing.
        bottom_row = geometry.track_top + geometry.track_height
        terminal.send_input(f"\x1b[<32;{column};{bottom_row}M")
        await wait_render()
        assert scroll_view.scroll_top > 0

        terminal.send_input(f"\x1b[<0;{column};{bottom_row}m")
        await wait_render()
        assert tui._scrollbar_drag is None
        tui.stop()

    @pytest.mark.asyncio
    async def test_scrollbar_hover_is_set_and_cleared_as_the_pointer_moves(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(
            _Text("\n".join(f"line {i + 1}" for i in range(50))),
            ScrollViewOptions(primary=True, scrollbar="always"),
        )
        tui.set_layout_root(scroll_view)
        tui.start()
        await wait_render()

        box = get_scroll_view_box(tui._current_layout, scroll_view)
        geometry = get_scrollbar_geometry(box)
        assert geometry is not None
        column = geometry.column + 1
        thumb_row = geometry.thumb_top + 1

        # A move (not a press) over the thumb sets hover; a subsequent move
        # away from it clears hover again.
        terminal.send_input("\x1b[<35;1;1M")
        await wait_render()
        assert tui._scrollbar_hover is None

        terminal.send_input(f"\x1b[<35;{column};{thumb_row}M")
        await wait_render()
        assert tui._scrollbar_hover is scroll_view

        terminal.send_input("\x1b[<35;1;1M")
        await wait_render()
        assert tui._scrollbar_hover is None
        tui.stop()


class TestNestedScrollWheelChaining:
    @pytest.mark.asyncio
    async def test_chains_unused_wheel_delta_to_an_outer_scroll_view(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal, options=TuiAltScreenOptions(wheel_scroll_lines=3))
        inner = ScrollView(_Text("\n".join(f"i{i + 1}" for i in range(6))))
        outer = ScrollView(
            _VStack(
                [
                    StackEntry(component=inner, basis=2),
                    StackEntry(component=_Text("tail1\ntail2\ntail3\ntail4\ntail5")),
                ]
            ),
            ScrollViewOptions(primary=True),
        )
        tui.set_layout_root(outer)
        tui.start()
        await wait_render()

        # First wheel-down: `inner` can absorb all 3 lines (6 lines of content
        # in a 2-row box scrolls 0 -> 3), so nothing chains to `outer`.
        terminal.send_input("\x1b[<65;1;1M")
        await wait_render()
        assert inner.scroll_top == 3
        assert outer.scroll_top == 0

        # Second wheel-down: `inner` has only 1 line of range left, so the
        # remaining 2 lines chain out to `outer`.
        terminal.send_input("\x1b[<65;1;1M")
        await wait_render()
        assert inner.scroll_top == 4
        assert outer.scroll_top == 2
        tui.stop()

    @pytest.mark.asyncio
    async def test_wheel_event_with_no_scroll_view_under_the_pointer_still_scrolls_the_primary_view(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(8))))
        tui.start()
        await wait_render()
        assert tui.viewport_top == 4

        # Coordinates entirely outside the terminal: no scroll view's clip
        # rect contains this point, so nothing is found by hit-testing. The
        # event still must not be silently dropped: it falls back to
        # scrolling the primary (implicit) scroll view.
        terminal.send_input("\x1b[<64;999;999M")
        await wait_render()
        assert tui.viewport_top == 3
        tui.stop()

    @pytest.mark.asyncio
    async def test_wheel_chains_through_an_inner_view_that_cannot_absorb_it(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        inner = ScrollView(_Text("\n".join(f"i{i + 1}" for i in range(3))), ScrollViewOptions(follow="end"))
        outer = ScrollView(inner, ScrollViewOptions(follow="end", primary=True))
        tui.set_layout_root(outer)
        tui.start()
        await wait_render()

        # `inner`'s 3 lines fit entirely within the 6-row viewport, so it has
        # no scroll range at all: any wheel event over it must chain through
        # to `outer` (the default "chain" overscroll policy) instead of
        # being absorbed silently.
        terminal.send_input("\x1b[<64;1;1M")
        await wait_render()
        assert inner.scroll_top == 0
        tui.stop()

    @pytest.mark.asyncio
    async def test_wheel_over_a_boundary_scroll_view_bubbles_to_the_primary_view(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        left = ScrollView(
            _Text("\n".join(f"a{i + 1}" for i in range(7))), ScrollViewOptions(follow="end", primary=True)
        )
        right = ScrollView(_Text("\n".join(f"b{i + 1}" for i in range(7))), ScrollViewOptions())
        tui.set_layout_root(
            _HStack(
                [
                    StackEntry(component=left, basis=10, shrink=0),
                    StackEntry(component=right, basis=10, shrink=0),
                ]
            )
        )
        tui.start()
        await wait_render()
        assert (left.scroll_top, right.scroll_top) == (3, 0)

        # `right` starts at scroll_top 0 (not following), so scrolling up
        # over it is fully rejected; since `right` isn't the primary view,
        # the leftover delta must bubble to `left` even though the pointer
        # never touched it.
        terminal.send_input("\x1b[<64;15;1M")
        await wait_render()
        assert left.scroll_top == 2
        assert right.scroll_top == 0
        tui.stop()


class TestKeyReleaseGuardsForViewportBindings:
    @pytest.mark.asyncio
    async def test_key_release_events_are_swallowed_without_scrolling(self) -> None:
        original = get_keybindings()
        terminal = FakeTerminal(20, 8)
        tui = TuiAltScreen(terminal)
        set_keybindings(
            KeybindingsManager(
                TUI_KEYBINDINGS,
                {"tui.altScreen.halfPageUp": "insert", "tui.altScreen.halfPageDown": "delete"},
            )
        )
        try:
            tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(30))))
            tui.start()
            await wait_render()
            top_before = tui.viewport_top

            release_sequences = [
                "\x1b[5:3~",  # pageUp release
                "\x1b[6:3~",  # pageDown release
                "\x1b[2:3~",  # halfPageUp (custom "insert") release
                "\x1b[3:3~",  # halfPageDown (custom "delete") release
                "\x1b[1;6:3A",  # previousPrompt release
                "\x1b[1;6:3B",  # nextPrompt release
                "\x1b[1;1:3H",  # top release
                "\x1b[1;1:3F",  # bottom release
            ]
            for seq in release_sequences:
                terminal.send_input(seq)
            await wait_render()
            assert tui.viewport_top == top_before
        finally:
            tui.stop()
            set_keybindings(original)


class TestLegacyX10HorizontalWheel:
    @pytest.mark.asyncio
    async def test_legacy_x10_horizontal_wheel_is_ignored(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(8))))
        tui.start()
        await wait_render()
        top_before = tui.viewport_top
        # button 98 = 32 (offset) + 64 (wheel bit) + 2 (horizontal direction,
        # unsupported: only vertical wheel directions 0/1 are handled).
        terminal.send_input("\x1b[M" + chr(98) + chr(33) + chr(33))
        await wait_render()
        assert tui.viewport_top == top_before
        tui.stop()


class TestScrollbarDragSurvivesLayoutSwap:
    @pytest.mark.asyncio
    async def test_scrollbar_drag_continues_gracefully_when_geometry_disappears_mid_drag(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(
            _Text("\n".join(f"line {i + 1}" for i in range(50))),
            ScrollViewOptions(primary=True, scrollbar="always"),
        )
        tui.set_layout_root(scroll_view)
        tui.start()
        await wait_render()

        box = get_scroll_view_box(tui._current_layout, scroll_view)
        geometry = get_scrollbar_geometry(box)
        assert geometry is not None
        column = geometry.column + 1
        thumb_row = geometry.thumb_top + 1
        terminal.send_input(f"\x1b[<0;{column};{thumb_row}M")
        await wait_render()
        assert tui._scrollbar_drag is not None

        # Swap the layout root away mid-drag: the dragged scroll view no
        # longer has a box in the freshly rendered layout.
        tui.set_layout_root(_Text("replacement"))
        tui.render_now()

        scroll_before = scroll_view.scroll_top
        terminal.send_input(f"\x1b[<32;{column};{thumb_row + 1}M")
        await wait_render()
        # Still "dragging" (no crash, no premature release) but with nothing
        # to update since the geometry is gone.
        assert tui._scrollbar_drag is not None
        assert scroll_view.scroll_top == scroll_before

        terminal.send_input(f"\x1b[<0;{column};{thumb_row + 1}m")
        await wait_render()
        assert tui._scrollbar_drag is None
        tui.stop()


class TestScrollViewSelectionAutoScroll:
    @pytest.mark.asyncio
    async def test_dragging_to_the_boundary_row_auto_scrolls_until_the_content_edge(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(_Text("\n".join(f"line {i + 1}" for i in range(20))), ScrollViewOptions(primary=True))
        tui.set_layout_root(scroll_view)
        tui.start()
        await wait_render()

        # Press inside the viewport, then drag to the bottom-most visible
        # row (the auto-scroll trigger zone).
        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<32;1;4M")
        await wait_render()
        assert tui._selection_auto_scroll_direction == 1
        assert tui._selection_auto_scroll_timer is not None

        # A second drag event at the same boundary row must not create a
        # second timer.
        existing_timer = tui._selection_auto_scroll_timer
        terminal.send_input("\x1b[<32;1;4M")
        await wait_render()
        assert tui._selection_auto_scroll_timer is existing_timer

        # Drive the (real, but never-slept-on) interval tick manually until
        # it reaches the bottom of the content and self-cancels.
        for _ in range(30):
            if tui._selection_auto_scroll_timer is None:
                break
            tui._auto_scroll_selection()
        assert tui._selection_auto_scroll_timer is None
        assert scroll_view.scroll_top == 20 - 4

        # A fresh press-and-drag back inside the viewport bounds must not
        # start auto-scroll at all.
        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<32;1;2M")
        await wait_render()
        assert tui._selection_auto_scroll_direction == 0
        assert tui._selection_auto_scroll_timer is None
        tui.stop()


class TestSelectionUnderOverlay:
    @pytest.mark.asyncio
    async def test_pressing_while_an_overlay_is_shown_skips_scroll_view_detection(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(_Text("\n".join(f"line {i + 1}" for i in range(20))), ScrollViewOptions(primary=True))
        tui.set_layout_root(scroll_view)
        overlay = _Text("overlay")
        tui.start()
        await wait_render()
        tui.show_overlay(overlay)
        await wait_render()

        terminal.send_input("\x1b[<0;1;1M")
        await wait_render()
        assert tui._selection_anchor is not None
        assert tui._selection_anchor.scroll_view is None

        # Drag toward the boundary row: because the anchor has no associated
        # scroll view (an overlay was showing at press time), auto-scroll
        # must be a no-op rather than scrolling the hidden base view.
        terminal.send_input("\x1b[<32;1;4M")
        await wait_render()
        assert tui._selection_auto_scroll_direction == 0
        assert tui._selection_auto_scroll_timer is None
        tui.stop()

    @pytest.mark.asyncio
    async def test_release_with_no_selection_anchor_is_a_noop(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha"))
        tui.start()
        await wait_render()
        # Simulate a release arriving while a press was "active" but the
        # anchor was already cleared by some other event.
        tui._selection_press_active = True
        tui._selection_anchor = None
        writes_before = len(terminal.writes)
        terminal.send_input("\x1b[<0;1;1m")
        await wait_render()
        assert len(terminal.writes) == writes_before
        tui.stop()


class TestSelectionHelperGuards:
    def test_scroll_selection_point_returns_none_before_any_render(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(_Text("x"), ScrollViewOptions(primary=True))
        assert tui._get_scroll_selection_point(scroll_view, 0, 0) is None

    @pytest.mark.asyncio
    async def test_scroll_selection_point_returns_none_for_a_scroll_view_outside_the_current_layout(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha"))
        tui.start()
        await wait_render()
        orphan = ScrollView(_Text("orphan"), ScrollViewOptions())
        assert tui._get_scroll_selection_point(orphan, 0, 0) is None
        tui.stop()

    @pytest.mark.asyncio
    async def test_selection_point_falls_back_to_the_default_point_when_the_scroll_view_lookup_fails(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha"))
        tui.start()
        await wait_render()
        orphan = ScrollView(_Text("orphan"), ScrollViewOptions())
        event = _SgrMouseEvent(button=0, x=2, y=1, release=False)
        point = tui._get_selection_point(event, orphan)
        assert point.scroll_view is None
        assert (point.row, point.col) == (1, 2)
        tui.stop()

    def test_get_word_selection_returns_none_past_the_end_of_the_line(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui._previous_screen = ["short line"]
        assert tui._get_word_selection(SelectionPoint(row=0, col=50)) is None

    def test_update_selection_focus_leaves_anchor_and_focus_untouched_past_the_end_of_the_line(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui._previous_screen = ["short line"]
        anchor = SelectionPoint(row=0, col=0)
        tui._selection_granularity = "word"
        tui._selection_initial_range = SelectionRange(start=anchor, end=replace(anchor, col=5, boundary=True))
        tui._selection_anchor = anchor
        tui._selection_focus = anchor
        # No word segment covers column 50: the update must be a no-op
        # rather than clearing the current selection.
        tui._update_selection_focus(SelectionPoint(row=0, col=50))
        assert tui._selection_focus is anchor

    def test_get_selection_bounds_is_none_across_different_scroll_views(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        sv_a = ScrollView(_Text("a"), ScrollViewOptions())
        sv_b = ScrollView(_Text("b"), ScrollViewOptions())
        tui._selection_anchor = SelectionPoint(row=0, col=0, scroll_view=sv_a)
        tui._selection_focus = SelectionPoint(row=0, col=3, scroll_view=sv_b)
        assert tui._get_selection_bounds() is None

    def test_copy_selection_to_clipboard_is_a_noop_before_any_render(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(_Text("alpha"), ScrollViewOptions())
        tui._selection_anchor = SelectionPoint(row=0, col=0, scroll_view=scroll_view)
        tui._selection_focus = SelectionPoint(row=0, col=3, scroll_view=scroll_view)
        writes_before = len(terminal.writes)
        # `_current_layout` is still None: nothing has been rendered yet.
        tui._copy_selection_to_clipboard()
        assert len(terminal.writes) == writes_before

    @pytest.mark.asyncio
    async def test_copy_selection_to_clipboard_is_a_noop_when_the_scroll_view_has_no_box(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha"))
        tui.start()
        await wait_render()
        orphan = ScrollView(_Text("orphan text"), ScrollViewOptions())
        tui._selection_anchor = SelectionPoint(row=0, col=0, scroll_view=orphan)
        tui._selection_focus = SelectionPoint(row=0, col=3, scroll_view=orphan)
        writes_before = len(terminal.writes)
        tui._copy_selection_to_clipboard()
        assert len(terminal.writes) == writes_before
        tui.stop()

    def test_copy_selection_with_only_blank_content_does_not_write_to_the_terminal(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui._previous_screen = [""]
        tui._selection_anchor = SelectionPoint(row=0, col=0)
        tui._selection_focus = SelectionPoint(row=0, col=1, boundary=True)
        writes_before = len(terminal.writes)
        tui._copy_selection_to_clipboard()
        assert len(terminal.writes) == writes_before

    def test_apply_selection_returns_the_screen_unchanged_before_any_render(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(_Text("alpha"), ScrollViewOptions())
        tui._selection_anchor = SelectionPoint(row=0, col=0, scroll_view=scroll_view)
        tui._selection_focus = SelectionPoint(row=0, col=3, scroll_view=scroll_view)
        screen = ["abcdef"]
        assert tui._apply_selection(screen, layout=None) == screen

    @pytest.mark.asyncio
    async def test_apply_selection_returns_the_screen_unchanged_when_the_scroll_view_box_is_missing(self) -> None:
        terminal = FakeTerminal(20, 4)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("alpha"))
        tui.start()
        await wait_render()
        orphan = ScrollView(_Text("orphan"), ScrollViewOptions())
        tui._selection_anchor = SelectionPoint(row=0, col=0, scroll_view=orphan)
        tui._selection_focus = SelectionPoint(row=0, col=3, scroll_view=orphan)
        screen = ["abcdef"]
        assert tui._apply_selection(screen) == screen
        tui.stop()


class _InputOverlay(Component):
    """Port of the upstream test's `InputOverlay`: records everything it receives."""

    def __init__(self) -> None:
        self.focused = False
        self.inputs: list[str] = []

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)

    def render(self, width: int) -> list[str]:
        del width
        return ["overlay"]

    def invalidate(self) -> None:
        pass


class TestOverlayViewportInput:
    """Port of upstream `2e4d23959`'s cases.

    Scrolling inside a focused overlay used to move the transcript behind it,
    because the alt screen consumed the wheel and the viewport keys before the
    overlay ever saw them.
    """

    @pytest.mark.asyncio
    async def test_gives_wheel_and_viewport_keys_to_a_focused_overlay(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(12))))
        overlay = _InputOverlay()
        tui.start()
        await wait_render()
        top_before = tui.viewport_top
        handle = tui.show_overlay(overlay)
        await wait_render()

        wheel = "\x1b[<64;10;3M"
        keys = ["\x1b[5~", "\x1b[6~", "\x1bOH", "\x1bOF", wheel]
        for key in keys:
            terminal.send_input(key)
        await wait_render()

        assert overlay.inputs == keys
        assert tui.viewport_top == top_before

        handle.hide()
        await wait_render()
        terminal.send_input("\x1b[5~")
        await wait_render()
        assert tui.viewport_top < top_before
        tui.stop()

    @pytest.mark.asyncio
    async def test_keeps_viewport_scrolling_when_an_overlay_is_not_focused(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("\n".join(f"line {i + 1}" for i in range(12))))
        tui.start()
        await wait_render()
        top_before = tui.viewport_top

        overlay = _InputOverlay()
        tui.show_overlay(overlay)
        await wait_render()
        tui.set_focus(None)
        await wait_render()

        terminal.send_input("\x1b[5~")
        await wait_render()

        assert tui.viewport_top < top_before
        tui.stop()
