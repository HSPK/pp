"""Python port of `packages/tui/test/tui-alt-screen.test.ts`.

`tests/test_tui_alt_screen.py` ports the scrolling/mouse/selection cases; this
module carries the remaining ones — the fixed dock, scrollbar-thumb dragging,
Kitty/iTerm2 image placement and grapheme-boundary selection — so every
TypeScript `it(...)` has a Python counterpart.
"""

from __future__ import annotations

import asyncio
import base64

import pytest
from pi_tui.component import Component
from pi_tui.components.image import Image, ImageOptions, ImageTheme
from pi_tui.components.scroll_view import ScrollView, ScrollViewOptions
from pi_tui.components.stack import Stack, StackEntry
from pi_tui.terminal_image import (
    ImageDimensions,
    KittyImageMetadata,
    TerminalCapabilities,
    encode_kitty,
    register_kitty_image_metadata,
    reset_capabilities_cache,
    set_capabilities,
)
from pi_tui.testing import FakeTerminal, MiniAltScreenModel, wait_until
from pi_tui.tui_alt_screen import TuiAltScreen


class _Text(Component):
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


class _Lines(Component):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, _width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        return None


class _VStack(Stack):
    layout_type = "vstack"


class _HStack(Stack):
    layout_type = "hstack"


async def wait_render() -> None:
    await asyncio.sleep(0.03)


def _viewport(terminal: FakeTerminal, width: int, height: int) -> list[str]:
    model = MiniAltScreenModel(width, height)
    for write in terminal.writes:
        model.feed(write)
    return [line.rstrip() for line in model.screen()]


def _all_writes(terminal: FakeTerminal) -> str:
    return "".join(terminal.writes)


class TestDockAndScrollbar:
    @pytest.mark.asyncio
    async def test_keeps_an_explicit_dock_fixed_while_the_transcript_scrolls(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        transcript_text = _Text("\n".join(f"line {i + 1}" for i in range(8)))
        transcript = ScrollView(transcript_text, ScrollViewOptions(follow="end", primary=True))
        dock = _VStack([_Text("editor"), _Text("footer")])
        tui.set_layout_root(
            _VStack(
                [
                    StackEntry(component=transcript, basis=0, grow=1, min_size=1),
                    StackEntry(component=dock, basis="auto", min_size=1),
                ]
            )
        )
        tui.start()
        await wait_render()

        assert _viewport(terminal, 20, 6) == ["line 5", "line 6", "line 7", "line 8", "editor", "footer"]

        # Wheel over the dock falls back to the primary transcript scroll view.
        terminal.send_input("\x1b[<64;1;6M")
        await wait_render()
        assert _viewport(terminal, 20, 6) == ["line 4", "line 5", "line 6", "line 7", "editor", "footer"]
        assert transcript.is_following_end is False

        transcript_text.set_text("\n".join(f"line {i + 1}" for i in range(10)))
        tui.request_render()
        await wait_render()
        assert _viewport(terminal, 20, 6) == ["line 4", "line 5", "line 6", "line 7", "editor", "footer"]

        tui.scroll_to_bottom()
        await wait_render()
        assert _viewport(terminal, 20, 6) == ["line 7", "line 8", "line 9", "line 10", "editor", "footer"]
        tui.stop()

    @pytest.mark.asyncio
    async def test_drags_a_visible_scrollbar_thumb_and_keeps_it_visible_until_release(self) -> None:
        terminal = FakeTerminal(10, 5)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(
            _Text("\n".join(f"line {i + 1}" for i in range(20))),
            ScrollViewOptions(primary=True, scrollbar="auto", scrollbar_hide_delay_ms=50),
        )
        tui.set_layout_root(scroll_view)
        tui.start()
        await wait_render()
        assert scroll_view.is_scrollbar_visible is False

        terminal.send_input("\x1b[<65;10;1M")
        await wait_render()
        assert scroll_view.scroll_top == 1
        assert scroll_view.is_scrollbar_visible is True

        terminal.send_input("\x1b[<0;10;1M")
        await wait_render()
        # Negative assertion, as in the TypeScript test's `setTimeout(70)`: the
        # 50 ms auto-hide must not fire while the thumb is held, so there is no
        # outcome to poll for and the wait stays a real one.
        await asyncio.sleep(0.07)
        assert scroll_view.is_scrollbar_visible is True

        terminal.send_input("\x1b[<32;10;4M")
        await wait_render()
        assert scroll_view.scroll_top == 15
        assert _viewport(terminal, 10, 5) == ["line 16", "line 17", "line 18", "line 19", "line 20"]

        terminal.send_input("\x1b[<0;10;4m")
        await wait_render()
        assert scroll_view.is_scrollbar_visible is True
        await asyncio.sleep(0.07)
        assert scroll_view.is_scrollbar_visible is True
        terminal.send_input("\x1b[<35;9;4M")
        # The 50 ms auto-hide runs on a `threading.Timer` here (a `setTimeout`
        # in TypeScript), so wait for the hide instead of assuming it lands
        # inside a fixed sleep on a loaded machine.
        await wait_until(
            lambda: scroll_view.is_scrollbar_visible is False,
            message="the scrollbar never auto-hid after the drag ended",
        )

        terminal.send_input("\x1b[<64;10;5M")
        await wait_render()
        assert scroll_view.scroll_top == 14
        await asyncio.sleep(0.07)
        assert scroll_view.is_scrollbar_visible is True
        terminal.send_input("\x1b[<35;9;5M")
        await wait_until(
            lambda: scroll_view.is_scrollbar_visible is False,
            message="the scrollbar never auto-hid after the wheel scroll",
        )

        # Dragging the thumb must never be mistaken for a text selection.
        assert "\x1b]52;c;" not in _all_writes(terminal)
        tui.stop()

    @pytest.mark.asyncio
    async def test_keeps_the_scrollbar_column_selectable_while_the_thumb_is_hidden(self) -> None:
        terminal = FakeTerminal(10, 2)
        tui = TuiAltScreen(terminal)
        scroll_view = ScrollView(
            _Text("123456789A\nabcdefghij\nmore\nlines"),
            ScrollViewOptions(scrollbar="auto"),
        )
        tui.set_layout_root(scroll_view)
        tui.start()
        await wait_render()
        assert scroll_view.is_scrollbar_visible is False

        terminal.send_input("\x1b[<0;10;1M")
        terminal.send_input("\x1b[<32;10;2M")
        terminal.send_input("\x1b[<0;10;2m")
        await wait_render()

        expected = base64.b64encode(b"A\nabcdefghij").decode("ascii")
        assert f"\x1b]52;c;{expected}\x07" in _all_writes(terminal)
        tui.stop()


class TestImagePlacement:
    @pytest.mark.asyncio
    async def test_does_not_emit_kitty_graphics_commands_or_osc_133_zones_in_iterm2(self) -> None:
        set_capabilities(TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True))
        try:
            terminal = FakeTerminal(20, 3)
            tui = TuiAltScreen(terminal)
            tui.add_child(_Lines(["\x1b]133;B\x07\x1b]133;C\x07\x1b]133;A\x07content"]))
            tui.add_child(
                Image(
                    "AAAA",
                    "image/png",
                    ImageTheme(fallback_color=lambda value: value),
                    ImageOptions(filename="example.png"),
                    ImageDimensions(width_px=10, height_px=10),
                )
            )
            tui.start()
            await wait_render()
            tui.stop()
            writes = _all_writes(terminal)
            assert "\x1b_G" not in writes
            assert "\x1b]133;" not in writes
            assert "\x1b]1337;File=" not in writes
            assert "[Image:" in writes
        finally:
            reset_capabilities_cache()

    @pytest.mark.asyncio
    async def test_clears_stale_iterm2_image_placements_when_they_leave_the_viewport(self) -> None:
        set_capabilities(TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True))
        try:
            terminal = FakeTerminal(20, 3)
            tui = TuiAltScreen(terminal)
            image_line = "\x1b]1337;File=inline=1;width=2;height=auto:AAAA\x07"
            tui.add_child(_Lines([image_line, "", "", "after", "more", "end"]))
            tui.start()
            await wait_render()
            tui.scroll_to_top()
            await wait_render()
            write_count = len(terminal.writes)

            tui.scroll_by(1)
            await wait_render()
            assert "\x1b[2J" in "".join(terminal.writes[write_count:])
            tui.stop()
        finally:
            reset_capabilities_cache()

    @pytest.mark.asyncio
    async def test_crops_a_kitty_image_whose_first_line_is_above_the_viewport(self) -> None:
        terminal = FakeTerminal(20, 3)
        tui = TuiAltScreen(terminal)
        image_id = 123
        image_line = encode_kitty("AAAA", columns=2, rows=3, image_id=image_id, move_cursor=False)
        register_kitty_image_metadata(
            KittyImageMetadata(image_id=image_id, columns=2, rows=3, width_px=100, height_px=100)
        )
        tui.add_child(_Lines(["before", image_line, "", "", "after", "end"]))
        tui.start()
        await wait_render()

        assert tui.viewport_top == 3
        writes = _all_writes(terminal)
        assert "i=123" in writes
        assert "y=66,h=34,r=1" in writes
        tui.stop()

    @pytest.mark.asyncio
    async def test_reuses_moved_kitty_images_without_dropping_hstack_siblings(self) -> None:
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        try:
            terminal = FakeTerminal(20, 6)
            tui = TuiAltScreen(terminal)
            label = _Text("left")
            image = Image(
                "A" * 8192,
                "image/png",
                ImageTheme(fallback_color=lambda value: value),
                ImageOptions(),
                ImageDimensions(width_px=100, height_px=100),
            )
            header = _Text("header")
            row = _HStack([StackEntry(component=label, basis=10), StackEntry(component=image, basis=10)])
            tui.set_layout_root(
                _VStack(
                    [
                        StackEntry(component=header, basis="auto"),
                        StackEntry(component=row, basis=4),
                    ]
                )
            )
            tui.start()
            await wait_render()
            assert "\x1b_Ga=T" in _all_writes(terminal)

            write_count = len(terminal.writes)
            label.set_text("changed")
            header.set_text("header\nsecond")
            tui.request_render()
            await wait_render()
            redraw = "".join(terminal.writes[write_count:])
            placement_index = redraw.find("\x1b_Ga=p,q=2")
            assert "\x1b_Ga=d,d=a,q=2\x1b\\" in redraw
            assert placement_index > redraw.find("changed")
            assert "\x1b_Ga=T" not in redraw
            assert len(redraw) < 2000, f"expected placement-only redraw, got {len(redraw)} bytes"
            assert any(line == "changed" for line in _viewport(terminal, 20, 6))
            tui.stop()
        finally:
            reset_capabilities_cache()

    @pytest.mark.asyncio
    async def test_retains_recently_offscreen_kitty_images_for_placement_only_reuse(self) -> None:
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        try:
            terminal = FakeTerminal(20, 1)
            tui = TuiAltScreen(terminal)
            image_id = 321
            image_line = encode_kitty("AAAA", columns=2, rows=1, image_id=image_id, move_cursor=False)
            register_kitty_image_metadata(
                KittyImageMetadata(image_id=image_id, columns=2, rows=1, width_px=100, height_px=50)
            )
            tui.set_layout_root(ScrollView(_Lines([image_line, "after"]), ScrollViewOptions(primary=True)))
            tui.start()
            await wait_render()
            assert "\x1b_Ga=T" in _all_writes(terminal)

            write_count = len(terminal.writes)
            tui.scroll_by(1)
            await wait_render()
            tui.scroll_by(-1)
            await wait_render()
            reentry = "".join(terminal.writes[write_count:])
            assert "\x1b_Ga=p,q=2" in reentry
            assert "\x1b_Ga=T" not in reentry
            assert f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\" not in reentry
            tui.stop()
        finally:
            reset_capabilities_cache()

    @pytest.mark.asyncio
    async def test_evicts_the_least_recently_visible_kitty_image_when_the_cache_is_full(self) -> None:
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        try:
            terminal = FakeTerminal(20, 1)
            tui = TuiAltScreen(terminal)
            first_image_id = 500
            image_lines: list[str] = []
            for index in range(18):
                image_id = first_image_id + index
                register_kitty_image_metadata(
                    KittyImageMetadata(image_id=image_id, columns=2, rows=1, width_px=100, height_px=50)
                )
                image_lines.append(encode_kitty("AAAA", columns=2, rows=1, image_id=image_id, move_cursor=False))
            tui.set_layout_root(ScrollView(_Lines(image_lines), ScrollViewOptions(primary=True)))
            tui.start()
            await wait_render()
            for _ in range(1, len(image_lines)):
                tui.scroll_by(1)
                await wait_render()
            assert f"\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\" in _all_writes(terminal)

            write_count = len(terminal.writes)
            tui.scroll_to_top()
            await wait_render()
            reentry = "".join(terminal.writes[write_count:])
            assert "\x1b_Ga=T" in reentry
            tui.stop()
        finally:
            reset_capabilities_cache()

    @pytest.mark.asyncio
    async def test_evicts_offscreen_kitty_images_when_raster_memory_exceeds_the_quota(self) -> None:
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        try:
            terminal = FakeTerminal(20, 1)
            tui = TuiAltScreen(terminal)
            first_image_id = 600
            image_lines: list[str] = []
            for index in range(4):
                image_id = first_image_id + index
                register_kitty_image_metadata(
                    KittyImageMetadata(image_id=image_id, columns=2, rows=1, width_px=3840, height_px=2160)
                )
                image_lines.append(encode_kitty("AAAA", columns=2, rows=1, image_id=image_id, move_cursor=False))
            tui.set_layout_root(ScrollView(_Lines(image_lines), ScrollViewOptions(primary=True)))
            tui.start()
            await wait_render()
            for _ in range(1, len(image_lines)):
                tui.scroll_by(1)
                await wait_render()
            assert f"\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\" in _all_writes(terminal)
            tui.stop()
        finally:
            reset_capabilities_cache()


class TestSelectionBoundaries:
    @pytest.mark.asyncio
    async def test_does_not_append_whitespace_to_double_click_word_highlighting(self) -> None:
        terminal = FakeTerminal(20, 1)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("foo  bar"))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<0;1;1m")
        terminal.send_input("\x1b[<0;3;1M")
        await wait_render()

        assert "foo\x1b[27m" in _all_writes(terminal)
        tui.stop()

    @pytest.mark.asyncio
    async def test_highlights_a_complete_whitespace_segment_during_a_word_drag(self) -> None:
        terminal = FakeTerminal(20, 1)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("foo  bar"))
        tui.start()
        await wait_render()

        terminal.send_input("\x1b[<0;1;1M")
        terminal.send_input("\x1b[<0;1;1m")
        terminal.send_input("\x1b[<0;2;1M")
        terminal.send_input("\x1b[<32;4;1M")
        await wait_render()

        assert "foo  \x1b[27m" in _all_writes(terminal)
        tui.stop()

    @pytest.mark.asyncio
    async def test_snaps_mouse_selection_to_cjk_emoji_and_combining_grapheme_boundaries(self) -> None:
        terminal = FakeTerminal(20, 2)
        tui = TuiAltScreen(terminal)
        tui.add_child(_Text("A界🙂éZ"))
        tui.start()
        await wait_render()

        wide_selection = f"\x1b]52;c;{base64.b64encode('界🙂'.encode()).decode('ascii')}\x07"
        terminal.send_input("\x1b[<0;3;1M")
        terminal.send_input("\x1b[<32;4;1M")
        terminal.send_input("\x1b[<0;4;1m")
        await wait_render()
        assert _all_writes(terminal).count(wide_selection) == 1

        # The reverse drag (right-to-left) must snap to the same grapheme run.
        terminal.send_input("\x1b[<0;5;1M")
        terminal.send_input("\x1b[<32;2;1M")
        terminal.send_input("\x1b[<0;2;1m")
        await wait_render()
        assert _all_writes(terminal).count(wide_selection) == 2

        combining_selection = f"\x1b]52;c;{base64.b64encode('éZ'.encode()).decode('ascii')}\x07"
        terminal.send_input("\x1b[<0;6;1M")
        terminal.send_input("\x1b[<32;7;1M")
        terminal.send_input("\x1b[<0;7;1m")
        await wait_render()
        assert combining_selection in _all_writes(terminal)
        tui.stop()
