"""Tests for the differential renderer (`TuiMainScreen.do_render`).

Python port of the non-Kitty-image cases in `packages/tui/test/tui-render.test.ts`.

Content assertions use `MiniTerminalModel` (see `pi_tui.testing`), a small
escape-sequence interpreter covering only the sequences `TuiMainScreen`
actually emits, playing the same role as the TypeScript suite's xterm.js-backed
`VirtualTerminal` without depending on a real terminal-emulator library. A
couple of the simplest scenarios (initial paint, single-line change) are also
asserted byte-for-byte against the exact written escape sequences, per the
task's emphasis on validating the differential renderer literally.

Kitty terminal image cases from the TypeScript suite live in
`tests/test_tui_render_kitty_images.py`, which forces image capabilities and
fixed cell dimensions for the duration of each test.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui import TuiStopOptions
from pi_tui.tui_main_screen import (
    TuiMainScreen,
    _extract_kitty_image_ids,
    _extract_kitty_image_rows,
    _parse_kitty_image_header,
    delete_kitty_image,
)

_SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"


class _TestComponent:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or []

    def render(self, _width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        return None


class _InputComponent(_TestComponent):
    def __init__(self) -> None:
        super().__init__()
        self.render_count = 0

    def render(self, width: int) -> list[str]:
        self.render_count += 1
        return super().render(width)

    def handle_input(self, data: str) -> None:
        self.lines = [data]


async def wait_render() -> None:
    await asyncio.sleep(0.03)


class TestRenderScheduling:
    @pytest.mark.asyncio
    async def test_renders_keyboard_input_without_waiting_for_throttled_frame(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _InputComponent()
        component.lines = ["initial"]
        tui.add_child(component)
        tui.set_focus(component)
        tui.start()
        tui.render_now()
        render_count_before_input = component.render_count

        component.lines = ["pending"]
        tui.request_render()
        terminal.send_input("first")
        terminal.send_input("second")
        terminal.send_input("typed")
        await asyncio.sleep(0)

        assert component.render_count == render_count_before_input + 1
        assert component.lines == ["typed"]
        tui.stop()


class TestDebugLogging:
    @pytest.mark.asyncio
    async def test_writes_redraw_logs_to_provided_directory(self, tmp_path: Path) -> None:
        # `tmp_path`, not a fixed directory beside this file: under `-n auto`
        # several workers run this module concurrently and the old shared path
        # meant one worker's `rmtree` deleted another's log mid-run.
        log_dir = tmp_path / "debug-log"
        log_dir.mkdir(parents=True)
        previous = os.environ.get("PI_DEBUG_REDRAW")
        os.environ["PI_DEBUG_REDRAW"] = "1"
        try:
            terminal = FakeTerminal(40, 10)
            tui = TuiMainScreen(terminal, None, str(log_dir))
            component = _TestComponent(["test"])
            tui.add_child(component)
            tui.start()
            await wait_render()

            log_content = (log_dir / "pi-debug.log").read_text(encoding="utf-8")
            assert "fullRender: first render" in log_content
            tui.stop()
        finally:
            if previous is None:
                os.environ.pop("PI_DEBUG_REDRAW", None)
            else:
                os.environ["PI_DEBUG_REDRAW"] = previous


class TestExactByteOutput:
    @pytest.mark.asyncio
    async def test_initial_render_writes_lines_joined_by_crlf(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2"])
        tui.add_child(component)
        tui.start()
        terminal.writes.clear()
        await wait_render()

        expected = (
            "\x1b[?2026h"
            + f"Line 0{_SEGMENT_RESET}"
            + "\r\n"
            + f"Line 1{_SEGMENT_RESET}"
            + "\r\n"
            + f"Line 2{_SEGMENT_RESET}"
            + "\x1b[?2026l"
            + "\x1b[?25l"
        )
        assert "".join(terminal.writes) == expected
        tui.stop()

    @pytest.mark.asyncio
    async def test_single_middle_line_change_writes_minimal_diff(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Header", "Working...", "Footer"])
        tui.add_child(component)
        tui.start()
        await wait_render()
        terminal.writes.clear()

        component.lines = ["Header", "Working |", "Footer"]
        tui.request_render()
        await wait_render()

        expected = (
            "\x1b[?2026h" + "\x1b[1A" + "\r" + "\x1b[2K" + f"Working |{_SEGMENT_RESET}" + "\x1b[?2026l" + "\x1b[?25l"
        )
        assert "".join(terminal.writes) == expected
        tui.stop()


class TestResizeHandling:
    @pytest.mark.asyncio
    async def test_triggers_full_rerender_when_height_changes(self) -> None:
        previous = os.environ.pop("TERMUX_VERSION", None)
        try:
            terminal = FakeTerminal(40, 10)
            tui = TuiMainScreen(terminal)
            component = _TestComponent(["Line 0", "Line 1", "Line 2"])
            tui.add_child(component)
            tui.start()
            await wait_render()

            initial_redraws = tui.full_redraws

            terminal.send_resize(40, 15)
            await wait_render()

            assert tui.full_redraws > initial_redraws

            viewport = _render_model(terminal, 40, 15).viewport()
            assert "Line 0" in viewport[0]
            tui.stop()
        finally:
            if previous is not None:
                os.environ["TERMUX_VERSION"] = previous

    @pytest.mark.asyncio
    async def test_skips_full_rerender_on_height_change_in_termux(self) -> None:
        previous = os.environ.get("TERMUX_VERSION")
        os.environ["TERMUX_VERSION"] = "1"
        try:
            terminal = FakeTerminal(40, 10)
            tui = TuiMainScreen(terminal)
            component = _TestComponent([f"Line {i}" for i in range(20)])
            tui.add_child(component)
            tui.start()
            await wait_render()
            # The TS test clears the emulator's write log here; the emulator
            # keeps its screen state. `MiniTerminalModel` is rebuilt from the
            # write log, so record an offset instead of dropping the writes.
            writes_before_resizes = len(terminal.writes)

            initial_redraws = tui.full_redraws
            for height in (15, 8, 14, 11):
                terminal.send_resize(40, height)
                await wait_render()

            assert tui.full_redraws == initial_redraws
            joined = "".join(terminal.writes[writes_before_resizes:])
            assert "\x1b[2J" not in joined
            assert "\x1b[3J" not in joined

            viewport = _render_model(terminal, 40, 11).viewport()
            assert "Line 19" in "\n".join(viewport)
            tui.stop()
        finally:
            if previous is None:
                os.environ.pop("TERMUX_VERSION", None)
            else:
                os.environ["TERMUX_VERSION"] = previous

    @pytest.mark.asyncio
    async def test_triggers_full_rerender_when_width_changes(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws
        terminal.send_resize(60, 10)
        await wait_render()

        assert tui.full_redraws > initial_redraws
        tui.stop()


def _render_model(terminal: FakeTerminal, width: int, height: int) -> MiniTerminalModel:
    model = MiniTerminalModel(width, height)
    for write in terminal.writes:
        model.feed(write)
    return model


class TestContentShrinkage:
    @pytest.mark.asyncio
    async def test_clears_empty_rows_when_content_shrinks_significantly(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui.set_clear_on_shrink(True)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4", "Line 5"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws

        component.lines = ["Line 0", "Line 1"]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws > initial_redraws

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "Line 0" in viewport[0]
        assert "Line 1" in viewport[1]
        assert viewport[2].strip() == ""
        assert viewport[3].strip() == ""
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_shrink_to_single_line(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui.set_clear_on_shrink(True)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = ["Only line"]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "Only line" in viewport[0]
        assert viewport[1].strip() == ""
        tui.stop()

    @pytest.mark.asyncio
    async def test_handles_shrink_to_empty(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui.set_clear_on_shrink(True)
        component = _TestComponent(["Line 0", "Line 1", "Line 2"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = []
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert viewport[0].strip() == ""
        assert viewport[1].strip() == ""
        tui.stop()


class TestDifferentialRendering:
    @pytest.mark.asyncio
    async def test_tracks_cursor_when_content_shrinks_with_unchanged_lines(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = ["Line 0", "Line 1", "Line 2"]
        tui.request_render()
        await wait_render()

        component.lines = ["Line 0", "CHANGED", "Line 2"]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "CHANGED" in viewport[1]
        tui.stop()

    @pytest.mark.asyncio
    async def test_spinner_case_only_middle_line_changes(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Header", "Working...", "Footer"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        for frame in ("|", "/", "-", "\\"):
            component.lines = ["Header", f"Working {frame}", "Footer"]
            tui.request_render()
            await wait_render()

            viewport = _render_model(terminal, 40, 10).viewport()
            assert "Header" in viewport[0]
            assert f"Working {frame}" in viewport[1]
            assert "Footer" in viewport[2]

        tui.stop()

    @pytest.mark.asyncio
    async def test_resets_styles_after_each_rendered_line(self) -> None:
        terminal = FakeTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["\x1b[3mItalic", "Plain"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        model = _render_model(terminal, 20, 6)
        assert model.cell_italic(1, 0) is False
        tui.stop()

    @pytest.mark.asyncio
    async def test_first_line_changes_rest_stays_same(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = ["CHANGED", "Line 1", "Line 2", "Line 3"]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "CHANGED" in viewport[0]
        assert "Line 1" in viewport[1]
        assert "Line 2" in viewport[2]
        assert "Line 3" in viewport[3]
        tui.stop()

    @pytest.mark.asyncio
    async def test_last_line_changes_rest_stays_same(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = ["Line 0", "Line 1", "Line 2", "CHANGED"]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "Line 0" in viewport[0]
        assert "Line 1" in viewport[1]
        assert "Line 2" in viewport[2]
        assert "CHANGED" in viewport[3]
        tui.stop()

    @pytest.mark.asyncio
    async def test_multiple_non_adjacent_lines_change(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = ["Line 0", "CHANGED 1", "Line 2", "CHANGED 3", "Line 4"]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "Line 0" in viewport[0]
        assert "CHANGED 1" in viewport[1]
        assert "Line 2" in viewport[2]
        assert "CHANGED 3" in viewport[3]
        assert "Line 4" in viewport[4]
        tui.stop()

    @pytest.mark.asyncio
    async def test_transition_from_content_to_empty_and_back(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "Line 0" in viewport[0]

        component.lines = []
        tui.request_render()
        await wait_render()

        component.lines = ["New Line 0", "New Line 1"]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert "New Line 0" in viewport[0]
        assert "New Line 1" in viewport[1]
        tui.stop()

    @pytest.mark.asyncio
    async def test_full_rerenders_when_deleted_lines_move_viewport_upward(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiMainScreen(terminal)
        component = _TestComponent([f"Line {i}" for i in range(12)])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws

        component.lines = [f"Line {i}" for i in range(7)]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws > initial_redraws
        viewport = _render_model(terminal, 20, 5).viewport()
        assert viewport == ["Line 2", "Line 3", "Line 4", "Line 5", "Line 6"]
        tui.stop()

    @pytest.mark.asyncio
    async def test_appends_after_shrink_without_another_full_redraw(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiMainScreen(terminal)
        component = _TestComponent([f"Line {i}" for i in range(8)])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws

        component.lines = ["Line 0", "Line 1"]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws > initial_redraws
        redraws_after_shrink = tui.full_redraws

        component.lines = ["Line 0", "Line 1", "Line 2"]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws == redraws_after_shrink
        viewport = _render_model(terminal, 20, 5).viewport()
        assert viewport == ["Line 0", "Line 1", "Line 2", "", ""]
        tui.stop()

    @pytest.mark.asyncio
    async def test_clears_stale_content_inflated_by_transient_component(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        chat = _TestComponent()
        editor = _TestComponent()
        tui.add_child(chat)
        tui.add_child(editor)

        long_chat = [f"Chat {i}" for i in range(15)]
        short_chat = [f"Chat {i}" for i in range(12)]
        editor_lines = ["Editor 0", "Editor 1", "Editor 2"]
        selector_lines = [f"Selector {i}" for i in range(8)]

        chat.lines = long_chat
        editor.lines = editor_lines
        tui.start()
        await wait_render()

        editor.lines = selector_lines
        tui.request_render()
        await wait_render()

        editor.lines = editor_lines
        tui.request_render()
        await wait_render()

        redraws_before_switch = tui.full_redraws
        chat.lines = short_chat
        tui.request_render()
        await wait_render()

        assert tui.full_redraws > redraws_before_switch

        viewport = _render_model(terminal, 40, 10).viewport()
        for line in viewport[:10]:
            assert "Chat 12" not in line
            assert "Chat 13" not in line
            assert "Chat 14" not in line

        assert viewport == [
            "Chat 5",
            "Chat 6",
            "Chat 7",
            "Chat 8",
            "Chat 9",
            "Chat 10",
            "Chat 11",
            "Editor 0",
            "Editor 1",
            "Editor 2",
        ]
        tui.stop()


class TestKittyHelperFunctions:
    """Direct unit tests for the module-level Kitty escape-sequence helpers.

    These are pure text-parsing helpers used by `_expand_changed_range_for_kitty_images`
    and friends; they are exercised here directly regardless of whether the
    render loop currently sets `_is_image_line` to always return `False`
    (see the module docstring: they are kept for line-for-line TS parity).
    """

    def test_parses_ids_and_row_count(self) -> None:
        header = _parse_kitty_image_header("\x1b_Ga=T,i=42,r=3;AAAA\x1b\\")
        assert header is not None
        assert header.ids == [42]
        assert header.rows == 3

    def test_returns_none_without_kitty_prefix(self) -> None:
        assert _parse_kitty_image_header("plain text, no image here") is None

    def test_returns_none_without_semicolon_terminator(self) -> None:
        assert _parse_kitty_image_header("\x1b_Ga=T,i=1") is None

    def test_ignores_malformed_and_out_of_range_params(self) -> None:
        # "i=oops" fails int(); "y=..." overflows uint32; both silently skipped.
        header = _parse_kitty_image_header("\x1b_Ga=T,i=oops,y=99999999999,solo;data")
        assert header is not None
        assert header.ids == []
        assert header.rows == 1

    def test_defaults_rows_to_one_when_not_present(self) -> None:
        header = _parse_kitty_image_header("\x1b_Ga=T,i=7;data")
        assert header is not None
        assert header.rows == 1

    def test_extract_ids_returns_empty_list_when_no_header(self) -> None:
        assert _extract_kitty_image_ids("no image here") == []

    def test_extract_rows_defaults_to_one_when_no_header(self) -> None:
        assert _extract_kitty_image_rows("no image here") == 1

    def test_delete_kitty_image_emits_the_kitty_delete_sequence(self) -> None:
        assert delete_kitty_image(99) == "\x1b_Ga=d,d=I,i=99,q=2\x1b\\"


class TestKittyImageInstanceHelpers:
    """Tests the `TuiMainScreen` instance-bound Kitty bookkeeping helpers directly."""

    def test_collect_kitty_image_ids_scans_all_lines(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        lines = ["plain", "\x1b_Ga=T,i=5,r=1;data\x1b\\", "\x1b_Ga=T,i=9,r=1;data\x1b\\"]
        assert tui._collect_kitty_image_ids(lines) == {5, 9}

    def test_delete_kitty_images_joins_one_delete_sequence_per_id(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        assert tui._delete_kitty_images({1, 2, 3}) == "".join(delete_kitty_image(i) for i in (1, 2, 3))

    def test_get_kitty_image_reserved_rows_defaults_to_one_for_plain_lines(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        lines = ["Line 0", "Line 1"]
        assert tui._get_kitty_image_reserved_rows(lines, 0) == 1

    def test_get_kitty_image_reserved_rows_expands_over_blank_trailing_lines(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        lines = ["\x1b_Ga=T,i=1,r=3;data\x1b\\", "", "", "Line 3"]
        # Header declares 3 rows; the two blank lines following it are reserved padding.
        assert tui._get_kitty_image_reserved_rows(lines, 0) == 3

    def test_get_kitty_image_reserved_rows_stops_at_first_non_blank_line(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        lines = ["\x1b_Ga=T,i=1,r=3;data\x1b\\", "", "Not blank", "Line 3"]
        # Row 2 is non-blank, so only 2 rows (the header line + one blank) are reserved.
        assert tui._get_kitty_image_reserved_rows(lines, 0) == 2

    def test_get_kitty_image_reserved_rows_bounds_to_remaining_lines(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        lines = ["\x1b_Ga=T,i=1,r=10;data\x1b\\", ""]
        # Only 1 more line exists after the header, so reserved rows can't exceed that.
        assert tui._get_kitty_image_reserved_rows(lines, 0) == 2

    def test_expand_changed_range_grows_to_cover_full_image_block(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        new_lines = ["Header", "\x1b_Ga=T,i=1,r=3;data\x1b\\", "", "", "Footer"]
        tui._previous_lines = ["Header", "", "", "", "Footer"]
        first, last = tui._expand_changed_range_for_kitty_images(1, 1, new_lines)
        assert first == 1
        assert last == 3

    def test_expand_changed_range_leaves_unrelated_ranges_untouched(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        # The image occupies index 0 only (rows=1, so its block is just [0, 0]),
        # entirely before the changed range [2, 2] - it should not be pulled in.
        new_lines = ["\x1b_Ga=T,i=1,r=1;data\x1b\\", "Body", "Changed"]
        tui._previous_lines = ["\x1b_Ga=T,i=1,r=1;data\x1b\\", "Body", "Old"]
        first, last = tui._expand_changed_range_for_kitty_images(2, 2, new_lines)
        assert (first, last) == (2, 2)

    def test_expand_changed_range_pulls_in_any_image_at_or_after_first_changed(self) -> None:
        # Matches `tui-main-screen.ts`'s `expandChangedRangeForKittyImages`: any image
        # whose start index is at or after `firstChanged` is swept into the redraw
        # range, since a differential redraw starting there could disturb it.
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        new_lines = ["Header", "Body", "\x1b_Ga=T,i=1,r=1;data\x1b\\"]
        tui._previous_lines = ["Header", "Body", "\x1b_Ga=T,i=1,r=1;data\x1b\\"]
        first, last = tui._expand_changed_range_for_kitty_images(0, 0, new_lines)
        assert (first, last) == (0, 2)

    def test_delete_changed_kitty_images_returns_empty_for_invalid_range(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        assert tui._delete_changed_kitty_images(-1, 5) == ""
        assert tui._delete_changed_kitty_images(5, 2) == ""

    def test_delete_changed_kitty_images_collects_ids_in_range(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Header", "\x1b_Ga=T,i=1,r=1;data\x1b\\", "\x1b_Ga=T,i=2,r=1;data\x1b\\", "Footer"]
        assert tui._delete_changed_kitty_images(1, 2) == delete_kitty_image(1) + delete_kitty_image(2)


class TestRenderStateCaptureRestore:
    def test_capture_and_restore_round_trips_render_state(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Line 0", "Line 1"]
        tui._previous_width = 40
        tui._previous_height = 10
        tui._cursor_row = 1
        tui._hardware_cursor_row = 1
        tui._max_lines_rendered = 2
        tui._previous_viewport_top = 0

        state = tui.capture_render_state()

        tui._previous_lines = []
        tui._previous_width = 0
        tui._previous_height = 0
        tui._cursor_row = 0
        tui._hardware_cursor_row = 0
        tui._max_lines_rendered = 0
        tui._previous_viewport_top = 5

        tui.restore_render_state(state)

        assert tui._previous_lines == ["Line 0", "Line 1"]
        assert tui._previous_width == 40
        assert tui._previous_height == 10
        assert tui._cursor_row == 1
        assert tui._hardware_cursor_row == 1
        assert tui._max_lines_rendered == 2
        assert tui._previous_viewport_top == 0

    def test_capture_returns_an_independent_copy_of_previous_lines(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Line 0"]
        state = tui.capture_render_state()
        tui._previous_lines.append("Line 1")
        assert state.previous_lines == ["Line 0"]

    def test_reset_render_state_clears_everything(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Line 0", "Line 1"]
        tui._previous_width = 40
        tui._previous_height = 10
        tui._cursor_row = 1
        tui._hardware_cursor_row = 1
        tui._max_lines_rendered = 2
        tui._previous_viewport_top = 3

        tui.reset_render_state()

        assert tui._previous_lines == []
        assert tui._previous_width == -1
        assert tui._previous_height == -1
        assert tui._cursor_row == 0
        assert tui._hardware_cursor_row == 0
        assert tui._max_lines_rendered == 0
        assert tui._previous_viewport_top == 0


class TestBeforeTerminalStop:
    def test_does_nothing_when_preserve_screen_is_requested(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Line 0", "Line 1"]
        terminal.writes.clear()
        tui.before_terminal_stop(TuiStopOptions(preserve_screen=True))
        assert terminal.writes == []

    def test_does_nothing_when_nothing_was_ever_rendered(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        terminal.writes.clear()
        tui.before_terminal_stop(TuiStopOptions(preserve_screen=False))
        assert terminal.writes == []

    def test_moves_cursor_down_when_target_row_is_below_hardware_cursor(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Line 0", "Line 1", "Line 2"]
        tui._hardware_cursor_row = 0
        terminal.writes.clear()
        tui.before_terminal_stop(TuiStopOptions(preserve_screen=False))
        joined = "".join(terminal.writes)
        assert "\x1b[3B" in joined
        assert joined.endswith("\r\n")

    def test_moves_cursor_up_when_target_row_is_above_hardware_cursor(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._previous_lines = ["Line 0"]
        tui._hardware_cursor_row = 5
        terminal.writes.clear()
        tui.before_terminal_stop(TuiStopOptions(preserve_screen=False))
        joined = "".join(terminal.writes)
        assert "\x1b[4A" in joined
        assert joined.endswith("\r\n")


class TestStoppedRenderIsNoOp:
    @pytest.mark.asyncio
    async def test_do_render_after_stop_writes_nothing_further(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0"])
        tui.add_child(component)
        tui.start()
        await wait_render()
        tui.stop()
        terminal.writes.clear()

        tui.do_render()

        assert terminal.writes == []


class TestCrashOnOverflowLine:
    @pytest.mark.asyncio
    async def test_raises_and_writes_crash_log_when_line_exceeds_width(self) -> None:
        log_dir = Path(__file__).parent / "_crash_log_test_dir"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)
        try:
            terminal = FakeTerminal(10, 5)
            tui = TuiMainScreen(terminal, None, str(log_dir))
            component = _TestComponent(["short"])
            tui.add_child(component)
            tui.start()
            await wait_render()

            # Grow a single line far past the 10-column terminal width to trigger the crash path.
            component.lines = ["short", "x" * 50]
            with pytest.raises(RuntimeError, match="exceeds terminal width"):
                tui.do_render()

            crash_log = log_dir / "pi-crash.log"
            assert crash_log.exists()
            content = crash_log.read_text(encoding="utf-8")
            assert "Crash at" in content
            assert "Terminal width: 10" in content
            assert "All rendered lines" in content
            assert tui.stopped is True
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


class TestPositionHardwareCursorDirect:
    def test_hides_cursor_when_no_cursor_position_available(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        terminal.writes.clear()
        tui._position_hardware_cursor(None, 3)
        assert "\x1b[?25l" in terminal.writes

    def test_hides_cursor_when_total_lines_is_zero(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        terminal.writes.clear()
        tui._position_hardware_cursor((0, 0), 0)
        assert "\x1b[?25l" in terminal.writes

    def test_shows_cursor_and_moves_it_when_enabled(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal, show_hardware_cursor=True)
        tui._hardware_cursor_row = 0
        terminal.writes.clear()
        tui._position_hardware_cursor((2, 4), 5)
        joined = "".join(terminal.writes)
        assert "\x1b[2B" in joined
        assert "\x1b[5G" in joined
        assert tui._hardware_cursor_row == 2
        assert "\x1b[?25h" in terminal.writes

    def test_moving_cursor_upward_emits_up_sequence(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal, show_hardware_cursor=True)
        tui._hardware_cursor_row = 4
        terminal.writes.clear()
        tui._position_hardware_cursor((1, 0), 5)
        joined = "".join(terminal.writes)
        assert "\x1b[3A" in joined
        assert tui._hardware_cursor_row == 1

    def test_same_row_only_moves_column_without_vertical_sequence(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal, show_hardware_cursor=True)
        tui._hardware_cursor_row = 2
        terminal.writes.clear()
        tui._position_hardware_cursor((2, 3), 5)
        joined = "".join(terminal.writes)
        assert "\x1b[4G" in joined
        assert "A" not in joined
        assert "B" not in joined

    def test_hides_cursor_by_default_when_show_hardware_cursor_not_enabled(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        tui._hardware_cursor_row = 0
        terminal.writes.clear()
        tui._position_hardware_cursor((0, 0), 3)
        assert "\x1b[?25l" in terminal.writes


class TestPartialDeletionWithoutFullRedraw:
    @pytest.mark.asyncio
    async def test_shrinks_trailing_lines_without_triggering_full_redraw(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = _TestComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws
        # The unchanged prefix ("Line 0", "Line 1", "Line 2") stays put; only the
        # trailing two lines are removed. This should be a differential deletion,
        # not a full re-render (small shrink, fits well within the viewport).
        component.lines = ["Line 0", "Line 1", "Line 2"]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws == initial_redraws
        viewport = _render_model(terminal, 40, 10).viewport()
        assert viewport[0] == "Line 0"
        assert viewport[1] == "Line 1"
        assert viewport[2] == "Line 2"
        assert viewport[3].strip() == ""
        tui.stop()

    @pytest.mark.asyncio
    async def test_moves_cursor_down_when_trailing_lines_after_last_change_are_dropped(self) -> None:
        terminal = FakeTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        # The trailing blank line is already blank in both old and new content, so it
        # never registers as "changed" - only the early "B" -> "X" edit does. This
        # exercises the branch where `render_end` ends up strictly before the new
        # content's final line, requiring an explicit cursor move down afterwards.
        component = _TestComponent(["A", "B", "C", "", ""])
        tui.add_child(component)
        tui.start()
        await wait_render()

        component.lines = ["A", "X", "C", ""]
        tui.request_render()
        await wait_render()

        viewport = _render_model(terminal, 40, 10).viewport()
        assert viewport[0] == "A"
        assert viewport[1] == "X"
        assert viewport[2] == "C"
        assert viewport[3].strip() == ""
        tui.stop()

    @pytest.mark.asyncio
    async def test_scrolls_viewport_down_when_appending_far_beyond_current_content(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiMainScreen(terminal)
        component = _TestComponent([f"Line {i}" for i in range(5)])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws
        component.lines = [f"Line {i}" for i in range(5)] + [f"New {i}" for i in range(20)]
        tui.request_render()
        await wait_render()

        # A large append should scroll the viewport differentially, not trigger a full redraw.
        assert tui.full_redraws == initial_redraws
        viewport = _render_model(terminal, 20, 5).viewport()
        assert viewport == ["New 15", "New 16", "New 17", "New 18", "New 19"]
        tui.stop()

    @pytest.mark.asyncio
    async def test_moves_cursor_up_to_deleted_line_target_without_moving_viewport(self) -> None:
        terminal = FakeTerminal(20, 5)
        tui = TuiMainScreen(terminal)
        component = _TestComponent([f"Line {i}" for i in range(5)])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws
        # All 5 lines fit exactly in the 5-row viewport (viewport top stays at 0).
        # Shrinking to 3 lines only deletes trailing content - the cursor needs to
        # move *up* to the new last line, without moving the viewport or triggering
        # a full re-render.
        component.lines = [f"Line {i}" for i in range(3)]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws == initial_redraws
        viewport = _render_model(terminal, 20, 5).viewport()
        assert viewport[0] == "Line 0"
        assert viewport[1] == "Line 1"
        assert viewport[2] == "Line 2"
        tui.stop()

    @pytest.mark.asyncio
    async def test_full_rerenders_when_deleted_extra_lines_exceed_height(self) -> None:
        terminal = FakeTerminal(10, 3)
        tui = TuiMainScreen(terminal)
        component = _TestComponent([f"Line {i}" for i in range(20)])
        tui.add_child(component)
        tui.start()
        await wait_render()

        initial_redraws = tui.full_redraws
        # Removing 18 lines at once (far more than the 3-row terminal height) can't
        # be cleared with per-line cursor moves, so it must fall back to a full redraw.
        component.lines = [f"Line {i}" for i in range(2)]
        tui.request_render()
        await wait_render()

        assert tui.full_redraws > initial_redraws
        viewport = _render_model(terminal, 10, 3).viewport()
        assert viewport[0] == "Line 0"
        assert viewport[1] == "Line 1"
        tui.stop()
