"""Differential renderer for the regular (non-alt-screen) terminal mode.

Python port of `packages/tui/src/tui-main-screen.ts`.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pi_tui.terminal import Terminal
from pi_tui.terminal_image import delete_kitty_image, is_image_line
from pi_tui.tui import TuiBase, TuiStopOptions
from pi_tui.utils import visible_width

_KITTY_SEQUENCE_PREFIX = "\x1b_G"
_KITTY_PARAM_PATTERN = re.compile(r"^[a-zA-Z]=-?\d+$")


@dataclass
class _KittyImageHeader:
    ids: list[int] = field(default_factory=list)
    rows: int = 1


def _parse_kitty_image_header(line: str) -> _KittyImageHeader | None:
    sequence_start = line.find(_KITTY_SEQUENCE_PREFIX)
    if sequence_start == -1:
        return None
    params_start = sequence_start + len(_KITTY_SEQUENCE_PREFIX)
    params_end = line.find(";", params_start)
    if params_end == -1:
        return None

    ids: list[int] = []
    rows = 1
    for param in line[params_start:params_end].split(","):
        key_value = param.split("=", 1)
        if len(key_value) != 2:
            continue
        key, value = key_value
        try:
            number_value = int(value)
        except ValueError:
            continue
        if number_value <= 0 or number_value > 0xFFFFFFFF:
            continue
        if key == "i":
            ids.append(number_value)
        elif key == "r":
            rows = number_value
    return _KittyImageHeader(ids=ids, rows=rows)


def _extract_kitty_image_ids(line: str) -> list[int]:
    header = _parse_kitty_image_header(line)
    return header.ids if header is not None else []


def _extract_kitty_image_rows(line: str) -> int:
    header = _parse_kitty_image_header(line)
    return header.rows if header is not None else 1


def _is_termux_session() -> bool:
    return bool(os.environ.get("TERMUX_VERSION"))


@dataclass
class TuiMainScreenRenderState:
    previous_lines: list[str]
    previous_width: int
    previous_height: int
    cursor_row: int
    hardware_cursor_row: int
    max_lines_rendered: int
    previous_viewport_top: int


class TuiMainScreen(TuiBase):
    """TUI implementation that renders into the terminal's main screen and scrollback."""

    mode = "regular"

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor, log_directory)
        self._previous_lines: list[str] = []
        self._previous_kitty_image_ids: set[int] = set()
        self._previous_width = 0
        self._previous_height = 0
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0

    def capture_render_state(self) -> TuiMainScreenRenderState:
        return TuiMainScreenRenderState(
            previous_lines=list(self._previous_lines),
            previous_width=self._previous_width,
            previous_height=self._previous_height,
            cursor_row=self._cursor_row,
            hardware_cursor_row=self._hardware_cursor_row,
            max_lines_rendered=self._max_lines_rendered,
            previous_viewport_top=self._previous_viewport_top,
        )

    def restore_render_state(self, state: TuiMainScreenRenderState) -> None:
        self._previous_lines = ["" if is_image_line(line) else line for line in state.previous_lines]
        self._previous_kitty_image_ids = set()
        self._previous_width = state.previous_width
        self._previous_height = state.previous_height
        self._cursor_row = state.cursor_row
        self._hardware_cursor_row = state.hardware_cursor_row
        self._max_lines_rendered = state.max_lines_rendered
        self._previous_viewport_top = state.previous_viewport_top

    def reset_render_state(self) -> None:
        self._previous_lines = []
        self._previous_width = -1
        self._previous_height = -1
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0

    def before_terminal_stop(self, options: TuiStopOptions) -> None:
        if options.preserve_screen or len(self._previous_lines) == 0:
            return
        self.terminal.write(" ")
        target_row = len(self._previous_lines)
        line_diff = target_row - self._hardware_cursor_row
        if line_diff > 0:
            self.terminal.write(f"\x1b[{line_diff}B")
        elif line_diff < 0:
            self.terminal.write(f"\x1b[{-line_diff}A")
        self.terminal.write("\r\n")

    def _collect_kitty_image_ids(self, lines: list[str]) -> set[int]:
        ids: set[int] = set()
        for line in lines:
            ids.update(_extract_kitty_image_ids(line))
        return ids

    def _delete_kitty_images(self, ids: set[int]) -> str:
        return "".join(delete_kitty_image(image_id) for image_id in ids)

    def _get_kitty_image_reserved_rows(self, lines: list[str], index: int, max_index: int | None = None) -> int:
        if max_index is None:
            max_index = len(lines) - 1
        rows = _extract_kitty_image_rows(lines[index] if index < len(lines) else "")
        if rows <= 1:
            return 1

        max_rows = min(rows, max_index - index + 1, len(lines) - index)
        reserved_rows = 1
        while reserved_rows < max_rows:
            line = lines[index + reserved_rows] if index + reserved_rows < len(lines) else ""
            if is_image_line(line) or visible_width(line) > 0:
                break
            reserved_rows += 1
        return reserved_rows

    def _expand_changed_range_for_kitty_images(
        self, first_changed: int, last_changed: int, new_lines: list[str]
    ) -> tuple[int, int]:
        expanded_first_changed = first_changed
        expanded_last_changed = last_changed

        def _expand_for_lines(lines: list[str]) -> None:
            nonlocal expanded_first_changed, expanded_last_changed
            for i, line in enumerate(lines):
                if len(_extract_kitty_image_ids(line)) == 0:
                    continue
                block_end = i + self._get_kitty_image_reserved_rows(lines, i) - 1
                if i >= first_changed or (i <= last_changed and block_end >= first_changed):
                    expanded_first_changed = min(expanded_first_changed, i)
                    expanded_last_changed = max(expanded_last_changed, block_end)

        _expand_for_lines(self._previous_lines)
        _expand_for_lines(new_lines)
        return expanded_first_changed, expanded_last_changed

    def _delete_changed_kitty_images(self, first_changed: int, last_changed: int) -> str:
        if first_changed < 0 or last_changed < first_changed:
            return ""

        ids: set[int] = set()
        max_line = min(last_changed, len(self._previous_lines) - 1)
        for i in range(first_changed, max_line + 1):
            ids.update(_extract_kitty_image_ids(self._previous_lines[i] if i < len(self._previous_lines) else ""))

        return self._delete_kitty_images(ids)

    def _log_debug_redraw(self, reason: str, new_lines_len: int, height: int) -> None:
        if os.environ.get("PI_DEBUG_REDRAW") != "1":
            return
        log_path = Path(self.log_directory) / "pi-debug.log"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        msg = (
            f"[{timestamp}] fullRender: {reason} "
            f"(prev={len(self._previous_lines)}, new={new_lines_len}, height={height})\n"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg)

    def do_render(self) -> None:
        if self.stopped:
            return
        width = self.terminal.columns
        height = self.terminal.rows
        width_changed = self._previous_width != 0 and self._previous_width != width
        height_changed = self._previous_height != 0 and self._previous_height != height
        previous_buffer_length = (
            self._previous_viewport_top + self._previous_height if self._previous_height > 0 else height
        )
        prev_viewport_top = max(0, previous_buffer_length - height) if height_changed else self._previous_viewport_top
        viewport_top = prev_viewport_top
        hardware_cursor_row = self._hardware_cursor_row

        def _compute_line_diff(target_row: int) -> int:
            current_screen_row = hardware_cursor_row - prev_viewport_top
            target_screen_row = target_row - viewport_top
            return target_screen_row - current_screen_row

        new_lines = self.render(width)

        if self.has_overlay_entries:
            new_lines = self.composite_overlays(new_lines, width, height)

        cursor_pos = self.extract_cursor_position(new_lines, height)

        new_lines = self.apply_line_resets(new_lines)

        def _full_render(clear: bool) -> None:
            nonlocal new_lines
            self.full_redraw_count += 1
            buffer = "\x1b[?2026h"
            if clear:
                buffer += self._delete_kitty_images(self._previous_kitty_image_ids)
                buffer += "\x1b[2J\x1b[H\x1b[3J"
            i = 0
            first = True
            while i < len(new_lines):
                if not first:
                    buffer += "\r\n"
                first = False
                line = new_lines[i]
                is_image = is_image_line(line)
                image_reserved_rows = self._get_kitty_image_reserved_rows(new_lines, i) if is_image else 1
                if image_reserved_rows > 1 and image_reserved_rows <= height:
                    buffer += "\r\n" * (image_reserved_rows - 1)
                    buffer += f"\x1b[{image_reserved_rows - 1}A"
                    buffer += line
                    buffer += f"\x1b[{image_reserved_rows - 1}B"
                    i += image_reserved_rows
                    continue
                buffer += line
                i += 1
            buffer += "\x1b[?2026l"
            self.terminal.write(buffer)
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            if clear:
                self._max_lines_rendered = len(new_lines)
            else:
                self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
            buffer_length = max(height, len(new_lines))
            self._previous_viewport_top = max(0, buffer_length - height)
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height

        # First render - just output everything without clearing (assumes clean screen)
        if len(self._previous_lines) == 0 and not width_changed and not height_changed:
            self._log_debug_redraw("first render", len(new_lines), height)
            _full_render(False)
            return

        # Width changes always need a full re-render because wrapping changes.
        if width_changed:
            self._log_debug_redraw(
                f"terminal width changed ({self._previous_width} -> {width})", len(new_lines), height
            )
            _full_render(True)
            return

        # Height changes normally need a full re-render to keep the visible viewport
        # aligned, but Termux changes height when the software keyboard shows/hides.
        if height_changed and not _is_termux_session():
            self._log_debug_redraw(
                f"terminal height changed ({self._previous_height} -> {height})", len(new_lines), height
            )
            _full_render(True)
            return

        # Content shrunk below the working area and no overlays - re-render to clear
        # empty rows (overlays need the padding, so only do this when no overlays active).
        if self.get_clear_on_shrink() and len(new_lines) < self._max_lines_rendered and not self.has_overlay_entries:
            self._log_debug_redraw(
                f"clearOnShrink (maxLinesRendered={self._max_lines_rendered})", len(new_lines), height
            )
            _full_render(True)
            return

        # Find first and last changed lines
        first_changed = -1
        last_changed = -1
        max_lines = max(len(new_lines), len(self._previous_lines))
        for i in range(max_lines):
            old_line = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""
            if old_line != new_line:
                if first_changed == -1:
                    first_changed = i
                last_changed = i

        appended_lines = len(new_lines) > len(self._previous_lines)
        if appended_lines:
            if first_changed == -1:
                first_changed = len(self._previous_lines)
            last_changed = len(new_lines) - 1

        if first_changed != -1:
            first_changed, last_changed = self._expand_changed_range_for_kitty_images(
                first_changed, last_changed, new_lines
            )
        append_start = appended_lines and first_changed == len(self._previous_lines) and first_changed > 0

        # No changes - but still need to update hardware cursor position if it moved
        if first_changed == -1:
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_viewport_top = prev_viewport_top
            self._previous_height = height
            return

        # All changes are in deleted lines (nothing to render, just clear)
        if first_changed >= len(new_lines):
            if len(self._previous_lines) > len(new_lines):
                buffer = "\x1b[?2026h"
                buffer += self._delete_changed_kitty_images(first_changed, last_changed)
                target_row = max(0, len(new_lines) - 1)
                if target_row < prev_viewport_top:
                    self._log_debug_redraw(
                        f"deleted lines moved viewport up ({target_row} < {prev_viewport_top})",
                        len(new_lines),
                        height,
                    )
                    _full_render(True)
                    return
                line_diff = _compute_line_diff(target_row)
                if line_diff > 0:
                    buffer += f"\x1b[{line_diff}B"
                elif line_diff < 0:
                    buffer += f"\x1b[{-line_diff}A"
                buffer += "\r"
                extra_lines = len(self._previous_lines) - len(new_lines)
                if extra_lines > height:
                    self._log_debug_redraw(f"extraLines > height ({extra_lines} > {height})", len(new_lines), height)
                    _full_render(True)
                    return
                clear_start_offset = 0 if len(new_lines) == 0 else 1
                if extra_lines > 0 and clear_start_offset > 0:
                    buffer += f"\x1b[{clear_start_offset}B"
                for i in range(extra_lines):
                    buffer += "\r\x1b[2K"
                    if i < extra_lines - 1:
                        buffer += "\x1b[1B"
                move_back = max(0, extra_lines - 1 + clear_start_offset)
                if move_back > 0:
                    buffer += f"\x1b[{move_back}A"
                buffer += "\x1b[?2026l"
                self.terminal.write(buffer)
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height
            self._previous_viewport_top = prev_viewport_top
            return

        # Differential rendering can only touch what was actually visible.
        if first_changed < prev_viewport_top:
            self._log_debug_redraw(
                f"firstChanged < viewportTop ({first_changed} < {prev_viewport_top})", len(new_lines), height
            )
            _full_render(True)
            return

        buffer = "\x1b[?2026h"
        buffer += self._delete_changed_kitty_images(first_changed, last_changed)
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed
        if move_target_row > prev_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - prev_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                buffer += f"\x1b[{move_to_bottom}B"
            scroll = move_target_row - prev_viewport_bottom
            buffer += "\r\n" * scroll
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        line_diff = _compute_line_diff(move_target_row)
        if line_diff > 0:
            buffer += f"\x1b[{line_diff}B"
        elif line_diff < 0:
            buffer += f"\x1b[{-line_diff}A"

        buffer += "\r\n" if append_start else "\r"

        render_end = min(last_changed, len(new_lines) - 1)
        i = first_changed
        first = True
        while i <= render_end:
            if not first:
                buffer += "\r\n"
            first = False
            line = new_lines[i]
            is_image = is_image_line(line)
            image_reserved_rows = self._get_kitty_image_reserved_rows(new_lines, i, render_end) if is_image else 1
            if image_reserved_rows > 1:
                image_start_screen_row = i - viewport_top
                if image_start_screen_row < 0 or image_start_screen_row + image_reserved_rows > height:
                    self._log_debug_redraw(
                        f"kitty image pre-clear would scroll ({image_start_screen_row} + {image_reserved_rows} > {height})",
                        len(new_lines),
                        height,
                    )
                    _full_render(True)
                    return
                buffer += "\x1b[2K"
                buffer += "\r\n\x1b[2K" * (image_reserved_rows - 1)
                buffer += f"\x1b[{image_reserved_rows - 1}A"
                buffer += line
                buffer += f"\x1b[{image_reserved_rows - 1}B"
                i += image_reserved_rows
                continue

            buffer += "\x1b[2K"
            if not is_image and visible_width(line) > width:
                self._write_crash_log(i, width, new_lines)
                self.stop()
                crash_log_path = Path(self.log_directory) / "pi-crash.log"
                error_msg = "\n".join(
                    [
                        f"Rendered line {i} exceeds terminal width ({visible_width(line)} > {width}).",
                        "",
                        "This is likely caused by a custom TUI component not truncating its output.",
                        "Use visible_width() to measure and truncate_to_width() to truncate lines.",
                        "",
                        f"Debug log written to: {crash_log_path}",
                    ]
                )
                raise RuntimeError(error_msg)
            buffer += line
            i += 1

        final_cursor_row = render_end

        if len(self._previous_lines) > len(new_lines):
            if render_end < len(new_lines) - 1:
                move_down = len(new_lines) - 1 - render_end
                buffer += f"\x1b[{move_down}B"
                final_cursor_row = len(new_lines) - 1
            extra_lines = len(self._previous_lines) - len(new_lines)
            for _ in range(len(new_lines), len(self._previous_lines)):
                buffer += "\r\n\x1b[2K"
            buffer += f"\x1b[{extra_lines}A"

        buffer += "\x1b[?2026l"

        self.terminal.write(buffer)

        self._cursor_row = max(0, len(new_lines) - 1)
        self._hardware_cursor_row = final_cursor_row
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(prev_viewport_top, final_cursor_row - height + 1)

        self._position_hardware_cursor(cursor_pos, len(new_lines))

        self._previous_lines = new_lines
        self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
        self._previous_width = width
        self._previous_height = height

    def _write_crash_log(self, index: int, width: int, new_lines: list[str]) -> None:
        crash_log_path = Path(self.log_directory) / "pi-crash.log"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        lines = [
            f"Crash at {timestamp}",
            f"Terminal width: {width}",
            f"Line {index} visible width: {visible_width(new_lines[index])}",
            "",
            "=== All rendered lines ===",
            *(f"[{idx}] (w={visible_width(text)}) {text}" for idx, text in enumerate(new_lines)),
            "",
        ]
        crash_log_path.parent.mkdir(parents=True, exist_ok=True)
        crash_log_path.write_text("\n".join(lines), encoding="utf-8")

    def _position_hardware_cursor(self, cursor_pos: tuple[int, int] | None, total_lines: int) -> None:
        """Position the hardware cursor for the IME candidate window."""
        if cursor_pos is None or total_lines <= 0:
            self.terminal.hide_cursor()
            return

        target_row = max(0, min(cursor_pos[0], total_lines - 1))
        target_col = max(0, cursor_pos[1])

        row_delta = target_row - self._hardware_cursor_row
        buffer = ""
        if row_delta > 0:
            buffer += f"\x1b[{row_delta}B"
        elif row_delta < 0:
            buffer += f"\x1b[{-row_delta}A"
        buffer += f"\x1b[{target_col + 1}G"

        if buffer:
            self.terminal.write(buffer)

        self._hardware_cursor_row = target_row
        if self.get_show_hardware_cursor():
            self.terminal.show_cursor()
        else:
            self.terminal.hide_cursor()
