"""Alternate-screen TUI: scrollable viewport, mouse selection, scrollbar.

Python port of `packages/tui/src/tui-alt-screen.ts`.

Windows right-click-paste (`onRightClickPaste`, gated on `process.platform
=== "win32"` in the TypeScript source) is out of scope, matching this
package's stated Windows-console limitation; `sys.platform` is checked the
same way so the callback is simply never invoked on POSIX.
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from pi_tui.component import CURSOR_MARKER, Component
from pi_tui.components.alt_screen_flash import AltScreenFlashContainer
from pi_tui.components.scroll_view import ScrollView, ScrollViewOptions
from pi_tui.keybindings import get_keybindings
from pi_tui.keys import is_key_release
from pi_tui.layout import (
    LayoutFrame,
    ScrollbarGeometry,
    get_scroll_view_box,
    get_scroll_views_at,
    get_scrollbar_geometry,
    render_layout_frame,
)
from pi_tui.terminal import Terminal
from pi_tui.terminal_image import (
    ImageProtocol,
    TerminalCapabilities,
    delete_all_kitty_images,
    delete_all_kitty_placements,
    delete_kitty_image,
    get_capabilities,
    get_kitty_image_placement,
    is_image_line,
    set_capabilities,
)
from pi_tui.timers import IntervalHandle, schedule_interval
from pi_tui.tui import TuiBase, TuiInputListenerResult, TuiStopOptions, composite_tui_line
from pi_tui.utils import (
    extract_ansi_code,
    get_grapheme_cell_range,
    get_osc8_link_at_column,
    iter_word_segments,
    slice_by_column,
    strip_terminal_sequences,
    visible_width,
)

_ENTER_ALT_SCREEN = "\x1b[?1049h"
_EXIT_ALT_SCREEN = "\x1b[?1049l"
_DISABLE_AUTOWRAP = "\x1b[?7l"
_ENABLE_AUTOWRAP = "\x1b[?7h"
_ENABLE_BUTTON_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h"
_ENABLE_ALL_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h"
_DISABLE_MOUSE = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
_FOCUS_IN = "\x1b[I"
_FOCUS_OUT = "\x1b[O"
_BEGIN_SYNCHRONIZED_OUTPUT = "\x1b[?2026h"
_END_SYNCHRONIZED_OUTPUT = "\x1b[?2026l"
_OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")
_OSC133_PROMPT_START = re.compile(r"^\x1b\]133;A(?:\x07|\x1b\\)")
_PAGE_SCROLL_OVERLAP = 4
_MAX_CACHED_OFFSCREEN_KITTY_IMAGES = 16
_MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES = 32 * 1024 * 1024
_MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES = 64 * 1024 * 1024
_DOUBLE_CLICK_INTERVAL_MS = 500

_SGR_MOUSE_PATTERN = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])$")
_SGR_WHEEL_PATTERN = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)[Mm]$")
_MOUSE_SEQUENCE_PATTERN = re.compile(r"^\x1b\[<\d+;\d+;\d+[Mm]$")

SelectionGranularity = Literal["character", "word", "line"]


@dataclass
class _CachedKittyImage:
    transmission_generation: int
    transmission_bytes: int
    estimated_decoded_bytes: int


# ---------------------------------------------------------------------------
# Selection / mouse data types
# ---------------------------------------------------------------------------


@dataclass
class SelectionPoint:
    row: int
    col: int
    scroll_view: ScrollView | None = None
    #: Whether this point lies between terminal cells rather than on a cell.
    boundary: bool = False


@dataclass
class SelectionRange:
    start: SelectionPoint
    end: SelectionPoint


@dataclass
class _ClickTarget:
    timestamp: float
    count: int
    row: int
    scroll_view: ScrollView | None
    word_start: int
    word_end: int


@dataclass
class _SgrMouseEvent:
    button: int
    x: int
    y: int
    release: bool


@dataclass
class _WheelEvent:
    direction: Literal[-1, 1]
    x: int
    y: int


@dataclass
class _ScrollbarDrag:
    scroll_view: ScrollView
    grab_offset: int


@dataclass
class _ScrollbarTarget:
    scroll_view: ScrollView
    geometry: ScrollbarGeometry


@dataclass
class TuiAltScreenOptions:
    """Options for `TuiAltScreen`."""

    #: Number of logical lines moved for each mouse-wheel event.
    wheel_scroll_lines: int | None = None
    #: Capture mouse events for viewport scrolling and application-owned text selection.
    mouse: bool | None = None
    #: Open an OSC 8 hyperlink activated with a primary-button click.
    open_url: Callable[[str], None] | None = None
    #: Handle an unmodified secondary-button press for clipboard paste.
    #: Currently enabled on Windows only (out of scope for this port; see
    #: module docstring).
    on_right_click_paste: Callable[[], None] | None = None


class TuiAltScreen(TuiBase):
    """Alternate-screen TUI with a scrollable, application-owned viewport."""

    mode = "fullscreen"
    VIEWPORT_TUI = True

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
        options: TuiAltScreenOptions | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor, log_directory)
        options = options or TuiAltScreenOptions()

        self._previous_screen: list[str] = []
        self._last_document: list[str] = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._layout_root: Component | None = None
        self._current_layout: LayoutFrame | None = None

        self._implicit_document = _ImplicitDocument(self)
        self._implicit_scroll_view = ScrollView(self._implicit_document, ScrollViewOptions(follow="end", primary=True))
        self._flashes = AltScreenFlashContainer(lambda: self.request_render())
        self._alt_screen_active = False
        self._image_protocol: ImageProtocol = None
        self._saved_capabilities: TerminalCapabilities | None = None
        self._uploaded_kitty_images: dict[int, _CachedKittyImage] = {}

        self._selection_anchor: SelectionPoint | None = None
        self._selection_focus: SelectionPoint | None = None
        self._selection_granularity: SelectionGranularity = "character"
        self._selection_initial_range: SelectionRange | None = None
        self._last_click: _ClickTarget | None = None
        self._selection_drag_pointer: tuple[int, int] | None = None
        self._selection_auto_scroll_direction: Literal[-1, 0, 1] = 0
        self._selection_auto_scroll_timer: IntervalHandle | None = None
        self._selection_press_active = False
        self._scrollbar_drag: _ScrollbarDrag | None = None
        self._scrollbar_hover: ScrollView | None = None
        self._pressed_url: str | None = None
        self._selection_dragged = False

        self._wheel_scroll_lines = max(1, int(options.wheel_scroll_lines or 1))
        self._mouse_enabled = options.mouse if options.mouse is not None else True
        self._open_url = options.open_url
        self._on_right_click_paste = options.on_right_click_paste

        self.add_input_listener(self._handle_viewport_input)

    # -- Public API ------------------------------------------------------

    @property
    def viewport_top(self) -> int:
        return self._get_primary_scroll_view().scroll_top

    @property
    def is_following_output(self) -> bool:
        return self._get_primary_scroll_view().is_following_end

    def set_layout_root(self, component: Component | None) -> None:
        if self._layout_root is component:
            return
        self._layout_root = component
        self._current_layout = None
        self.request_render()

    def render(self, width: int) -> list[str]:
        if self._layout_root is not None:
            return self._layout_root.render(width)
        return super().render(width)

    def get_mounted_roots(self) -> list[Component]:
        return [self._layout_root] if self._layout_root is not None else self.children

    def _get_primary_scroll_view(self) -> ScrollView:
        if self._current_layout is not None and self._current_layout.primary_scroll_view is not None:
            return self._current_layout.primary_scroll_view
        return self._implicit_scroll_view

    # -- Lifecycle hooks --------------------------------------------------

    def before_terminal_start(self) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        self._alt_screen_active = True
        capabilities = get_capabilities()
        self._image_protocol = capabilities.images
        self._uploaded_kitty_images.clear()
        if capabilities.images == "iterm2":
            self._saved_capabilities = capabilities
            set_capabilities(replace(capabilities, images=None))
            self.invalidate()
        self._last_document = []
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self.reset_render_state()

        term = os.environ.get("TERM", "").lower()
        # Multiplexers can lag when every pointer movement is forwarded.
        # Button-motion tracking preserves clicks, wheel events, selections,
        # and scrollbar dragging.
        use_button_motion = (
            os.environ.get("TMUX") is not None
            or os.environ.get("ZELLIJ") is not None
            or os.environ.get("STY") is not None
            or term.startswith("tmux")
            or term.startswith("screen")
        )
        mouse_sequence = _ENABLE_BUTTON_MOTION_MOUSE if use_button_motion else _ENABLE_ALL_MOTION_MOUSE
        self.terminal.write(
            f"{_ENTER_ALT_SCREEN}{_DISABLE_AUTOWRAP}"
            f"{mouse_sequence if self._mouse_enabled else ''}\x1b[2J\x1b[H\x1b[?25l"
        )

    def before_terminal_stop(self, options: TuiStopOptions) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        if not self._alt_screen_active:
            return
        self.terminal.write(
            f"{_BEGIN_SYNCHRONIZED_OUTPUT}{self._delete_kitty_images()}"
            f"{_DISABLE_MOUSE if self._mouse_enabled else ''}{_ENABLE_AUTOWRAP}{_END_SYNCHRONIZED_OUTPUT}"
        )
        self._uploaded_kitty_images.clear()

    def after_terminal_stop(self, options: TuiStopOptions) -> None:
        if not self._alt_screen_active:
            return
        self._alt_screen_active = False
        if options.preserve_screen:
            self.terminal.write(f"{_BEGIN_SYNCHRONIZED_OUTPUT}{_EXIT_ALT_SCREEN}\x1b[?25h{_END_SYNCHRONIZED_OUTPUT}")
        else:
            width = max(1, self.terminal.columns)
            document_lines = [_OSC133_ZONE_PREFIX.sub("", line) for line in self.render(width)]
            reset_lines = self.apply_line_resets([line.replace(CURSOR_MARKER, "") for line in document_lines])
            self._last_document = [
                line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
                for line in reset_lines
            ]
            buffer = f"{_BEGIN_SYNCHRONIZED_OUTPUT}{_EXIT_ALT_SCREEN}{_DISABLE_AUTOWRAP}"
            for row, line in enumerate(self._last_document):
                if row > 0:
                    buffer += "\r\n"
                buffer += f"\r\x1b[2K{line}"
            buffer += f"\x1b[0m{_ENABLE_AUTOWRAP}\r\n\x1b[?25h{_END_SYNCHRONIZED_OUTPUT}"
            self.terminal.write(buffer)
        if self._saved_capabilities is not None:
            set_capabilities(self._saved_capabilities)
            self._saved_capabilities = None

    def _delete_kitty_images(self) -> str:
        return delete_all_kitty_images() if self._image_protocol == "kitty" else ""

    def _prepare_kitty_screen(self, screen: list[str]) -> tuple[list[str], str]:
        visible_image_ids: set[int] = set()
        lines: list[str] = []
        for line in screen:
            placement = get_kitty_image_placement(line)
            if placement is None:
                lines.append(line)
                continue
            visible_image_ids.add(placement.image_id)
            cached_image = self._uploaded_kitty_images.get(placement.image_id)
            next_cached_image = _CachedKittyImage(
                transmission_generation=placement.transmission_generation,
                transmission_bytes=placement.transmission_bytes,
                estimated_decoded_bytes=placement.estimated_decoded_bytes,
            )
            self._uploaded_kitty_images.pop(placement.image_id, None)
            self._uploaded_kitty_images[placement.image_id] = next_cached_image
            if cached_image is not None and cached_image.transmission_generation == placement.transmission_generation:
                lines.append(placement.replacement_line)
            else:
                lines.append(line)

        cached_offscreen_image_count = 0
        cached_offscreen_transmission_bytes = 0
        cached_offscreen_decoded_bytes = 0
        for image_id, cached_image in self._uploaded_kitty_images.items():
            if image_id in visible_image_ids:
                continue
            cached_offscreen_image_count += 1
            cached_offscreen_transmission_bytes += cached_image.transmission_bytes
            cached_offscreen_decoded_bytes += cached_image.estimated_decoded_bytes

        evicted_image_deletion = ""
        for image_id, cached_image in list(self._uploaded_kitty_images.items()):
            if (
                cached_offscreen_image_count <= _MAX_CACHED_OFFSCREEN_KITTY_IMAGES
                and cached_offscreen_transmission_bytes <= _MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES
                and cached_offscreen_decoded_bytes <= _MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES
            ):
                break
            if image_id in visible_image_ids:
                continue
            evicted_image_deletion += delete_kitty_image(image_id)
            del self._uploaded_kitty_images[image_id]
            cached_offscreen_image_count -= 1
            cached_offscreen_transmission_bytes -= cached_image.transmission_bytes
            cached_offscreen_decoded_bytes -= cached_image.estimated_decoded_bytes
        return lines, evicted_image_deletion

    def reset_render_state(self) -> None:
        self._previous_screen = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._current_layout = None

    # -- Scrolling ---------------------------------------------------

    def scroll_by(self, lines: int) -> None:
        self._get_primary_scroll_view().scroll_by(lines)
        self.request_render()

    def scroll_to_top(self) -> None:
        self._get_primary_scroll_view().scroll_to_start()
        self.request_render()

    def scroll_to_bottom(self) -> None:
        self._get_primary_scroll_view().scroll_to_end()
        self.request_render()

    def _scroll_to_prompt(self, direction: Literal[-1, 1]) -> None:
        if self._current_layout is None:
            return
        scroll_view = self._get_primary_scroll_view()
        box = get_scroll_view_box(self._current_layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if not lines:
            return

        row = scroll_view.scroll_top + direction
        while 0 <= row < len(lines):
            if _OSC133_PROMPT_START.match(lines[row] or ""):
                scroll_view.scroll_to(row)
                self.request_render()
                return
            row += direction

    def flash(self, message: str, duration_ms: float | None = None) -> None:
        """Show a transient message in the alternate-screen flash stack."""
        if duration_ms is None:
            self._flashes.flash(message)
        else:
            self._flashes.flash(message, duration_ms)

    # -- Input handling ----------------------------------------------

    def _handle_viewport_input(self, data: str) -> TuiInputListenerResult | None:
        if data == _FOCUS_OUT:
            had_active_selection = self._selection_press_active
            had_non_empty_active_selection = had_active_selection and self._get_selection_bounds() is not None
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            self._stop_scrollbar_hover()
            self._stop_scrollbar_drag()
            self._pressed_url = None
            self._selection_dragged = False
            if had_active_selection:
                self._selection_anchor = None
                self._selection_focus = None
                self._selection_granularity = "character"
                self._selection_initial_range = None
                if had_non_empty_active_selection:
                    self.request_render()
            self._last_click = None
            return TuiInputListenerResult(consume=True)
        if data == _FOCUS_IN:
            return TuiInputListenerResult(consume=True)

        wheel_event = self._parse_wheel_event(data)
        if wheel_event is not None:
            if self._should_defer_viewport_input_to_overlay():
                return None
            self._route_wheel(wheel_event)
            return TuiInputListenerResult(consume=True)
        mouse_event = self._parse_sgr_mouse_event(data)
        if mouse_event is not None:
            if self._handle_right_click_paste(mouse_event):
                return TuiInputListenerResult(consume=True)
            handled = self._handle_scrollbar_mouse_event(mouse_event)
            if self._scrollbar_drag is None:
                self._update_scrollbar_hover(mouse_event.x, mouse_event.y)
            if not handled:
                self._handle_selection_mouse_event(mouse_event)
            return TuiInputListenerResult(consume=True)
        if self._is_mouse_sequence(data):
            return TuiInputListenerResult(consume=True)

        keybindings = get_keybindings()
        if self._should_defer_viewport_input_to_overlay():
            return None
        is_release = is_key_release(data)
        if keybindings.matches(data, "tui.altScreen.pageUp"):
            if not is_release:
                self.scroll_by(-max(1, self._get_primary_scroll_view().viewport_height - _PAGE_SCROLL_OVERLAP))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.pageDown"):
            if not is_release:
                self.scroll_by(max(1, self._get_primary_scroll_view().viewport_height - _PAGE_SCROLL_OVERLAP))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.halfPageUp"):
            if not is_release:
                self.scroll_by(-max(1, self._get_primary_scroll_view().viewport_height // 2))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.halfPageDown"):
            if not is_release:
                self.scroll_by(max(1, self._get_primary_scroll_view().viewport_height // 2))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.lineUp"):
            if not is_release:
                self.scroll_by(-1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.lineDown"):
            if not is_release:
                self.scroll_by(1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.previousPrompt"):
            if not is_release:
                self._scroll_to_prompt(-1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.nextPrompt"):
            if not is_release:
                self._scroll_to_prompt(1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.top"):
            if not is_release:
                self.scroll_to_top()
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.bottom"):
            if not is_release:
                self.scroll_to_bottom()
            return TuiInputListenerResult(consume=True)
        return None

    def _should_defer_viewport_input_to_overlay(self) -> bool:
        """Let a focused overlay keep the wheel and the viewport keys.

        Port of `shouldDeferViewportInputToOverlay` (`tui-alt-screen.ts`).
        Without it, scrolling inside a focused overlay moved the transcript
        behind it instead. Upstream also excludes its own search overlay from
        this rule; that overlay is not ported here, so there is nothing to
        exclude.
        """
        return self._is_overlay_focused()

    def _parse_wheel_event(self, data: str) -> _WheelEvent | None:
        sgr = _SGR_WHEEL_PATTERN.match(data)
        if sgr:
            button = int(sgr.group(1))
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction not in (0, 1):
                return None
            return _WheelEvent(
                direction=-1 if direction == 0 else 1,
                x=int(sgr.group(2)) - 1,
                y=int(sgr.group(3)) - 1,
            )
        if len(data) == 6 and data.startswith("\x1b[M"):
            button = ord(data[3]) - 32
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction not in (0, 1):
                return None
            return _WheelEvent(
                direction=-1 if direction == 0 else 1,
                x=ord(data[4]) - 33,
                y=ord(data[5]) - 33,
            )
        return None

    def _route_wheel(self, event: _WheelEvent) -> None:
        remaining = event.direction * self._wheel_scroll_lines
        seen: set[ScrollView] = set()
        scroll_views = get_scroll_views_at(self._current_layout, event.x, event.y) if self._current_layout else []
        for scroll_view in scroll_views:
            seen.add(scroll_view)
            remaining = scroll_view.scroll_by(remaining)
            if remaining == 0 or scroll_view.overscroll == "contain":
                break
        primary = self._get_primary_scroll_view()
        if remaining != 0 and primary not in seen:
            primary.scroll_by(remaining)
        self._update_scrollbar_hover(event.x, event.y)
        self.request_render()

    def _parse_sgr_mouse_event(self, data: str) -> _SgrMouseEvent | None:
        match = _SGR_MOUSE_PATTERN.match(data)
        if not match:
            return None
        return _SgrMouseEvent(
            button=int(match.group(1)),
            x=int(match.group(2)) - 1,
            y=int(match.group(3)) - 1,
            release=match.group(4) == "m",
        )

    def _handle_right_click_paste(self, event: _SgrMouseEvent) -> bool:
        if self._on_right_click_paste is None or sys.platform != "win32" or event.release or event.button != 2:
            return False
        with contextlib.suppress(Exception):
            self._on_right_click_paste()
        return True

    # -- Scrollbar -----------------------------------------------------

    def _get_scrollbar_target_at(self, x: int, y: int) -> _ScrollbarTarget | None:
        if self.has_overlay() or self._current_layout is None:
            return None
        for scroll_view in get_scroll_views_at(self._current_layout, x, y):
            box = get_scroll_view_box(self._current_layout, scroll_view)
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if (
                geometry is not None
                and x == geometry.column
                and geometry.thumb_top <= y < geometry.thumb_top + geometry.thumb_height
            ):
                return _ScrollbarTarget(scroll_view=scroll_view, geometry=geometry)
        return None

    def _set_scrollbar_hover(self, scroll_view: ScrollView | None) -> None:
        if scroll_view is self._scrollbar_hover:
            return
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(False)
        self._scrollbar_hover = scroll_view
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(True)

    def _update_scrollbar_hover(self, x: int, y: int) -> None:
        target = self._get_scrollbar_target_at(x, y)
        self._set_scrollbar_hover(target.scroll_view if target is not None else None)

    def _stop_scrollbar_hover(self) -> None:
        self._set_scrollbar_hover(None)

    def _handle_scrollbar_mouse_event(self, event: _SgrMouseEvent) -> bool:
        if self._scrollbar_drag is not None:
            if event.release:
                self._stop_scrollbar_drag()
                return True
            box = (
                get_scroll_view_box(self._current_layout, self._scrollbar_drag.scroll_view)
                if self._current_layout is not None
                else None
            )
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if geometry is not None:
                max_thumb_offset = geometry.track_height - geometry.thumb_height
                thumb_offset = max(
                    0,
                    min(
                        max_thumb_offset,
                        event.y - geometry.track_top - self._scrollbar_drag.grab_offset,
                    ),
                )
                scroll_top = (
                    0 if max_thumb_offset == 0 else round((thumb_offset / max_thumb_offset) * geometry.max_scroll_top)
                )
                self._scrollbar_drag.scroll_view.scroll_to(scroll_top)
            return True

        if event.release or (event.button & 32) != 0 or (event.button & 3) != 0:
            return False
        target = self._get_scrollbar_target_at(event.x, event.y)
        if target is None:
            return False
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._set_scrollbar_hover(target.scroll_view)
        self._scrollbar_drag = _ScrollbarDrag(
            scroll_view=target.scroll_view,
            grab_offset=event.y - target.geometry.thumb_top,
        )
        return True

    def _stop_scrollbar_drag(self) -> None:
        self._scrollbar_drag = None

    # -- Selection -----------------------------------------------------

    def _get_scroll_selection_point(self, scroll_view: ScrollView, x: int, y: int) -> SelectionPoint | None:
        if self._current_layout is None:
            return None
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            return None
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        if visible_bottom < visible_top:
            return None
        pointer_row = max(visible_top, min(visible_bottom, y))
        max_content_row = max(0, len(box.scroll_content_lines or [""]) - 1)
        return SelectionPoint(
            row=max(0, min(max_content_row, scroll_view.scroll_top + pointer_row - box.rect.y)),
            col=max(0, min(box.rect.width - 1, x - box.rect.x)),
            scroll_view=scroll_view,
        )

    def _get_selection_point(self, event: _SgrMouseEvent, scroll_view: ScrollView | None = None) -> SelectionPoint:
        if scroll_view is not None:
            point = self._get_scroll_selection_point(scroll_view, event.x, event.y)
            if point is not None:
                return point
        return SelectionPoint(
            row=max(0, min(self.terminal.rows - 1, event.y)),
            col=max(0, min(self.terminal.columns - 1, event.x)),
        )

    def _get_selection_source_line(self, point: SelectionPoint) -> str:
        if point.scroll_view is not None and self._current_layout is not None:
            box = get_scroll_view_box(self._current_layout, point.scroll_view)
            if box is not None and box.scroll_content_lines is not None:
                lines = box.scroll_content_lines
                return lines[point.row] if 0 <= point.row < len(lines) else ""
        return self._previous_screen[point.row] if 0 <= point.row < len(self._previous_screen) else ""

    def _get_word_selection(self, point: SelectionPoint) -> SelectionRange | None:
        line = strip_terminal_sequences(self._get_selection_source_line(point))
        start = 0
        for segment in iter_word_segments(line):
            end = start + visible_width(segment.segment)
            if start <= point.col < end:
                return SelectionRange(
                    start=replace(point, col=start),
                    end=replace(point, col=end, boundary=True),
                )
            start = end
        return None

    def _get_line_selection(self, point: SelectionPoint) -> SelectionRange:
        return SelectionRange(
            start=replace(point, col=0),
            end=replace(point, col=visible_width(self._get_selection_source_line(point)), boundary=True),
        )

    def _update_selection_focus(self, point: SelectionPoint) -> None:
        if self._selection_granularity == "character" or self._selection_initial_range is None:
            self._selection_focus = point
            return
        range_ = (
            self._get_word_selection(point)
            if self._selection_granularity == "word"
            else self._get_line_selection(point)
        )
        if range_ is None:
            return
        initial = self._selection_initial_range
        target_before_initial = range_.start.row < initial.start.row or (
            range_.start.row == initial.start.row and range_.start.col < initial.start.col
        )
        if target_before_initial:
            self._selection_anchor = initial.end
            self._selection_focus = range_.start
        else:
            self._selection_anchor = initial.start
            self._selection_focus = range_.end

    def _get_click_count(self, point: SelectionPoint, word: SelectionRange | None) -> int:
        now = time.monotonic() * 1000
        previous = self._last_click
        count = (
            (previous.count % 3) + 1
            if (
                word is not None
                and previous is not None
                and now - previous.timestamp <= _DOUBLE_CLICK_INTERVAL_MS
                and previous.row == point.row
                and previous.scroll_view is point.scroll_view
                and previous.word_start == word.start.col
                and previous.word_end == word.end.col
            )
            else 1
        )
        self._last_click = (
            _ClickTarget(
                timestamp=now,
                count=count,
                row=point.row,
                scroll_view=point.scroll_view,
                word_start=word.start.col,
                word_end=word.end.col,
            )
            if word is not None
            else None
        )
        return count

    def _update_selection_auto_scroll(self, event: _SgrMouseEvent) -> None:
        scroll_view = self._selection_anchor.scroll_view if self._selection_anchor is not None else None
        if scroll_view is None or self._current_layout is None:
            self._stop_selection_auto_scroll()
            return
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            self._stop_selection_auto_scroll()
            return
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        self._selection_drag_pointer = (event.x, event.y)
        self._selection_auto_scroll_direction = (
            -1 if event.y <= visible_top else (1 if event.y >= visible_bottom else 0)
        )
        if self._selection_auto_scroll_direction == 0:
            self._stop_selection_auto_scroll()
            return
        if self._selection_auto_scroll_timer is not None:
            return
        self._selection_auto_scroll_timer = schedule_interval(self._auto_scroll_selection, 0.05)

    def _auto_scroll_selection(self) -> None:
        scroll_view = self._selection_anchor.scroll_view if self._selection_anchor is not None else None
        pointer = self._selection_drag_pointer
        direction = self._selection_auto_scroll_direction
        if scroll_view is None or pointer is None or direction == 0:
            self._stop_selection_auto_scroll()
            return
        remaining = scroll_view.scroll_by(direction)
        if remaining == direction:
            self._stop_selection_auto_scroll()
            return
        point = self._get_scroll_selection_point(scroll_view, pointer[0], pointer[1])
        if point is not None:
            self._update_selection_focus(point)
        self.request_render()

    def _stop_selection_auto_scroll(self) -> None:
        if self._selection_auto_scroll_timer is not None:
            self._selection_auto_scroll_timer.cancel()
            self._selection_auto_scroll_timer = None
        self._selection_auto_scroll_direction = 0
        self._selection_drag_pointer = None

    def _handle_selection_mouse_event(self, event: _SgrMouseEvent) -> None:
        if (event.button & 3) != 0:
            return
        anchor_scroll_view = self._selection_anchor.scroll_view if self._selection_anchor is not None else None
        point = self._get_selection_point(event, anchor_scroll_view)
        if event.release:
            if not self._selection_press_active:
                return
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            if self._selection_anchor is None:
                return
            self._update_selection_focus(point)
            clicked_url = (
                self._pressed_url
                if (
                    not self._selection_dragged
                    and self._selection_anchor.scroll_view is point.scroll_view
                    and self._selection_anchor.row == point.row
                    and self._selection_anchor.col == point.col
                )
                else None
            )
            self._pressed_url = None
            if clicked_url and self._open_url is not None:
                self._selection_anchor = None
                self._selection_focus = None
                # URL activation is best-effort.
                with contextlib.suppress(Exception):
                    self._open_url(clicked_url)
                self.request_render()
                return
            self._copy_selection_to_clipboard()
            self.request_render()
            return
        if (event.button & 32) != 0:
            if not self._selection_press_active or self._selection_anchor is None:
                return
            self._selection_dragged = True
            self._last_click = None
            self._pressed_url = None
            self._update_selection_focus(point)
            self._update_selection_auto_scroll(event)
            self.request_render()
            return
        self._stop_selection_auto_scroll()
        self._selection_press_active = True
        scroll_view = None
        if not self.has_overlay() and self._current_layout is not None:
            candidates = get_scroll_views_at(self._current_layout, event.x, event.y)
            scroll_view = candidates[0] if candidates else None
        anchor = self._get_selection_point(event, scroll_view)
        word = self._get_word_selection(anchor)
        click_count = self._get_click_count(anchor, word)
        range_ = word if click_count == 2 else (self._get_line_selection(anchor) if click_count == 3 else None)
        self._selection_granularity = ("word" if click_count == 2 else "line") if range_ is not None else "character"
        self._selection_initial_range = range_
        self._selection_anchor = range_.start if range_ is not None else anchor
        self._selection_focus = range_.end if range_ is not None else anchor
        self._selection_dragged = False
        self._pressed_url = (
            None
            if range_ is not None
            else get_osc8_link_at_column(
                self._previous_screen[max(0, min(self.terminal.rows - 1, event.y))]
                if max(0, min(self.terminal.rows - 1, event.y)) < len(self._previous_screen)
                else "",
                max(0, min(self.terminal.columns - 1, event.x)),
            )
        )
        self.request_render()

    def _get_selection_bounds(self) -> SelectionRange | None:
        if self._selection_anchor is None or self._selection_focus is None:
            return None
        if self._selection_anchor.scroll_view is not self._selection_focus.scroll_view:
            return None
        anchor_before_focus = self._selection_anchor.row < self._selection_focus.row or (
            self._selection_anchor.row == self._selection_focus.row
            and self._selection_anchor.col < self._selection_focus.col
        )
        if (
            self._selection_anchor.row == self._selection_focus.row
            and self._selection_anchor.col == self._selection_focus.col
        ):
            return None
        return (
            SelectionRange(start=self._selection_anchor, end=self._selection_focus)
            if anchor_before_focus
            else SelectionRange(start=self._selection_focus, end=self._selection_anchor)
        )

    def _get_selection_columns(
        self,
        line: str,
        row: int,
        selection: SelectionRange,
        min_column: int = 0,
        max_column: int | None = None,
    ) -> tuple[int, int]:
        line_width = visible_width(line)
        if max_column is None:
            max_column = line_width
        start = max(0, min_column)
        end = min(line_width, max_column)
        if row == selection.start.row:
            cell_range = get_grapheme_cell_range(line, selection.start.col)
            start = cell_range[0] if cell_range is not None else min(selection.start.col, line_width)
        if row == selection.end.row:
            if selection.end.boundary:
                end = min(selection.end.col, line_width)
            else:
                cell_range = get_grapheme_cell_range(line, selection.end.col)
                end = cell_range[1] if cell_range is not None else min(selection.end.col + 1, line_width)
        return max(min_column, start), min(max_column, end)

    def _copy_selection_to_clipboard(self) -> None:
        selection = self._get_selection_bounds()
        if selection is None:
            return
        source_lines: list[str] = self._previous_screen
        if selection.start.scroll_view is not None:
            if self._current_layout is None:
                return
            box = get_scroll_view_box(self._current_layout, selection.start.scroll_view)
            if box is None or box.scroll_content_lines is None:
                return
            source_lines = box.scroll_content_lines
        lines: list[str] = []
        for row in range(selection.start.row, selection.end.row + 1):
            line = source_lines[row] if 0 <= row < len(source_lines) else ""
            start, end = self._get_selection_columns(line, row, selection)
            lines.append(strip_terminal_sequences(slice_by_column(line, start, max(0, end - start), True)).rstrip())
        text = "\n".join(lines)
        if len(text) == 0:
            return
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.terminal.write(f"\x1b]52;c;{encoded}\x07")
        self.flash("Copied!")

    def _apply_selection_highlight(self, text: str) -> str:
        result = "\x1b[7m"
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if ansi is None:
                result += text[index]
                index += 1
                continue
            code, length = ansi
            result += code
            if code.endswith("m"):
                result += "\x1b[7m"
            index += length
        return f"{result}\x1b[27m"

    def _apply_selection(self, screen: list[str], layout: LayoutFrame | None = None) -> list[str]:
        if layout is None:
            layout = self._current_layout
        selection = self._get_selection_bounds()
        if selection is None:
            return screen
        screen_selection = selection
        min_row = 0
        max_row = len(screen) - 1
        min_column = 0
        max_column = self.terminal.columns
        if selection.start.scroll_view is not None:
            if layout is None:
                return screen
            box = get_scroll_view_box(layout, selection.start.scroll_view)
            if box is None:
                return screen
            min_row = max(0, box.rect.y, box.clip.y)
            max_row = min(len(screen) - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1)
            min_column = max(0, box.rect.x, box.clip.x)
            max_column = min(self.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width)
            screen_selection = SelectionRange(
                start=replace(
                    selection.start,
                    row=box.rect.y + selection.start.row - selection.start.scroll_view.scroll_top,
                    col=box.rect.x + selection.start.col,
                ),
                end=replace(
                    selection.end,
                    row=box.rect.y + selection.end.row - selection.start.scroll_view.scroll_top,
                    col=box.rect.x + selection.end.col,
                ),
            )

        result: list[str] = []
        for row, line in enumerate(screen):
            if (
                row < min_row
                or row > max_row
                or row < screen_selection.start.row
                or row > screen_selection.end.row
                or is_image_line(line)
            ):
                result.append(line)
                continue
            line_width = visible_width(line)
            start, end = self._get_selection_columns(line, row, screen_selection, min_column, max_column)
            if end <= start:
                result.append(line)
                continue
            before = slice_by_column(line, 0, start, True)
            selected = slice_by_column(line, start, end - start, True)
            after = slice_by_column(line, end, max(0, line_width - end), True)
            result.append(f"{before}{self._apply_selection_highlight(selected)}{after}")
        return result

    def _is_mouse_sequence(self, data: str) -> bool:
        return bool(_MOUSE_SEQUENCE_PATTERN.match(data)) or (len(data) == 6 and data.startswith("\x1b[M"))

    def _composite_flashes(self, screen: list[str], width: int, height: int) -> list[str]:
        flash_lines = self._flashes.render(width)[-height:] if height > 0 else []
        if not flash_lines:
            return screen
        result = list(screen)
        while len(result) < height:
            result.append("")
        for row, line in enumerate(flash_lines):
            flash_width = visible_width(line)
            if flash_width == 0:
                continue
            result[row] = composite_tui_line(
                result[row] if row < len(result) else "", line, width - flash_width, flash_width, width
            )
        return result

    # -- Rendering ------------------------------------------------------

    def do_render(self) -> None:
        if self.stopped or not self._alt_screen_active:
            return
        width = max(1, self.terminal.columns)
        height = max(1, self.terminal.rows)
        root = self._layout_root if self._layout_root is not None else self._implicit_scroll_view
        next_layout = render_layout_frame(root, width, height, lambda: self.request_render())
        screen = [_OSC133_ZONE_PREFIX.sub("", line) for line in next_layout.lines]
        screen = self.composite_overlays(screen, width, height)
        if len(screen) > height:
            screen = screen[len(screen) - height :]
        screen = self._apply_selection(screen, next_layout)
        screen = self._composite_flashes(screen, width, height)

        cursor_pos = self.extract_cursor_position(screen, height)
        screen = [
            line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
            for line in self.apply_line_resets(screen)
        ]

        full_redraw = (
            len(self._previous_screen) == 0
            or self._previous_screen_width != width
            or self._previous_screen_height != height
        )
        images_need_redraw = any(
            line != (self._previous_screen[row] if row < len(self._previous_screen) else None)
            and (
                is_image_line(line)
                or is_image_line(self._previous_screen[row] if row < len(self._previous_screen) else "")
            )
            for row, line in enumerate(screen)
        )
        redraw_images = full_redraw or images_need_redraw
        had_uploaded_kitty_images = len(self._uploaded_kitty_images) > 0
        if redraw_images and self._image_protocol == "kitty":
            prepared_lines, evicted_image_deletion = self._prepare_kitty_screen(screen)
        else:
            prepared_lines, evicted_image_deletion = screen, ""

        buffer = _BEGIN_SYNCHRONIZED_OUTPUT
        if full_redraw:
            self.full_redraw_count += 1
            clear_images = (
                delete_all_kitty_placements()
                if self._image_protocol == "kitty" and had_uploaded_kitty_images
                else self._delete_kitty_images()
            )
            buffer += f"{clear_images}\x1b[2J"
        elif images_need_redraw:
            if self._image_protocol == "iterm2":
                buffer += "\x1b[2J"
            elif self._image_protocol == "kitty":
                buffer += delete_all_kitty_placements()
        buffer += evicted_image_deletion

        for row in range(height):
            previous_line = self._previous_screen[row] if row < len(self._previous_screen) else None
            current_line = screen[row] if row < len(screen) else ""
            if not full_redraw and not images_need_redraw and current_line == previous_line:
                continue
            prepared_line = prepared_lines[row] if row < len(prepared_lines) else ""
            buffer += f"\x1b[{row + 1};1H\x1b[2K{prepared_line}"

        if cursor_pos is not None:
            cursor_row, cursor_col = cursor_pos
            buffer += f"\x1b[{cursor_row + 1};{min(width, cursor_col) + 1}H"
            buffer += "\x1b[?25h" if self.get_show_hardware_cursor() else "\x1b[?25l"
        else:
            buffer += "\x1b[?25l"
        buffer += _END_SYNCHRONIZED_OUTPUT
        self.terminal.write(buffer)

        self._previous_screen = screen
        self._previous_screen_width = width
        self._previous_screen_height = height
        self._current_layout = next_layout


class _ImplicitDocument(Component):
    """Renders `TuiAltScreen`'s own child components (its `Container.render`).

    Mirrors the TypeScript source's inline object literal that calls
    `super.render(width)` from within the constructor: `TuiAltScreen.render`
    is overridden to delegate to `layoutRoot` when present, so this wrapper
    must invoke `TuiBase.render` (i.e. `Container.render`) directly to avoid
    recursing back into `TuiAltScreen.render`.
    """

    def __init__(self, owner: TuiAltScreen) -> None:
        self._owner = owner

    def render(self, width: int) -> list[str]:
        return TuiBase.render(self._owner, width)

    def invalidate(self) -> None:
        for child in self._owner.children:
            child.invalidate()
