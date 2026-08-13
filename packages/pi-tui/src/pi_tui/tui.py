"""Main TUI base class: overlay stack, focus, differential-render scheduling.

Python port of `packages/tui/src/tui.ts`. `Component`, `Container`, `Focusable`,
`CURSOR_MARKER`, and `is_focusable` are already ported in `component.py` and are
reused (not redefined) here.

Rendering is scheduled with `asyncio` (`loop.call_soon`/`call_later`) instead
of `process.nextTick`/`setTimeout`; every public method that schedules a
render (`request_render`, `start`, `handle_terminal_input`, ...) therefore
requires a running event loop, matching the rest of this package.
"""

from __future__ import annotations

import asyncio
import os
import re
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pi_tui.component import CURSOR_MARKER, Component, Container, is_focusable
from pi_tui.keys import is_key_release, matches_key
from pi_tui.terminal import Terminal
from pi_tui.terminal_colors import (
    RgbColor,
    TerminalColorScheme,
    is_osc11_background_color_response,
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)
from pi_tui.terminal_image import (
    CellDimensions,
    get_capabilities,
    is_image_line,
    set_cell_dimensions,
)
from pi_tui.utils import (
    extract_segments,
    normalize_terminal_output,
    slice_by_column,
    slice_with_width,
    visible_width,
)

__all__ = [
    "CURSOR_MARKER",
    "Component",
    "Container",
    "OverlayAnchor",
    "OverlayHandle",
    "OverlayMargin",
    "OverlayOptions",
    "OverlayUnfocusOptions",
    "TuiBase",
    "TuiInputListener",
    "TuiInputListenerResult",
    "TuiMode",
    "TuiStopOptions",
    "composite_tui_line",
    "is_focusable",
    "is_viewport_tui",
    "visible_width",
]

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

TuiMode = Literal["regular", "fullscreen"]

OverlayAnchor = Literal[
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "top-center",
    "bottom-center",
    "left-center",
    "right-center",
]

# Absolute column/row count, or a percentage string like "50%".
SizeValue = int | str


@dataclass
class OverlayMargin:
    top: int | None = None
    right: int | None = None
    bottom: int | None = None
    left: int | None = None


@dataclass
class OverlayOptions:
    """Overlay positioning/sizing options. Values may be absolute or `"N%"`."""

    width: SizeValue | None = None
    min_width: int | None = None
    max_height: SizeValue | None = None
    anchor: OverlayAnchor | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    row: SizeValue | None = None
    col: SizeValue | None = None
    margin: OverlayMargin | int | None = None
    visible: Callable[[int, int], bool] | None = None
    non_capturing: bool = False


@dataclass
class OverlayUnfocusOptions:
    """Options for `OverlayHandle.unfocus`."""

    target: Component | None


@dataclass
class OverlayHandle:
    """Handle returned by `TuiBase.show_overlay` for controlling the overlay."""

    hide: Callable[[], None]
    set_hidden: Callable[[bool], None]
    is_hidden: Callable[[], bool]
    focus: Callable[[], None]
    unfocus: Callable[[OverlayUnfocusOptions | None], None]
    is_focused: Callable[[], bool]


@dataclass
class TuiStopOptions:
    """Leave renderer output in place for another TUI taking over the terminal."""

    preserve_screen: bool = False


TuiInputListener = Callable[[str], "TuiInputListenerResult | None"]


@dataclass
class TuiInputListenerResult:
    consume: bool | None = None
    data: str | None = None


# ---------------------------------------------------------------------------
# compositeTuiLine
# ---------------------------------------------------------------------------

_SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"


def composite_tui_line(
    base_line: str,
    overlay_line: str,
    start_col: int,
    overlay_width: int,
    total_width: int,
) -> str:
    """Composite overlay content into a terminal line at a fixed column."""
    if is_image_line(base_line):
        return base_line

    after_start = start_col + overlay_width
    before, before_width, after, after_width = extract_segments(
        base_line,
        start_col,
        after_start,
        total_width - after_start,
        True,
    )
    overlay_text, overlay_rendered_width = slice_with_width(overlay_line, 0, overlay_width, True)
    before_pad = max(0, start_col - before_width)
    overlay_pad = max(0, overlay_width - overlay_rendered_width)
    actual_before_width = max(start_col, before_width)
    actual_overlay_width = max(overlay_width, overlay_rendered_width)
    after_target = max(0, total_width - actual_before_width - actual_overlay_width)
    after_pad = max(0, after_target - after_width)
    result = (
        before
        + (" " * before_pad)
        + _SEGMENT_RESET
        + overlay_text
        + (" " * overlay_pad)
        + _SEGMENT_RESET
        + after
        + (" " * after_pad)
    )
    if visible_width(result) <= total_width:
        return result
    return slice_by_column(result, 0, total_width, True)


# ---------------------------------------------------------------------------
# Overlay stack internals
# ---------------------------------------------------------------------------


@dataclass
class _OverlayStackEntry:
    component: Component
    options: OverlayOptions | None
    pre_focus: Component | None
    hidden: bool = False
    focus_order: int = 0


@dataclass
class _RestoreOverlayResume:
    status: Literal["restore-overlay"] = "restore-overlay"


@dataclass
class _FocusTargetResume:
    target: Component | None
    status: Literal["focus-target"] = "focus-target"


_OverlayBlockedFocusResume = _RestoreOverlayResume | _FocusTargetResume


@dataclass
class _InactiveFocusRestoreState:
    status: Literal["inactive"] = "inactive"


@dataclass
class _EligibleFocusRestoreState:
    overlay: _OverlayStackEntry
    status: Literal["eligible"] = "eligible"


@dataclass
class _BlockedFocusRestoreState:
    overlay: _OverlayStackEntry
    blocked_by: Component
    resume: _OverlayBlockedFocusResume
    status: Literal["blocked"] = "blocked"


_OverlayFocusRestoreState = _InactiveFocusRestoreState | _EligibleFocusRestoreState | _BlockedFocusRestoreState

_OverlayFocusRestorePolicy = Literal["clear", "preserve"]

_PERCENT_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)%$")


def _parse_size_value(value: SizeValue | None, reference_size: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = _PERCENT_PATTERN.match(value)
    if match:
        return int((reference_size * float(match.group(1))) / 100)
    return None


@dataclass
class _PendingOsc11BackgroundQuery:
    settled: bool = False
    future: asyncio.Future[RgbColor | None] | None = None
    timer: asyncio.TimerHandle | None = None


def is_viewport_tui(tui: TuiBase) -> bool:
    """Return whether `tui` is a viewport-style TUI (e.g. `TuiAltScreen`)."""
    return bool(getattr(tui, "VIEWPORT_TUI", False))


class TuiBase(Container):
    """Terminal-UI base class: differential-render scheduling, overlay stack, focus."""

    #: Overridden to `True` by viewport-style TUIs (e.g. `TuiAltScreen`).
    VIEWPORT_TUI = False

    MIN_RENDER_INTERVAL_S = 16 / 1000

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
    ) -> None:
        super().__init__()
        self.terminal = terminal
        self._focused_component: Component | None = None
        self._input_listeners: list[TuiInputListener] = []

        #: Global callback for the debug key (Shift+Ctrl+D). Called before
        #: input is forwarded to the focused component.
        self.on_debug: Callable[[], None] | None = None

        self._render_requested = False
        self._immediate_render_scheduled = False
        #: Event loop captured at `start()`. Used to bounce `request_render`
        #: calls made from worker threads back onto the loop.
        self._render_loop: asyncio.AbstractEventLoop | None = None
        self._render_timer: asyncio.TimerHandle | None = None
        self._last_render_at = 0.0
        self._show_hardware_cursor = os.environ.get("PI_HARDWARE_CURSOR") == "1"
        self._clear_on_shrink = os.environ.get("PI_CLEAR_ON_SHRINK") == "1"
        self.full_redraw_count = 0
        self.stopped = False
        self._pending_osc11_background_replies = 0
        self._pending_osc11_background_queries: list[_PendingOsc11BackgroundQuery] = []
        self._terminal_color_scheme_listeners: list[Callable[[TerminalColorScheme], None]] = []
        self._terminal_color_scheme_notifications_enabled = False
        self.log_directory = (
            log_directory or os.environ.get("PI_CODING_AGENT_DIR") or str(Path.home() / ".pi" / "agent")
        )

        self._focus_order_counter = 0
        self._overlay_stack: list[_OverlayStackEntry] = []
        self._overlay_focus_restore: _OverlayFocusRestoreState = _InactiveFocusRestoreState()

        if show_hardware_cursor is not None:
            self._show_hardware_cursor = show_hardware_cursor

    # -- Abstract / overridable hooks ----------------------------------

    @property
    @abstractmethod
    def mode(self) -> TuiMode: ...

    @abstractmethod
    def do_render(self) -> None: ...

    def reset_render_state(self) -> None:
        return None

    def before_terminal_start(self) -> None:
        return None

    def after_terminal_start(self) -> None:
        return None

    def before_terminal_stop(self, options: TuiStopOptions) -> None:
        return None

    def after_terminal_stop(self, options: TuiStopOptions) -> None:
        return None

    def get_mounted_roots(self) -> list[Component]:
        return self.children

    # -- Simple settings --------------------------------------------------

    @property
    def full_redraws(self) -> int:
        return self.full_redraw_count

    @property
    def has_overlay_entries(self) -> bool:
        return len(self._overlay_stack) > 0

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        if self._show_hardware_cursor == enabled:
            return
        self._show_hardware_cursor = enabled
        if not enabled:
            self.terminal.hide_cursor()
        self.request_render()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self._clear_on_shrink = enabled

    def get_focused_component(self) -> Component | None:
        return self._focused_component

    # -- Focus --------------------------------------------------------

    def set_focus(self, component: Component | None) -> None:
        self._set_focus_internal(component, "clear")

    def _set_focus_internal(
        self,
        component: Component | None,
        overlay_focus_restore: _OverlayFocusRestorePolicy,
    ) -> None:
        previous_focus = self._focused_component
        next_focus = component
        previous_focused_overlay = None
        if previous_focus is not None:
            previous_focused_overlay = next(
                (e for e in self._overlay_stack if e.component is previous_focus and self._is_overlay_visible(e)),
                None,
            )
        next_focus_is_overlay = next_focus is not None and any(e.component is next_focus for e in self._overlay_stack)
        restore_state = self._get_visible_overlay_focus_restore()

        if next_focus is not None and not next_focus_is_overlay:
            if isinstance(restore_state, _BlockedFocusRestoreState) and restore_state.blocked_by is previous_focus:
                if isinstance(restore_state.resume, _FocusTargetResume) or not self._is_component_mounted(
                    restore_state.blocked_by
                ):
                    next_focus = self._resolve_blocked_overlay_focus_resume(restore_state)
                else:
                    self._overlay_focus_restore = _BlockedFocusRestoreState(
                        overlay=restore_state.overlay,
                        blocked_by=next_focus,
                        resume=restore_state.resume,
                    )
            elif (
                previous_focused_overlay is not None
                and not isinstance(restore_state, _InactiveFocusRestoreState)
                and restore_state.overlay is previous_focused_overlay
                and not self._is_overlay_focus_ancestor(previous_focused_overlay, next_focus)
            ):
                self._overlay_focus_restore = _BlockedFocusRestoreState(
                    overlay=previous_focused_overlay,
                    blocked_by=next_focus,
                    resume=_RestoreOverlayResume(),
                )
        elif next_focus is None:
            if isinstance(restore_state, _BlockedFocusRestoreState) and restore_state.blocked_by is previous_focus:
                next_focus = self._resolve_blocked_overlay_focus_resume(restore_state)
            elif overlay_focus_restore == "clear":
                self._clear_overlay_focus_restore()

        if is_focusable(self._focused_component):
            self._focused_component.focused = False  # type: ignore[attr-defined]

        self._focused_component = next_focus

        if is_focusable(next_focus):
            next_focus.focused = True  # type: ignore[attr-defined]

        focused_overlay = None
        if next_focus is not None:
            focused_overlay = next(
                (e for e in self._overlay_stack if e.component is next_focus and self._is_overlay_visible(e)),
                None,
            )
        if focused_overlay is not None:
            self._overlay_focus_restore = _EligibleFocusRestoreState(overlay=focused_overlay)

    def _clear_overlay_focus_restore(self) -> None:
        self._overlay_focus_restore = _InactiveFocusRestoreState()

    def _clear_overlay_focus_restore_for(self, overlay: _OverlayStackEntry) -> None:
        state = self._overlay_focus_restore
        if not isinstance(state, _InactiveFocusRestoreState) and state.overlay is overlay:
            self._clear_overlay_focus_restore()

    def _resolve_blocked_overlay_focus_resume(self, restore_state: _BlockedFocusRestoreState) -> Component | None:
        if isinstance(restore_state.resume, _RestoreOverlayResume):
            return restore_state.overlay.component
        self._clear_overlay_focus_restore()
        return restore_state.resume.target

    def _get_visible_overlay_focus_restore(self) -> _OverlayFocusRestoreState:
        state = self._overlay_focus_restore
        if isinstance(state, _InactiveFocusRestoreState):
            return state
        if state.overlay not in self._overlay_stack or not self._is_overlay_visible(state.overlay):
            return _InactiveFocusRestoreState()
        return state

    def _is_overlay_focus_ancestor(self, entry: _OverlayStackEntry, component: Component) -> bool:
        visited: set[int] = set()
        current = entry.pre_focus
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if current is component:
                return True
            next_entry = next((o for o in self._overlay_stack if o.component is current), None)
            current = next_entry.pre_focus if next_entry is not None else None
        return False

    def _retarget_overlay_pre_focus(self, removed: _OverlayStackEntry) -> None:
        for overlay in self._overlay_stack:
            if overlay is not removed and overlay.pre_focus is removed.component:
                overlay.pre_focus = removed.pre_focus

    def _is_component_mounted(self, component: Component) -> bool:
        return any(self._contains_component(child, component) for child in self.get_mounted_roots())

    def _contains_component(self, root: Component, target: Component) -> bool:
        if root is target:
            return True
        if not isinstance(root, Container):
            return False
        return any(self._contains_component(child, target) for child in root.children)

    # -- Overlay stack -----------------------------------------------

    def show_overlay(self, component: Component, options: OverlayOptions | None = None) -> OverlayHandle:
        entry = _OverlayStackEntry(
            component=component,
            options=options,
            pre_focus=self._focused_component,
            focus_order=self._next_focus_order(),
        )
        self._overlay_stack.append(entry)
        if not (options is not None and options.non_capturing) and self._is_overlay_visible(entry):
            self.set_focus(component)
        self.terminal.hide_cursor()
        self.request_render()

        def _hide() -> None:
            if entry not in self._overlay_stack:
                return
            self._clear_overlay_focus_restore_for(entry)
            self._retarget_overlay_pre_focus(entry)
            self._overlay_stack.remove(entry)
            if self._focused_component is component:
                top_visible = self._get_topmost_visible_overlay()
                self.set_focus(top_visible.component if top_visible is not None else entry.pre_focus)
            if not self._overlay_stack:
                self.terminal.hide_cursor()
            self.request_render()

        def _set_hidden(hidden: bool) -> None:
            if entry.hidden == hidden:
                return
            entry.hidden = hidden
            if hidden:
                self._clear_overlay_focus_restore_for(entry)
                if self._focused_component is component:
                    top_visible = self._get_topmost_visible_overlay()
                    self.set_focus(top_visible.component if top_visible is not None else entry.pre_focus)
            else:
                if not (options is not None and options.non_capturing) and self._is_overlay_visible(entry):
                    entry.focus_order = self._next_focus_order()
                    self.set_focus(component)
            self.request_render()

        def _is_hidden() -> bool:
            return entry.hidden

        def _focus() -> None:
            if entry not in self._overlay_stack or not self._is_overlay_visible(entry):
                return
            entry.focus_order = self._next_focus_order()
            self.set_focus(component)
            self.request_render()

        def _unfocus(unfocus_options: OverlayUnfocusOptions | None = None) -> None:
            is_focused = self._focused_component is component
            restore_state = self._overlay_focus_restore
            has_pending_restore = (
                not isinstance(restore_state, _InactiveFocusRestoreState) and restore_state.overlay is entry
            )
            if not is_focused and not has_pending_restore:
                return
            if (
                isinstance(restore_state, _BlockedFocusRestoreState)
                and restore_state.overlay is entry
                and self._focused_component is restore_state.blocked_by
            ):
                if unfocus_options is not None:
                    self._overlay_focus_restore = _BlockedFocusRestoreState(
                        overlay=entry,
                        blocked_by=restore_state.blocked_by,
                        resume=_FocusTargetResume(target=unfocus_options.target),
                    )
                else:
                    self._clear_overlay_focus_restore()
                self.request_render()
                return
            self._clear_overlay_focus_restore_for(entry)
            if is_focused or unfocus_options is not None:
                top_visible = self._get_topmost_visible_overlay()
                fallback_target = (
                    top_visible.component if top_visible is not None and top_visible is not entry else entry.pre_focus
                )
                self.set_focus(unfocus_options.target if unfocus_options is not None else fallback_target)
            self.request_render()

        def _is_focused() -> bool:
            return self._focused_component is component

        return OverlayHandle(
            hide=_hide,
            set_hidden=_set_hidden,
            is_hidden=_is_hidden,
            focus=_focus,
            unfocus=_unfocus,
            is_focused=_is_focused,
        )

    def _next_focus_order(self) -> int:
        self._focus_order_counter += 1
        return self._focus_order_counter

    def hide_overlay(self) -> None:
        if not self._overlay_stack:
            return
        overlay = self._overlay_stack[-1]
        self._clear_overlay_focus_restore_for(overlay)
        self._retarget_overlay_pre_focus(overlay)
        self._overlay_stack.pop()
        if self._focused_component is overlay.component:
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible.component if top_visible is not None else overlay.pre_focus)
        if not self._overlay_stack:
            self.terminal.hide_cursor()
        self.request_render()

    def has_overlay(self) -> bool:
        return any(self._is_overlay_visible(o) for o in self._overlay_stack)

    def _is_overlay_focused(self) -> bool:
        """Whether focus sits on a visible overlay. Port of `isOverlayFocused`."""
        return any(
            entry.component is self._focused_component and self._is_overlay_visible(entry)
            for entry in self._overlay_stack
        )

    def _is_overlay_visible(self, entry: _OverlayStackEntry) -> bool:
        if entry.hidden:
            return False
        if entry.options is not None and entry.options.visible is not None:
            return entry.options.visible(self.terminal.columns, self.terminal.rows)
        return True

    def _get_topmost_visible_overlay(self) -> _OverlayStackEntry | None:
        topmost: _OverlayStackEntry | None = None
        for overlay in self._overlay_stack:
            if (overlay.options is not None and overlay.options.non_capturing) or not self._is_overlay_visible(overlay):
                continue
            if topmost is None or overlay.focus_order > topmost.focus_order:
                topmost = overlay
        return topmost

    # -- Container overrides -------------------------------------------

    def invalidate(self) -> None:
        for root in self.get_mounted_roots():
            root.invalidate()
        for overlay in self._overlay_stack:
            overlay.component.invalidate()

    # -- Lifecycle ------------------------------------------------------

    def start(self) -> None:
        self.stopped = False
        try:
            self._render_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._render_loop = None
        self.before_terminal_start()
        self.terminal.start(self.handle_terminal_input, lambda: self.request_render())
        self.after_terminal_start()
        self.terminal.hide_cursor()
        if self._terminal_color_scheme_notifications_enabled:
            self.terminal.write("\x1b[?2031h")
        self._query_cell_size()
        self.request_render()

    def add_input_listener(self, listener: TuiInputListener) -> Callable[[], None]:
        self._input_listeners.append(listener)

        def _unregister() -> None:
            if listener in self._input_listeners:
                self._input_listeners.remove(listener)

        return _unregister

    def remove_input_listener(self, listener: TuiInputListener) -> None:
        if listener in self._input_listeners:
            self._input_listeners.remove(listener)

    def on_terminal_color_scheme_change(self, listener: Callable[[TerminalColorScheme], None]) -> Callable[[], None]:
        self._terminal_color_scheme_listeners.append(listener)

        def _unregister() -> None:
            if listener in self._terminal_color_scheme_listeners:
                self._terminal_color_scheme_listeners.remove(listener)

        return _unregister

    def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        if self._terminal_color_scheme_notifications_enabled == enabled:
            return
        self._terminal_color_scheme_notifications_enabled = enabled
        if not self.stopped:
            self.terminal.write("\x1b[?2031h" if enabled else "\x1b[?2031l")

    def _query_cell_size(self) -> None:
        # Cell size in pixels is only meaningful for image rendering, which is
        # out of scope for this port phase (see module docstring).
        if not get_capabilities().images:
            return
        self.terminal.write("\x1b[16t")

    def stop(self, options: TuiStopOptions | None = None) -> None:
        resolved_options = options or TuiStopOptions()
        self.stopped = True
        self._cancel_render_timer()
        if self._terminal_color_scheme_notifications_enabled:
            self.terminal.write("\x1b[?2031l")
        self.before_terminal_stop(resolved_options)
        self.terminal.show_cursor()
        self.terminal.stop()
        self.after_terminal_stop(resolved_options)

    # -- Render scheduling ------------------------------------------------

    def render_now(self, force: bool = False) -> None:
        if force:
            self.reset_render_state()
        self._render_requested = False
        self._cancel_render_timer()
        self._last_render_at = asyncio.get_running_loop().time()
        self.do_render()

    def request_render(self, force: bool = False) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Called from a worker thread — e.g. the `ScrollView` scrollbar
            # auto-hide timer, which is a `threading.Timer` here but a
            # `setTimeout` (always on the event loop) in TypeScript. Bounce the
            # request back onto the loop so the frame is actually drawn.
            loop = self._render_loop
            if loop is None or loop.is_closed() or self.stopped:
                return
            loop.call_soon_threadsafe(self.request_render, force)
            return
        if force:
            self.reset_render_state()
            self._request_immediate_render()
            return
        if self._render_requested:
            return
        self._render_requested = True
        asyncio.get_running_loop().call_soon(self._schedule_render)

    def _request_immediate_render(self) -> None:
        self._cancel_render_timer()
        self._render_requested = True
        if self._immediate_render_scheduled:
            return
        self._immediate_render_scheduled = True
        loop = asyncio.get_running_loop()

        def _run() -> None:
            self._immediate_render_scheduled = False
            if self.stopped or not self._render_requested:
                return
            # A previously queued _schedule_render() can create a timer before
            # this callback runs. User input must preempt that throttled frame.
            self._cancel_render_timer()
            self._render_requested = False
            self._last_render_at = loop.time()
            self.do_render()

        loop.call_soon(_run)

    def _cancel_render_timer(self) -> None:
        if self._render_timer is None:
            return
        self._render_timer.cancel()
        self._render_timer = None

    def _schedule_render(self) -> None:
        if self.stopped or self._render_timer is not None or not self._render_requested:
            return
        loop = asyncio.get_running_loop()
        elapsed = loop.time() - self._last_render_at
        delay = max(0.0, self.MIN_RENDER_INTERVAL_S - elapsed)

        def _fire() -> None:
            self._render_timer = None
            if self.stopped or not self._render_requested:
                return
            self._render_requested = False
            self._last_render_at = loop.time()
            self.do_render()
            if self._render_requested:
                self._schedule_render()

        self._render_timer = loop.call_later(delay, _fire)

    # -- Input handling ----------------------------------------------

    def handle_terminal_input(self, data: str) -> None:
        if self._consume_osc11_background_response(data):
            return
        if self._consume_terminal_color_scheme_report(data):
            return

        if self._input_listeners:
            current = data
            for listener in list(self._input_listeners):
                result = listener(current)
                if result is not None and result.consume:
                    return
                if result is not None and result.data is not None:
                    current = result.data
            if len(current) == 0:
                return
            data = current

        if self._consume_cell_size_response(data):
            return

        if matches_key(data, "shift+ctrl+d") and self.on_debug is not None:
            self.on_debug()
            return

        focused_overlay = next((o for o in self._overlay_stack if o.component is self._focused_component), None)
        if focused_overlay is not None and not self._is_overlay_visible(focused_overlay):
            top_visible = self._get_topmost_visible_overlay()
            if top_visible is not None:
                self.set_focus(top_visible.component)
            else:
                self._set_focus_internal(focused_overlay.pre_focus, "preserve")

        focus_is_overlay = any(o.component is self._focused_component for o in self._overlay_stack)
        if not focus_is_overlay:
            restore_state = self._get_visible_overlay_focus_restore()
            if isinstance(restore_state, _EligibleFocusRestoreState):
                self.set_focus(restore_state.overlay.component)
            elif (
                isinstance(restore_state, _BlockedFocusRestoreState)
                and restore_state.blocked_by is not self._focused_component
            ):
                if isinstance(restore_state.resume, _RestoreOverlayResume):
                    self.set_focus(restore_state.overlay.component)
                else:
                    self._clear_overlay_focus_restore()
                    self.set_focus(restore_state.resume.target)

        handle_input = getattr(self._focused_component, "handle_input", None)
        if handle_input is not None:
            wants_key_release = getattr(self._focused_component, "wants_key_release", False)
            if is_key_release(data) and not wants_key_release:
                return
            handle_input(data)
            # Keyboard input is latency-sensitive. Avoid the throttled timer
            # path, where even a zero-delay timer can take a full tick.
            self._request_immediate_render()

    def _consume_osc11_background_response(self, data: str) -> bool:
        if self._pending_osc11_background_replies <= 0:
            return False
        if not is_osc11_background_color_response(data):
            return False

        rgb = parse_osc11_background_color(data)
        self._pending_osc11_background_replies -= 1
        if self._pending_osc11_background_queries:
            query = self._pending_osc11_background_queries.pop(0)
            if not query.settled:
                query.settled = True
                if query.timer is not None:
                    query.timer.cancel()
                    query.timer = None
                if query.future is not None and not query.future.done():
                    query.future.set_result(rgb)
        return True

    def _consume_terminal_color_scheme_report(self, data: str) -> bool:
        scheme = parse_terminal_color_scheme_report(data)
        if scheme is None:
            return False
        for listener in list(self._terminal_color_scheme_listeners):
            listener(scheme)
        return True

    _CELL_SIZE_RESPONSE_PATTERN = re.compile(r"^\x1b\[6;(\d+);(\d+)t$")

    def _consume_cell_size_response(self, data: str) -> bool:
        match = self._CELL_SIZE_RESPONSE_PATTERN.match(data)
        if not match:
            return False
        height_px = int(match.group(1))
        width_px = int(match.group(2))
        if height_px <= 0 or width_px <= 0:
            return True
        set_cell_dimensions(CellDimensions(width_px=width_px, height_px=height_px))
        # Invalidate all components so images re-render with correct dimensions.
        self.invalidate()
        self.request_render()
        return True

    # -- Overlay layout ------------------------------------------------

    def _resolve_overlay_layout(
        self,
        options: OverlayOptions | None,
        overlay_height: int,
        term_width: int,
        term_height: int,
    ) -> tuple[int, int, int, int | None]:
        """Returns `(width, row, col, max_height)`."""
        opt = options or OverlayOptions()

        if isinstance(opt.margin, int):
            margin = OverlayMargin(top=opt.margin, right=opt.margin, bottom=opt.margin, left=opt.margin)
        else:
            margin = opt.margin or OverlayMargin()
        margin_top = max(0, margin.top or 0)
        margin_right = max(0, margin.right or 0)
        margin_bottom = max(0, margin.bottom or 0)
        margin_left = max(0, margin.left or 0)

        avail_width = max(1, term_width - margin_left - margin_right)
        avail_height = max(1, term_height - margin_top - margin_bottom)

        width = _parse_size_value(opt.width, term_width)
        if width is None:
            width = min(80, avail_width)
        if opt.min_width is not None:
            width = max(width, opt.min_width)
        width = max(1, min(width, avail_width))

        max_height = _parse_size_value(opt.max_height, term_height)
        if max_height is not None:
            max_height = max(1, min(max_height, avail_height))

        effective_height = min(overlay_height, max_height) if max_height is not None else overlay_height

        if opt.row is not None:
            if isinstance(opt.row, str):
                match = _PERCENT_PATTERN.match(opt.row)
                if match:
                    max_row = max(0, avail_height - effective_height)
                    percent = float(match.group(1)) / 100
                    row = margin_top + int(max_row * percent)
                else:
                    row = self._resolve_anchor_row("center", effective_height, avail_height, margin_top)
            else:
                row = opt.row
        else:
            anchor = opt.anchor or "center"
            row = self._resolve_anchor_row(anchor, effective_height, avail_height, margin_top)

        if opt.col is not None:
            if isinstance(opt.col, str):
                match = _PERCENT_PATTERN.match(opt.col)
                if match:
                    max_col = max(0, avail_width - width)
                    percent = float(match.group(1)) / 100
                    col = margin_left + int(max_col * percent)
                else:
                    col = self._resolve_anchor_col("center", width, avail_width, margin_left)
            else:
                col = opt.col
        else:
            anchor = opt.anchor or "center"
            col = self._resolve_anchor_col(anchor, width, avail_width, margin_left)

        if opt.offset_y is not None:
            row += opt.offset_y
        if opt.offset_x is not None:
            col += opt.offset_x

        row = max(margin_top, min(row, term_height - margin_bottom - effective_height))
        col = max(margin_left, min(col, term_width - margin_right - width))

        return width, row, col, max_height

    def _resolve_anchor_row(self, anchor: OverlayAnchor, height: int, avail_height: int, margin_top: int) -> int:
        if anchor in ("top-left", "top-center", "top-right"):
            return margin_top
        if anchor in ("bottom-left", "bottom-center", "bottom-right"):
            return margin_top + avail_height - height
        return margin_top + (avail_height - height) // 2

    def _resolve_anchor_col(self, anchor: OverlayAnchor, width: int, avail_width: int, margin_left: int) -> int:
        if anchor in ("top-left", "left-center", "bottom-left"):
            return margin_left
        if anchor in ("top-right", "right-center", "bottom-right"):
            return margin_left + avail_width - width
        return margin_left + (avail_width - width) // 2

    def composite_overlays(self, lines: list[str], term_width: int, term_height: int) -> list[str]:
        """Composite all overlays into content lines (sorted, higher `focus_order` = on top)."""
        if not self._overlay_stack:
            return lines
        result = list(lines)

        rendered: list[tuple[list[str], int, int, int]] = []
        min_lines_needed = len(result)

        visible_entries = [e for e in self._overlay_stack if self._is_overlay_visible(e)]
        visible_entries.sort(key=lambda e: e.focus_order)
        for entry in visible_entries:
            component, options = entry.component, entry.options

            width, _row_unused, _col_unused, max_height = self._resolve_overlay_layout(
                options, 0, term_width, term_height
            )

            overlay_lines = component.render(width)
            if max_height is not None and len(overlay_lines) > max_height:
                overlay_lines = overlay_lines[:max_height]

            _width_unused, row, col, _max_height_unused = self._resolve_overlay_layout(
                options, len(overlay_lines), term_width, term_height
            )

            rendered.append((overlay_lines, row, col, width))
            min_lines_needed = max(min_lines_needed, row + len(overlay_lines))

        working_height = max(len(result), term_height, min_lines_needed)
        while len(result) < working_height:
            result.append("")

        viewport_start = max(0, working_height - term_height)

        for overlay_lines, row, col, w in rendered:
            for i, overlay_line in enumerate(overlay_lines):
                idx = viewport_start + row + i
                if 0 <= idx < len(result):
                    truncated = (
                        overlay_line if visible_width(overlay_line) <= w else slice_by_column(overlay_line, 0, w, True)
                    )
                    result[idx] = self._composite_line_at(result[idx], truncated, col, w, term_width)

        return result

    def apply_line_resets(self, lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            if not is_image_line(line):
                lines[i] = normalize_terminal_output(line) + _SEGMENT_RESET
        return lines

    def _composite_line_at(
        self, base_line: str, overlay_line: str, start_col: int, overlay_width: int, total_width: int
    ) -> str:
        return composite_tui_line(base_line, overlay_line, start_col, overlay_width, total_width)

    def extract_cursor_position(self, lines: list[str], height: int) -> tuple[int, int] | None:
        """Find, strip, and return the `CURSOR_MARKER` position (row, col), searching only the visible viewport."""
        viewport_top = max(0, len(lines) - height)
        for row in range(len(lines) - 1, viewport_top - 1, -1):
            line = lines[row]
            marker_index = line.find(CURSOR_MARKER)
            if marker_index != -1:
                before_marker = line[:marker_index]
                col = visible_width(before_marker)
                lines[row] = line[:marker_index] + line[marker_index + len(CURSOR_MARKER) :]
                return row, col
        return None

    # -- Background-color / color-scheme queries -----------------------

    async def query_terminal_background_color(self, timeout_ms: float) -> RgbColor | None:
        """Query the terminal's default background color with OSC 11 (`ESC ] 11 ; ? BEL`)."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RgbColor | None] = loop.create_future()
        query = _PendingOsc11BackgroundQuery(future=future)

        def _on_timeout() -> None:
            if query.settled:
                return
            query.settled = True
            query.timer = None
            if not future.done():
                future.set_result(None)

        query.timer = loop.call_later(timeout_ms / 1000, _on_timeout)
        self._pending_osc11_background_queries.append(query)
        self._pending_osc11_background_replies += 1
        self.terminal.write("\x1b]11;?\x07")
        return await future

    async def query_terminal_color_scheme(self, timeout_ms: float) -> TerminalColorScheme | None:
        """Query the terminal's color-scheme preference with DSR (`CSI ? 996 n`)."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[TerminalColorScheme | None] = loop.create_future()
        state = {"settled": False, "timer": None, "unsubscribe": lambda: None}

        def _settle(scheme: TerminalColorScheme | None) -> None:
            if state["settled"]:
                return
            state["settled"] = True
            if state["timer"] is not None:
                state["timer"].cancel()
                state["timer"] = None
            state["unsubscribe"]()
            if not future.done():
                future.set_result(scheme)

        state["unsubscribe"] = self.on_terminal_color_scheme_change(_settle)
        state["timer"] = loop.call_later(timeout_ms / 1000, lambda: _settle(None))
        self.terminal.write("\x1b[?996n")
        return await future
