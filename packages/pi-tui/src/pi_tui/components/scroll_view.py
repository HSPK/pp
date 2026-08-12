"""Scroll view component for `packages/tui/src/components/scroll-view.ts`."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pi_tui.component import Component, Container
from pi_tui.layout_node import ScrollLayoutNode

ScrollViewScrollbar: TypeAlias = Literal["hidden", "auto", "always"]


@dataclass
class ScrollViewOptions:
    axis: Literal["vertical"] | None = None
    follow: Literal["none", "end"] | None = None
    primary: bool | None = None
    overscroll: Literal["chain", "contain"] | None = None
    scrollbar: ScrollViewScrollbar | None = None
    scrollbar_style: Callable[[str], str] | None = None
    scrollbar_hide_delay_ms: int | None = None


class ScrollView(Container):
    def __init__(self, component: Component, options: ScrollViewOptions | None = None) -> None:
        super().__init__()
        options = options or ScrollViewOptions()
        if options.axis is not None and options.axis != "vertical":
            raise ValueError(f"Unsupported ScrollView axis: {options.axis}")

        self._child = component
        self.children.append(component)
        self._follow_end = (options.follow or "none") == "end"
        self._following_end = self._follow_end
        self.primary = options.primary if options.primary is not None else False
        self.overscroll = options.overscroll or "chain"
        self._current_scrollbar = options.scrollbar or "hidden"
        self.scrollbar_style = options.scrollbar_style or (lambda text: f"\x1b[100m{text}\x1b[49m")
        hide_delay = 1000 if options.scrollbar_hide_delay_ms is None else options.scrollbar_hide_delay_ms
        self._scrollbar_hide_delay_ms = max(0, math.floor(hide_delay))
        self._current_scroll_top = 0
        self._content_height = 0
        self._current_viewport_height = 0
        self._request_render_callback: Callable[[], None] | None = None
        self._transient_scrollbar_visible = False
        self._scrollbar_active = False
        self._scrollbar_hide_timer: threading.Timer | None = None

    @property
    def scroll_top(self) -> int:
        return self._current_scroll_top

    @property
    def is_following_end(self) -> bool:
        return self._following_end

    @property
    def viewport_height(self) -> int:
        return self._current_viewport_height

    @property
    def scrollbar(self) -> ScrollViewScrollbar:
        return self._current_scrollbar

    @property
    def is_scrollbar_visible(self) -> bool:
        if self.scrollbar == "always":
            return self._current_viewport_height > 0
        return (
            self.scrollbar == "auto"
            and self._content_height > self._current_viewport_height
            and self._transient_scrollbar_visible
        )

    def set_scrollbar(self, scrollbar: ScrollViewScrollbar) -> None:
        if scrollbar == self._current_scrollbar:
            return
        self._current_scrollbar = scrollbar
        if scrollbar != "auto":
            self._hide_transient_scrollbar()
        elif self._scrollbar_active:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def get_content_width(self, width: int) -> int:
        return width - 1 if self.scrollbar == "always" and width > 1 else width

    def _mark_scrollbar_activity(self) -> None:
        if self.scrollbar != "auto" or self._content_height <= self._current_viewport_height:
            return
        self._transient_scrollbar_visible = True
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None
        if self._scrollbar_active:
            return

        def hide() -> None:
            self._scrollbar_hide_timer = None
            self._transient_scrollbar_visible = False
            if self._request_render_callback is not None:
                self._request_render_callback()

        timer = threading.Timer(self._scrollbar_hide_delay_ms / 1000, hide)
        timer.daemon = True
        timer.start()
        self._scrollbar_hide_timer = timer

    def _hide_transient_scrollbar(self) -> None:
        self._transient_scrollbar_visible = False
        if self._scrollbar_hide_timer is None:
            return
        self._scrollbar_hide_timer.cancel()
        self._scrollbar_hide_timer = None

    def set_scrollbar_active(self, active: bool) -> None:
        if active == self._scrollbar_active:
            return
        self._scrollbar_active = active
        self._mark_scrollbar_activity()

    def scroll_to(self, scroll_top: int) -> None:
        requested = (
            math.trunc(scroll_top)
            if isinstance(scroll_top, int | float) and math.isfinite(scroll_top)
            else self._current_scroll_top
        )
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        next_scroll_top = max(0, min(max_scroll_top, requested))
        if next_scroll_top == self._current_scroll_top:
            return
        self._current_scroll_top = next_scroll_top
        self._following_end = self._follow_end and next_scroll_top == max_scroll_top
        self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def scroll_by(self, lines: int) -> int:
        requested = math.trunc(lines) if isinstance(lines, int | float) and math.isfinite(lines) else 0
        if requested == 0:
            return 0
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        start = max_scroll_top if self._following_end else self._current_scroll_top
        next_scroll_top = max(0, min(max_scroll_top, start + requested))
        moved = next_scroll_top - start
        self._current_scroll_top = next_scroll_top
        self._following_end = self._follow_end and next_scroll_top == max_scroll_top
        if moved != 0:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()
        return requested - moved

    def scroll_to_start(self) -> None:
        changed = self._current_scroll_top != 0 or self._following_end != (
            self._follow_end and self._content_height <= self._current_viewport_height
        )
        self._current_scroll_top = 0
        self._following_end = self._follow_end and self._content_height <= self._current_viewport_height
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def scroll_to_end(self) -> None:
        next_scroll_top = max(0, self._content_height - self._current_viewport_height)
        changed = self._current_scroll_top != next_scroll_top or self._following_end != self._follow_end
        self._current_scroll_top = next_scroll_top
        self._following_end = self._follow_end
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None:
        self._content_height = max(0, math.floor(content_height))
        self._current_viewport_height = max(0, math.floor(viewport_height))
        self._request_render_callback = request_render
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        if self._following_end:
            self._current_scroll_top = max_scroll_top
        else:
            self._current_scroll_top = max(0, min(self._current_scroll_top, max_scroll_top))
        if self._follow_end and self._current_scroll_top == max_scroll_top:
            self._following_end = True
        if self._content_height <= self._current_viewport_height:
            self._hide_transient_scrollbar()

    def add_child(self, _component: Component) -> None:
        raise RuntimeError("ScrollView has exactly one child")

    def remove_child(self, _component: Component) -> None:
        raise RuntimeError("ScrollView child cannot be removed")

    def clear(self) -> None:
        raise RuntimeError("ScrollView child cannot be cleared")

    def render(self, width: int) -> list[str]:
        content_width = self.get_content_width(width)
        lines = self._child.render(content_width)
        if content_width == width:
            return lines
        return [f"{line} " for line in lines]

    def __pi_tui_layout_node__(self) -> ScrollLayoutNode:
        return ScrollLayoutNode(component=self._child, state=self)
