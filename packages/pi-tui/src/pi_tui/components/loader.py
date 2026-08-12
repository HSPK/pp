"""Spinner/loader components.

Ported from ``packages/tui/src/components/loader.ts`` and
``packages/tui/src/components/cancellable-loader.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..keybindings import get_keybindings
from ..timers import IntervalHandle, schedule_interval
from .text import Text

if TYPE_CHECKING:
    from ..tui import TuiBase

DEFAULT_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
DEFAULT_INTERVAL_MS = 80


@dataclass
class LoaderIndicatorOptions:
    """Animation frames. Use an empty list to hide the indicator."""

    frames: list[str] | None = None
    interval_ms: int | None = None


class Loader(Text):
    """Text line with an optional spinning animation."""

    def __init__(
        self,
        ui: TuiBase | None,
        spinner_color_fn: Callable[[str], str],
        message_color_fn: Callable[[str], str],
        message: str = "Loading...",
        indicator: LoaderIndicatorOptions | None = None,
    ) -> None:
        super().__init__("", 1, 0)
        self.ui = ui
        self.spinner_color_fn = spinner_color_fn
        self.message_color_fn = message_color_fn
        self.message = message
        self._frames = list(DEFAULT_FRAMES)
        self._interval_ms = DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self._interval: IntervalHandle | None = None
        self._render_indicator_verbatim = False
        self.set_indicator(indicator)

    def render(self, width: int) -> list[str]:
        return ["", *super().render(width)]

    def start(self) -> None:
        self._update_display()
        self._restart_animation()

    def stop(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None

    def set_message(self, message: str) -> None:
        self.message = message
        self._update_display()

    def set_indicator(self, indicator: LoaderIndicatorOptions | None = None) -> None:
        self._render_indicator_verbatim = indicator is not None
        frames = None if indicator is None else indicator.frames
        self._frames = list(DEFAULT_FRAMES) if frames is None else list(frames)
        interval_ms = None if indicator is None else indicator.interval_ms
        self._interval_ms = interval_ms if interval_ms and interval_ms > 0 else DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self.start()

    def _restart_animation(self) -> None:
        self.stop()
        if len(self._frames) <= 1:
            return
        self._interval = schedule_interval(self._advance_frame, self._interval_ms / 1000)

    def _advance_frame(self) -> None:
        self._current_frame = (self._current_frame + 1) % len(self._frames)
        self._update_display()

    def _update_display(self) -> None:
        frame = self._frames[self._current_frame] if self._current_frame < len(self._frames) else ""
        rendered_frame = frame if self._render_indicator_verbatim else self.spinner_color_fn(frame)
        indicator = f"{rendered_frame} " if len(frame) > 0 else ""
        self.set_text(f"{indicator}{self.message_color_fn(self.message)}")
        if self.ui is not None:
            self.ui.request_render()


class _AbortController:
    """Minimal stand-in for the DOM ``AbortController`` used by the TS source."""

    def __init__(self) -> None:
        self.aborted = False
        self._listeners: list[Callable[[], None]] = []

    def abort(self) -> None:
        if self.aborted:
            return
        self.aborted = True
        for listener in list(self._listeners):
            listener()

    def add_listener(self, listener: Callable[[], None]) -> None:
        if self.aborted:
            listener()
            return
        self._listeners.append(listener)


class CancellableLoader(Loader):
    """Loader that can be cancelled with Escape.

    ``on_abort`` is called when the user presses the cancel key, and
    ``signal.aborted`` flips so in-flight async work can bail out.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._abort_controller = _AbortController()
        self.on_abort: Callable[[], None] | None = None

    @property
    def signal(self) -> _AbortController:
        return self._abort_controller

    @property
    def aborted(self) -> bool:
        return self._abort_controller.aborted

    def handle_input(self, data: str) -> None:
        if get_keybindings().matches(data, "tui.select.cancel"):
            self._abort_controller.abort()
            if self.on_abort is not None:
                self.on_abort()

    def dispose(self) -> None:
        self.stop()
