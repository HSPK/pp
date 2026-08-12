"""Transient flash messages composited by the alternate-screen renderer.

Python port of `packages/tui/src/components/alt-screen-flash.ts`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from pi_tui.component import Component
from pi_tui.utils import truncate_to_width

_DEFAULT_DURATION_MS = 1000


@dataclass
class _FlashEntry:
    id: int
    message: str
    timer: asyncio.TimerHandle


class AltScreenFlashContainer(Component):
    """Renders a stack of transient reverse-video messages that auto-expire."""

    def __init__(self, request_render: Callable[[], None]) -> None:
        self._entries: list[_FlashEntry] = []
        self._next_id = 0
        self._request_render = request_render

    def flash(self, message: str, duration_ms: float = _DEFAULT_DURATION_MS) -> None:
        entry_id = self._next_id
        self._next_id += 1

        def _expire() -> None:
            index = next((i for i, e in enumerate(self._entries) if e.id == entry_id), -1)
            if index == -1:
                return
            self._entries.pop(index)
            self._request_render()

        loop = asyncio.get_event_loop()
        timer = loop.call_later(max(0.0, duration_ms) / 1000, _expire)
        self._entries.append(_FlashEntry(id=entry_id, message=message, timer=timer))
        self._request_render()

    def dispose(self) -> None:
        for entry in self._entries:
            entry.timer.cancel()
        self._entries.clear()

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        lines = []
        for entry in self._entries:
            message = truncate_to_width(f" {entry.message} ", width, "")
            lines.append(f"\x1b[7m{message}\x1b[27m")
        return lines
