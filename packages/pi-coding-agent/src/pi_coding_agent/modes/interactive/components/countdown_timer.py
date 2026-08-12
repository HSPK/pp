"""Reusable countdown timer for dialog components.

Ported from ``packages/coding-agent/src/modes/interactive/components/countdown-timer.ts``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from pi_tui.timers import IntervalHandle, schedule_interval

if TYPE_CHECKING:
    from pi_tui.tui import TuiBase


class CountdownTimer:
    def __init__(
        self,
        timeout_ms: int,
        tui: TuiBase | None,
        on_tick: Callable[[int], None],
        on_expire: Callable[[], None],
    ) -> None:
        self._tui = tui
        self._on_tick = on_tick
        self._on_expire = on_expire
        self._remaining_seconds = math.ceil(timeout_ms / 1000)
        self._on_tick(self._remaining_seconds)
        self._interval: IntervalHandle | None = schedule_interval(self._tick, 1.0)

    @property
    def remaining_seconds(self) -> int:
        return self._remaining_seconds

    def _tick(self) -> None:
        self._remaining_seconds -= 1
        self._on_tick(self._remaining_seconds)
        if self._tui is not None:
            self._tui.request_render()

        if self._remaining_seconds <= 0:
            self.dispose()
            self._on_expire()

    def dispose(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None
