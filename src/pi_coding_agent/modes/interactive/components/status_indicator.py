"""Spinner-based status indicators shown above the editor.

Ported from ``packages/coding-agent/src/modes/interactive/components/status-indicator.ts``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pi_tui.component import Component
from pi_tui.components.loader import Loader, LoaderIndicatorOptions

from ..theme.theme import theme
from .countdown_timer import CountdownTimer
from .keybinding_hints import key_text

if TYPE_CHECKING:
    from pi_tui.tui import TuiBase

StatusIndicatorKind = Literal["working", "retry", "compaction", "branchSummary"]
CompactionStatusReason = Literal["manual", "threshold", "overflow"]


class StatusIndicator(Loader):
    def __init__(
        self,
        kind: StatusIndicatorKind,
        ui: TuiBase | None,
        spinner_color_fn: Callable[[str], str],
        message_color_fn: Callable[[str], str],
        message: str,
        indicator: LoaderIndicatorOptions | None = None,
    ) -> None:
        super().__init__(ui, spinner_color_fn, message_color_fn, message, indicator)
        self.kind = kind

    def dispose(self) -> None:
        self.stop()


class WorkingStatusIndicator(StatusIndicator):
    def __init__(self, ui: TuiBase | None, message: str, indicator: LoaderIndicatorOptions | None = None) -> None:
        super().__init__(
            "working",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            message,
            indicator,
        )


class RetryStatusIndicator(StatusIndicator):
    def __init__(self, ui: TuiBase | None, attempt: int, max_attempts: int, delay_ms: int) -> None:
        def retry_message(seconds: int) -> str:
            return f"Retrying ({attempt}/{max_attempts}) in {seconds}s... ({key_text('app.interrupt')} to cancel)"

        super().__init__(
            "retry",
            ui,
            lambda spinner: theme.fg("warning", spinner),
            lambda text: theme.fg("muted", text),
            retry_message(math.ceil(delay_ms / 1000)),
        )
        self.countdown: CountdownTimer | None = CountdownTimer(
            delay_ms,
            ui,
            lambda seconds: self.set_message(retry_message(seconds)),
            self._on_countdown_expired,
        )

    def _on_countdown_expired(self) -> None:
        self.countdown = None

    def dispose(self) -> None:
        if self.countdown is not None:
            self.countdown.dispose()
            self.countdown = None
        super().dispose()


class CompactionStatusIndicator(StatusIndicator):
    def __init__(self, ui: TuiBase | None, reason: CompactionStatusReason) -> None:
        cancel_hint = f"({key_text('app.interrupt')} to cancel)"
        if reason == "manual":
            label = f"Compacting context... {cancel_hint}"
        else:
            prefix = "Context overflow detected, " if reason == "overflow" else ""
            label = f"{prefix}Auto-compacting... {cancel_hint}"
        super().__init__(
            "compaction",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            label,
        )


class BranchSummaryStatusIndicator(StatusIndicator):
    def __init__(self, ui: TuiBase | None) -> None:
        super().__init__(
            "branchSummary",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            f"Summarizing branch... ({key_text('app.interrupt')} to cancel)",
        )


class IdleStatus(Component):
    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        empty_line = " " * width
        return [empty_line, empty_line]


__all__ = [
    "BranchSummaryStatusIndicator",
    "CompactionStatusIndicator",
    "CompactionStatusReason",
    "IdleStatus",
    "RetryStatusIndicator",
    "StatusIndicator",
    "StatusIndicatorKind",
    "WorkingStatusIndicator",
]
