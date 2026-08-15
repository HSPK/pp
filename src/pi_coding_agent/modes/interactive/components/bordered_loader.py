"""Loader wrapped in borders, for extension UI.

Ported from ``packages/coding-agent/src/modes/interactive/components/bordered-loader.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pi_tui.component import Container
from pi_tui.components.loader import CancellableLoader, Loader
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class _InertSignal:
    """Stand-in for the ``AbortController`` a non-cancellable loader exposes."""

    aborted = False


class BorderedLoader(Container):
    def __init__(
        self,
        tui: Any,
        theme_instance: Any,
        message: str,
        cancellable: bool = True,
    ) -> None:
        super().__init__()
        self.cancellable = cancellable

        def border_color(text: str) -> str:
            return theme_instance.fg("border", text)

        self.add_child(DynamicBorder(border_color))

        loader_cls = CancellableLoader if cancellable else Loader
        self.loader: Loader = loader_cls(
            tui,
            lambda text: theme_instance.fg("accent", text),
            lambda text: theme_instance.fg("muted", text),
            message,
        )
        self._inert_signal = _InertSignal()
        self.add_child(self.loader)

        if cancellable:
            self.add_child(Spacer(1))
            self.add_child(Text(key_hint("tui.select.cancel", "cancel"), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(border_color))

    @property
    def signal(self) -> Any:
        if self.cancellable:
            return self.loader.signal  # type: ignore[attr-defined]
        return self._inert_signal

    @property
    def on_abort(self) -> Callable[[], None] | None:
        return getattr(self.loader, "on_abort", None)

    @on_abort.setter
    def on_abort(self, callback: Callable[[], None] | None) -> None:
        if self.cancellable:
            self.loader.on_abort = callback  # type: ignore[attr-defined]

    def handle_input(self, data: str) -> None:
        if self.cancellable:
            self.loader.handle_input(data)  # type: ignore[attr-defined]

    def dispose(self) -> None:
        dispose = getattr(self.loader, "dispose", None)
        if dispose is not None:
            dispose()
        else:
            self.loader.stop()


__all__ = ["BorderedLoader"]
