"""Full-width horizontal rule.

Ported from ``packages/coding-agent/src/modes/interactive/components/dynamic-border.ts``.
"""

from __future__ import annotations

from collections.abc import Callable

from pi_tui.component import Component

from ..theme.theme import theme


class DynamicBorder(Component):
    """Border that adjusts to the viewport width."""

    def __init__(self, color: Callable[[str], str] | None = None) -> None:
        self.color = color if color is not None else (lambda text: theme.fg("border", text))

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        return [self.color("─" * max(1, width))]
