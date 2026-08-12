"""Empty-line spacer component.

Ported from ``packages/tui/src/components/spacer.ts``.
"""

from __future__ import annotations

from ..component import Component


class Spacer(Component):
    """Renders a fixed number of empty lines."""

    def __init__(self, lines: int = 1) -> None:
        self.lines = lines

    def set_lines(self, lines: int) -> None:
        self.lines = lines

    def invalidate(self) -> None:
        return None

    def render(self, _width: int) -> list[str]:
        return ["" for _ in range(self.lines)]
