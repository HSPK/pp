"""Single-line text component that truncates to the viewport width.

Ported from ``packages/tui/src/components/truncated-text.ts``.
"""

from __future__ import annotations

from ..component import Component
from ..utils import truncate_to_width, visible_width


class TruncatedText(Component):
    def __init__(self, text: str, padding_x: int = 0, padding_y: int = 0) -> None:
        self.text = text
        self.padding_x = padding_x
        self.padding_y = padding_y

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        result: list[str] = []
        empty_line = " " * width

        for _ in range(self.padding_y):
            result.append(empty_line)

        available_width = max(1, width - self.padding_x * 2)

        single_line_text = self.text
        newline_index = self.text.find("\n")
        if newline_index != -1:
            single_line_text = self.text[:newline_index]

        display_text = truncate_to_width(single_line_text, available_width)

        padding = " " * self.padding_x
        line_with_padding = padding + display_text + padding
        padding_needed = max(0, width - visible_width(line_with_padding))
        result.append(line_with_padding + " " * padding_needed)

        for _ in range(self.padding_y):
            result.append(empty_line)

        return result
