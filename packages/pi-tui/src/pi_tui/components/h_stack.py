"""Horizontal stack layout.

Ported from ``packages/tui/src/components/h-stack.ts``.
"""

from __future__ import annotations

import math
import sys
from typing import ClassVar, Literal

from ..layout_node import LayoutViewport
from ..tui import composite_tui_line
from ..utils import visible_width
from .stack import (
    Stack,
    StackChild,
    StackOptions,
    allocate_stack_sizes,
    visible_stack_entries,
)


class HStack(Stack):
    layout_type: ClassVar[Literal["vstack", "hstack"]] = "hstack"

    def __init__(self, children: list[StackChild] | None = None, options: StackOptions | None = None) -> None:
        super().__init__(children, options)

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        viewport = LayoutViewport(width=safe_width, height=sys.maxsize)
        entries = visible_stack_entries(self.entries, viewport)
        if len(entries) == 0:
            return []

        intrinsic_widths: list[int] = []
        for entry in entries:
            lines = entry.component.render(safe_width)
            intrinsic_widths.append(max((visible_width(line) for line in lines), default=0))

        widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, self.gap)
        rendered = [
            [] if widths[index] == 0 else entry.component.render(widths[index]) for index, entry in enumerate(entries)
        ]
        height = max((len(lines) for lines in rendered), default=0)
        result = [""] * height

        x = 0
        for index, lines in enumerate(rendered):
            child_width = widths[index]
            offset = 0
            if self.align == "center":
                offset = math.floor((height - len(lines)) / 2)
            elif self.align == "end":
                offset = height - len(lines)
            for row, line in enumerate(lines):
                target = row + offset
                if target < 0 or target >= len(result):
                    continue
                result[target] = composite_tui_line(result[target], line, x, child_width, safe_width)
            x += child_width + self.gap
        return result
