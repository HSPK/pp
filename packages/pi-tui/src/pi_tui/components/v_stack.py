"""Vertical stack layout.

Ported from ``packages/tui/src/components/v-stack.ts``.
"""

from __future__ import annotations

import sys
from typing import ClassVar, Literal

from ..layout_node import LayoutViewport
from .stack import (
    Stack,
    StackChild,
    StackOptions,
    allocate_stack_sizes,
    visible_stack_entries,
)


class VStack(Stack):
    layout_type: ClassVar[Literal["vstack", "hstack"]] = "vstack"

    def __init__(self, children: list[StackChild] | None = None, options: StackOptions | None = None) -> None:
        super().__init__(children, options)

    def render(self, width: int) -> list[str]:
        viewport = LayoutViewport(width=max(1, width), height=sys.maxsize)
        entries = visible_stack_entries(self.entries, viewport)
        rendered = [entry.component.render(viewport.width) for entry in entries]
        sizes = allocate_stack_sizes(entries, [len(lines) for lines in rendered], None, self.gap)

        lines: list[str] = []
        for index in range(len(entries)):
            if index > 0:
                lines.extend("" for _ in range(self.gap))
            child_lines = rendered[index][: sizes[index]]
            lines.extend(child_lines)
            lines.extend("" for _ in range(len(child_lines), sizes[index]))
        return lines
