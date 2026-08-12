"""Tests for the stack layout port.

Python port of relevant `packages/tui/test/layout.test.ts` stack cases.
"""

from __future__ import annotations

import re
import unicodedata

from pi_tui.component import Component
from pi_tui.components.stack import (
    Stack,
    StackEntry,
    StackEntryOptions,
    StackOptions,
    allocate_stack_sizes,
    visible_stack_entries,
)
from pi_tui.layout_node import LayoutViewport, get_layout_node

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b_[^\x07\x1b]*(?:\x07|\x1b\\)")


def _char_width(char: str) -> int:
    if char == "\t":
        return 3
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _visible_width(text: str) -> int:
    width = 0
    stripped = _ANSI_RE.sub("", text)
    for char in stripped:
        width += _char_width(char)
    return width


def _wrap_plain(line: str, width: int) -> list[str]:
    if line == "":
        return [""]
    result: list[str] = []
    current = ""
    current_width = 0
    for char in line:
        char_width = _char_width(char)
        if current and current_width + char_width > width:
            result.append(current)
            current = ""
            current_width = 0
        current += char
        current_width += char_width
        if current_width >= width:
            result.append(current)
            current = ""
            current_width = 0
    if current or not result:
        result.append(current)
    return result


class _Text(Component):
    def __init__(self, text: str) -> None:
        self.text = text

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        if not self.text or self.text.strip() == "":
            return []
        content_width = max(1, width)
        lines: list[str] = []
        for raw_line in self.text.replace("\t", "   ").split("\n"):
            for wrapped in _wrap_plain(raw_line, content_width):
                padding = " " * max(0, content_width - _visible_width(wrapped))
                lines.append(wrapped + padding)
        return lines


class _VStack(Stack):
    layout_type = "vstack"

    def render(self, width: int) -> list[str]:
        viewport = LayoutViewport(width=max(1, width), height=2**31 - 1)
        entries = visible_stack_entries(self.entries, viewport)
        rendered = [entry.component.render(viewport.width) for entry in entries]
        sizes = allocate_stack_sizes(entries, [len(lines) for lines in rendered], None, self.gap)
        lines: list[str] = []
        for index, child_lines in enumerate(rendered):
            if index > 0:
                lines.extend("" for _ in range(self.gap))
            clipped = child_lines[: sizes[index]]
            lines.extend(clipped)
            lines.extend("" for _ in range(len(clipped), sizes[index]))
        return lines


class _Box(Component):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        return self.lines


def test_visible_stack_entries_filters_hidden_entries() -> None:
    visible = _Box(["visible"])
    hidden = _Box(["hidden"])
    entries = [
        StackEntry(component=visible),
        StackEntry(component=hidden, visible=lambda viewport: viewport.width > 10),
    ]

    result = visible_stack_entries(entries, LayoutViewport(width=5, height=4))

    assert [entry.component for entry in result] == [visible]


def test_allocate_stack_sizes_redistributes_growth_after_hitting_max_size() -> None:
    entries = [
        StackEntry(component=_Box([]), basis=1, grow=1, max_size=2),
        StackEntry(component=_Box([]), basis=1, grow=1),
    ]

    assert allocate_stack_sizes(entries, [1, 1], 6, 0) == [2, 4]


def test_allocate_stack_sizes_shrinks_to_minimums() -> None:
    entries = [
        StackEntry(component=_Box([]), shrink=1, min_size=1),
        StackEntry(component=_Box([]), shrink=0),
    ]

    assert allocate_stack_sizes(entries, [3, 3], 4, 0) == [1, 3]


def test_stack_tracks_entries_and_exposes_layout_node() -> None:
    first = _Box(["a"])
    second = _Box(["b"])
    stack = _VStack([first], StackOptions(gap=2, align="end"))
    stack.add_child(second, StackEntryOptions(grow=1, min_size=1))
    stack.remove_child(first)

    node = get_layout_node(stack)

    assert node is not None
    assert node.type == "vstack"
    assert node.gap == 2
    assert node.align == "end"
    assert [entry.component for entry in node.entries] == [second]

    stack.clear()
    assert stack.children == []
    assert stack.entries == []


def test_vstack_render_omits_gaps_around_invisible_entries() -> None:
    stack = _VStack(
        [
            _Text("one"),
            StackEntry(component=_Text("hidden"), visible=lambda viewport: False),
            _Text("two"),
        ],
        StackOptions(gap=1),
    )

    assert [line.rstrip() for line in stack.render(10)] == ["one", "", "two"]
