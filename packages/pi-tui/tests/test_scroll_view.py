"""Tests for the scroll view port.

Python port of direct `ScrollView` behaviour from `packages/tui/test/layout.test.ts`.
"""

from __future__ import annotations

import time
import unicodedata

import pytest
from pi_tui.component import Component
from pi_tui.components.scroll_view import ScrollView, ScrollViewOptions
from pi_tui.layout import render_layout_frame
from pi_tui.layout_node import get_layout_node


def _char_width(char: str) -> int:
    if char == "\t":
        return 3
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


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

    def set_text(self, text: str) -> None:
        self.text = text

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        if not self.text or self.text.strip() == "":
            return []
        lines: list[str] = []
        for raw_line in self.text.replace("\t", "   ").split("\n"):
            for wrapped in _wrap_plain(raw_line, max(1, width)):
                padding = max(0, width - sum(_char_width(char) for char in wrapped))
                lines.append(wrapped + (" " * padding))
        return lines


class _Box(Component):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        return self.lines


def test_scroll_view_rejects_unsupported_axis() -> None:
    with pytest.raises(ValueError, match="Unsupported ScrollView axis: horizontal"):
        ScrollView(_Box([]), ScrollViewOptions(axis="horizontal"))  # type: ignore[arg-type]


def test_scroll_view_child_mutation_operations_raise() -> None:
    scroll_view = ScrollView(_Box(["x"]))

    with pytest.raises(RuntimeError, match="exactly one child"):
        scroll_view.add_child(_Box(["y"]))
    with pytest.raises(RuntimeError, match="cannot be removed"):
        scroll_view.remove_child(scroll_view.children[0])
    with pytest.raises(RuntimeError, match="cannot be cleared"):
        scroll_view.clear()


def test_scroll_view_tracks_follow_end_and_unused_delta() -> None:
    scroll_view = ScrollView(_Text("1\n2\n3\n4\n5\n6"), ScrollViewOptions(follow="end", primary=True))

    render_layout_frame(scroll_view, 10, 3, lambda: None)

    assert scroll_view.scroll_top == 3
    assert scroll_view.is_following_end is True
    assert scroll_view.scroll_by(-2) == 0
    assert scroll_view.scroll_top == 1
    assert scroll_view.is_following_end is False
    assert scroll_view.scroll_by(-3) == -2
    assert scroll_view.scroll_top == 0
    assert scroll_view.scroll_by(10) == 7
    assert scroll_view.scroll_top == 3
    assert scroll_view.is_following_end is True


def test_scroll_view_render_reserves_scrollbar_column_when_always_visible() -> None:
    scroll_view = ScrollView(_Text("123456"), ScrollViewOptions(scrollbar="always"))

    assert scroll_view.render(6) == ["12345 ", "6     "]


def test_scroll_view_auto_scrollbar_stays_visible_while_active_then_hides() -> None:
    scroll_view = ScrollView(_Text("1\n2\n3\n4\n5\n6"), ScrollViewOptions(scrollbar="auto", scrollbar_hide_delay_ms=10))
    callbacks: list[str] = []
    scroll_view.update_layout(6, 3, lambda: callbacks.append("render"))

    scroll_view.set_scrollbar_active(True)
    scroll_view.scroll_by(1)
    time.sleep(0.03)
    assert scroll_view.is_scrollbar_visible is True

    scroll_view.set_scrollbar_active(False)
    time.sleep(0.03)
    assert scroll_view.is_scrollbar_visible is False
    assert callbacks


def test_scroll_view_exposes_scroll_layout_node() -> None:
    child = _Box(["x"])
    scroll_view = ScrollView(child, ScrollViewOptions(primary=True))

    node = get_layout_node(scroll_view)

    assert node is not None
    assert node.type == "scroll"
    assert node.component is child
    assert node.state is scroll_view
