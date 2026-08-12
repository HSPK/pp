"""Tests for the layout engine port.

Python port of relevant `packages/tui/test/layout.test.ts` cases.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Sequence

from pi_tui.component import Component
from pi_tui.components.scroll_view import ScrollView, ScrollViewOptions
from pi_tui.components.stack import Stack, StackEntry, StackOptions, allocate_stack_sizes, visible_stack_entries
from pi_tui.layout import (
    contains_point,
    get_scroll_view_box,
    get_scroll_views_at,
    get_scrollbar_geometry,
    render_layout_frame,
)
from pi_tui.layout_node import LayoutViewport
from pi_tui.terminal_image import (
    KittyImageMetadata,
    encode_kitty,
    register_kitty_image_metadata,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b_[^\x07\x1b]*(?:\x07|\x1b\\)")


def _strip_terminal_sequences(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _char_width(char: str) -> int:
    if char == "\t":
        return 3
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _visible_width(text: str) -> int:
    return sum(_char_width(char) for char in _strip_terminal_sequences(text))


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


def _composite_line(base: str, overlay: str, start_col: int, overlay_width: int, total_width: int) -> str:
    visible = [" "] * total_width
    for index, char in enumerate(_strip_terminal_sequences(base)):
        if index >= total_width:
            break
        visible[index] = char
    overlay_text = _strip_terminal_sequences(overlay)
    for index, char in enumerate(overlay_text[:overlay_width]):
        target = start_col + index
        if 0 <= target < total_width:
            visible[target] = char
    return "".join(visible).rstrip()


class _Text(Component):
    def __init__(self, text: str = "", bg_fn=None) -> None:
        self.text = text
        self.bg_fn = bg_fn

    def set_text(self, text: str) -> None:
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
                padded = wrapped + (" " * max(0, content_width - _visible_width(wrapped)))
                lines.append(self.bg_fn(padded) if self.bg_fn is not None else padded)
        return lines


class _Lines(Component):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def invalidate(self) -> None:
        return None

    def render(self, _width: int) -> list[str]:
        return self.lines


class _RenderSpy(Component):
    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = list(lines)
        self.render_count = 0

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        self.render_count += 1
        return self.lines


class _HugeLines(Sequence[str]):
    def __init__(self, length: int, values: dict[int, str]) -> None:
        self._length = length
        self._values = values

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return self._values.get(index, "")


class _HugeContent(Component):
    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = lines

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> Sequence[str]:
        return self.lines


class _VStack(Stack):
    layout_type = "vstack"

    def render(self, width: int) -> list[str]:
        viewport = LayoutViewport(width=max(1, width), height=2**31 - 1)
        entries = visible_stack_entries(self.entries, viewport)
        rendered = [list(entry.component.render(viewport.width)) for entry in entries]
        sizes = allocate_stack_sizes(entries, [len(lines) for lines in rendered], None, self.gap)
        lines: list[str] = []
        for index, child_lines in enumerate(rendered):
            if index > 0:
                lines.extend("" for _ in range(self.gap))
            clipped = child_lines[: sizes[index]]
            lines.extend(clipped)
            lines.extend("" for _ in range(len(clipped), sizes[index]))
        return lines


class _HStack(Stack):
    layout_type = "hstack"

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        viewport = LayoutViewport(width=safe_width, height=2**31 - 1)
        entries = visible_stack_entries(self.entries, viewport)
        if not entries:
            return []
        intrinsic_widths = []
        for entry in entries:
            intrinsic_widths.append(
                max((_visible_width(line) for line in entry.component.render(safe_width)), default=0)
            )
        widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, self.gap)
        rendered = [
            ([] if widths[index] == 0 else list(entry.component.render(widths[index])))
            for index, entry in enumerate(entries)
        ]
        height = max((len(lines) for lines in rendered), default=0)
        result = ["" for _ in range(height)]
        x = 0
        for index, lines in enumerate(rendered):
            child_width = widths[index]
            offset = 0
            if self.align == "center":
                offset = (height - len(lines)) // 2
            elif self.align == "end":
                offset = height - len(lines)
            for row, line in enumerate(lines):
                target = row + offset
                if 0 <= target < len(result):
                    result[target] = _composite_line(result[target], line, x, child_width, safe_width)
            x += child_width + self.gap
        return result


def _visible_lines(lines: list[str]) -> list[str]:
    return [_strip_terminal_sequences(line).rstrip() for line in lines]


def test_allocates_vertical_grow_space_deterministically() -> None:
    frame = render_layout_frame(
        _VStack(
            [
                StackEntry(component=_Text("top"), basis=1, shrink=0),
                StackEntry(component=_Text("body"), basis=0, grow=1),
            ]
        ),
        10,
        4,
        lambda: None,
    )

    assert [child.rect.height for child in frame.root.children] == [1, 3]
    assert _visible_lines(frame.lines) == ["top", "body", "", ""]


def test_does_not_render_fixed_basis_scroll_content_during_stack_measurement() -> None:
    content = _RenderSpy(["one", "two", "three"])
    transcript = ScrollView(content)
    root = _VStack(
        [StackEntry(component=transcript, basis=0, grow=1), StackEntry(component=_Text("dock"), basis="auto")]
    )

    render_layout_frame(root, 10, 3, lambda: None)

    assert content.render_count == 1


def test_paints_only_clipped_rows_from_very_large_scroll_content() -> None:
    line_count = 1_000_000_000
    lines = _HugeLines(
        line_count,
        {
            line_count - 4: "before",
            line_count - 3: "visible 1",
            line_count - 2: "visible 2",
            line_count - 1: "visible 3",
        },
    )
    transcript = ScrollView(_HugeContent(lines), ScrollViewOptions(follow="end"))

    frame = render_layout_frame(transcript, 10, 3, lambda: None)

    assert _visible_lines(frame.lines) == ["visible 1", "visible 2", "visible 3"]


def test_shrinks_entries_to_their_minimum_sizes() -> None:
    frame = render_layout_frame(
        _VStack(
            [
                StackEntry(component=_Text("a1\na2\na3"), shrink=1, min_size=1),
                StackEntry(component=_Text("b1\nb2\nb3"), shrink=0),
            ]
        ),
        10,
        4,
        lambda: None,
    )

    assert [child.rect.height for child in frame.root.children] == [1, 3]
    assert _visible_lines(frame.lines) == ["a1", "b1", "b2", "b3"]


def test_includes_nested_minimum_sizes_in_intrinsic_stack_measurement() -> None:
    dock = _VStack(
        [
            _Text("top1\ntop2\ntop3"),
            StackEntry(component=_Text("selector"), min_size=3),
            _Text("below"),
            StackEntry(component=_Text("footer"), min_size=1),
        ]
    )
    frame = render_layout_frame(
        _VStack(
            [
                StackEntry(component=_Text("body"), basis=0, grow=1, min_size=1),
                StackEntry(component=dock, basis="auto", min_size=1),
            ]
        ),
        10,
        9,
        lambda: None,
    )

    assert _visible_lines(frame.lines) == ["body", "top1", "top2", "top3", "selector", "", "", "below", "footer"]


def test_omits_gaps_around_invisible_entries() -> None:
    stack = _VStack(
        [
            _Text("one"),
            StackEntry(component=_Text("hidden"), visible=lambda _viewport: False),
            _Text("two"),
        ],
        StackOptions(gap=1),
    )

    assert [line.rstrip() for line in stack.render(10)] == ["one", "", "two"]


def test_crops_kitty_images_at_a_scroll_views_lower_boundary() -> None:
    image_id = 124
    image_line = encode_kitty("AAAA", columns=2, rows=3, image_id=image_id, move_cursor=False)
    register_kitty_image_metadata(KittyImageMetadata(image_id=image_id, columns=2, rows=3, width_px=100, height_px=100))
    transcript = ScrollView(_Lines(["one", "two", image_line, "", ""]))

    frame = render_layout_frame(
        _VStack([StackEntry(component=transcript, basis=0, grow=1), _Text("dock")]),
        20,
        4,
        lambda: None,
    )

    # The image declares 3 rows but only 1 fits above the dock, so the emitted
    # sequence must carry a source-crop (`y=`/`h=`) and a reduced row count.
    assert "y=0,h=34,r=1" in frame.lines[2]


def test_composes_horizontal_children_at_allocated_widths() -> None:
    frame = render_layout_frame(
        _HStack(
            [
                StackEntry(component=_Text("left"), basis=6, shrink=0),
                StackEntry(component=_Text("right"), basis=6, shrink=0),
            ]
        ),
        12,
        1,
        lambda: None,
    )

    assert _visible_lines(frame.lines) == ["left  right"]


def test_does_not_paint_zero_width_horizontal_children() -> None:
    frame = render_layout_frame(
        _HStack(
            [
                StackEntry(component=_Text("hidden"), basis=0, shrink=0),
                StackEntry(component=_Text("shown"), basis=0, grow=1),
            ]
        ),
        5,
        1,
        lambda: None,
    )

    assert _visible_lines(frame.lines) == ["shown"]


def test_renders_a_transient_proportional_scrollbar_without_replacing_cell_content() -> None:
    source_lines = ["abcd界", "abcde2", "abcde3", "abcde4", "abcde5", "abcde6", "abcde7", "abcde8"]
    content_background = "\x1b[42m"
    scrollbar_background = "\x1b[48;5;1m"

    def scrollbar_style(text: str) -> str:
        return f"{scrollbar_background}{text}\x1b[49m"

    def content_bg(text: str) -> str:
        return f"{content_background}{text}\x1b[49m"

    content = _Text("\n".join(source_lines), bg_fn=content_bg)
    scroll_view = ScrollView(
        content,
        ScrollViewOptions(scrollbar="auto", scrollbar_style=scrollbar_style, scrollbar_hide_delay_ms=10),
    )

    def render() -> list[str]:
        return render_layout_frame(scroll_view, 6, 4, lambda: None).lines

    def thumb_rows(lines: list[str]) -> list[bool]:
        return [scrollbar_background in line for line in lines]

    lines = render()
    assert thumb_rows(lines) == [False, False, False, False]
    assert [_strip_terminal_sequences(line) for line in lines] == source_lines[:4]

    scroll_view.scroll_by(2)
    lines = render()
    assert thumb_rows(lines) == [False, True, True, False]
    assert [_strip_terminal_sequences(line) for line in lines] == source_lines[2:6]
    assert lines[1].rfind(content_background) < lines[1].rfind(scrollbar_background)

    time.sleep(0.03)
    lines = render()
    assert thumb_rows(lines) == [False, False, False, False]

    scroll_view.scroll_to_end()
    lines = render()
    assert thumb_rows(lines) == [False, False, True, True]
    assert [_strip_terminal_sequences(line) for line in lines] == source_lines[4:]

    followed_content = _Text("\n".join(source_lines))
    followed = ScrollView(
        followed_content, ScrollViewOptions(follow="end", scrollbar="auto", scrollbar_style=scrollbar_style)
    )
    render_layout_frame(followed, 6, 4, lambda: None)
    assert followed.scroll_top == 4
    followed_content.set_text("\n".join(source_lines) + "\nabcde9")
    growth_frame = render_layout_frame(followed, 6, 4, lambda: None)
    assert followed.scroll_top == 5
    assert all(scrollbar_background not in line for line in growth_frame.lines)

    fitting_content = _Text("1\n2")
    automatic = ScrollView(fitting_content, ScrollViewOptions(scrollbar="auto", scrollbar_style=scrollbar_style))
    render_layout_frame(automatic, 6, 4, lambda: None)
    automatic.scroll_by(1)
    assert all(scrollbar_background not in line for line in render_layout_frame(automatic, 6, 4, lambda: None).lines)

    always_fitting = ScrollView(fitting_content, ScrollViewOptions(scrollbar="always", scrollbar_style=scrollbar_style))
    always_fitting_frame = render_layout_frame(always_fitting, 6, 4, lambda: None)
    assert always_fitting_frame.root.children[0].rect.width == 5
    assert all(scrollbar_background in line for line in always_fitting_frame.lines)

    always_overflowing = ScrollView(content, ScrollViewOptions(scrollbar="always", scrollbar_style=scrollbar_style))
    always_overflowing_frame = render_layout_frame(always_overflowing, 6, 4, lambda: None)
    assert always_overflowing_frame.root.children[0].rect.width == 5
    assert sum(scrollbar_background in line for line in always_overflowing_frame.lines) == 2

    def thumb_height_for(content_height: int) -> int:
        sized = ScrollView(
            _Text("\n".join("x" for _ in range(content_height))),
            ScrollViewOptions(scrollbar="auto", scrollbar_style=scrollbar_style),
        )
        render_layout_frame(sized, 6, 20, lambda: None)
        sized.scroll_by(1)
        return sum(scrollbar_background in line for line in render_layout_frame(sized, 6, 20, lambda: None).lines)

    assert thumb_height_for(21) == 19
    assert thumb_height_for(40) == 10
    assert thumb_height_for(100) == 4
    assert thumb_height_for(400) == 2


def test_updates_reserved_scrollbar_layout_at_runtime() -> None:
    scroll_view = ScrollView(_Text("123456"), ScrollViewOptions(scrollbar="always"))

    def render():
        return render_layout_frame(_HStack([scroll_view], StackOptions(align="start")), 6, 2, lambda: None)

    always = render()
    assert _visible_lines(always.lines) == ["12345", "6"]
    assert always.root.children[0].rect.width == 6
    assert always.root.children[0].children[0].rect.width == 5

    scroll_view.set_scrollbar("hidden")
    assert render().root.children[0].children[0].rect.width == 6
    assert scroll_view.is_scrollbar_visible is False


def test_measures_nested_scroll_content_from_constrained_child_geometry() -> None:
    inner = ScrollView(_Text("1\n2\n3\n4\n5\n6"))
    outer = ScrollView(_VStack([StackEntry(component=inner, basis=2), _Text("tail")]))

    render_layout_frame(outer, 10, 2, lambda: None)

    assert inner.viewport_height == 2
    assert outer.scroll_by(10) == 9
    assert outer.scroll_top == 1


def test_rebuilds_geometry_after_content_changes() -> None:
    text = _Text("one")
    root = _VStack([text])

    first = render_layout_frame(root, 10, 4, lambda: None)
    text.set_text("one\ntwo\nthree")
    second = render_layout_frame(root, 10, 4, lambda: None)

    assert len(first.root.children[0].lines or []) == 1
    assert len(second.root.children[0].lines or []) == 3


def test_scrollbar_geometry_and_lookup_helpers() -> None:
    inner = ScrollView(_Text("i1\ni2\ni3\ni4\ni5\ni6"), ScrollViewOptions(scrollbar="auto"))
    outer = ScrollView(
        _VStack([StackEntry(component=inner, basis=2), _Text("tail")]),
        ScrollViewOptions(scrollbar="auto"),
    )
    render_layout_frame(outer, 6, 2, lambda: None)
    inner.scroll_by(1)
    outer.scroll_by(1)
    frame = render_layout_frame(outer, 6, 2, lambda: None)

    inner_box = get_scroll_view_box(frame, inner)
    outer_box = get_scroll_view_box(frame, outer)

    assert inner_box is not None
    assert outer_box is not None
    assert get_scrollbar_geometry(inner_box) is not None
    assert get_scroll_views_at(frame, 0, 0) == [inner, outer]
    assert contains_point(outer_box.rect, 0, 0) is True
    assert contains_point(outer_box.rect, 6, 0) is False
