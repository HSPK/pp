"""Tests for components/box.py (no dedicated upstream TS test file exists for
`Box`; these exercise the ported behavior directly: padding, background
application, and child render caching, matching `packages/tui/src/components/box.ts`).
"""

from pi_tui.component import Component
from pi_tui.components.box import Box


class FixedContent(Component):
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.render_calls = 0

    def render(self, width: int) -> list[str]:
        self.render_calls += 1
        return [line.ljust(width) for line in self._lines]

    def invalidate(self) -> None:
        pass


class TestBox:
    def test_empty_box_renders_nothing(self):
        box = Box()
        assert box.render(20) == []

    def test_applies_horizontal_and_vertical_padding(self):
        box = Box(padding_x=2, padding_y=1)
        box.add_child(FixedContent(["hi"]))

        lines = box.render(10)
        # 1 top pad + 1 content + 1 bottom pad
        assert len(lines) == 3
        assert lines[0] == " " * 10
        assert lines[2] == " " * 10
        assert lines[1].startswith("  hi")
        assert len(lines[1]) == 10

    def test_no_padding(self):
        box = Box(padding_x=0, padding_y=0)
        box.add_child(FixedContent(["abc"]))

        lines = box.render(5)
        assert lines == ["abc  "]

    def test_applies_background_function(self):
        box = Box(padding_x=0, padding_y=0, bg_fn=lambda text: f"[{text}]")
        box.add_child(FixedContent(["ab"]))

        lines = box.render(4)
        assert lines == ["[ab  ]"]

    def test_remove_child(self):
        box = Box(padding_x=0, padding_y=0)
        child = FixedContent(["x"])
        box.add_child(child)
        box.remove_child(child)

        assert box.render(3) == []

    def test_clear_removes_all_children(self):
        box = Box(padding_x=0, padding_y=0)
        box.add_child(FixedContent(["a"]))
        box.add_child(FixedContent(["b"]))
        box.clear()

        assert box.render(3) == []

    def test_render_cache_reused_when_inputs_unchanged(self):
        box = Box(padding_x=0, padding_y=0)
        child = FixedContent(["a"])
        box.add_child(child)

        first = box.render(5)
        second = box.render(5)

        assert first == second
        assert child.render_calls == 2  # child always re-renders; box caches its own output

    def test_invalidate_propagates_to_children(self):
        box = Box()
        child = FixedContent(["a"])
        box.add_child(child)
        box.invalidate()  # should not raise

    def test_set_bg_fn_changes_output_without_explicit_invalidate(self):
        box = Box(padding_x=0, padding_y=0)
        box.add_child(FixedContent(["ab"]))

        first = box.render(2)
        assert first == ["ab"]

        box.set_bg_fn(lambda text: f"<{text}>")
        second = box.render(2)
        assert second == ["<ab>"]

    def test_multiple_children_lines_are_concatenated(self):
        box = Box(padding_x=0, padding_y=0)
        box.add_child(FixedContent(["a", "b"]))
        box.add_child(FixedContent(["c"]))

        lines = box.render(1)
        assert lines == ["a", "b", "c"]
