"""Box container component, ported from `packages/tui/src/components/box.ts`.

No dedicated upstream test file exists for `Box`; `tests/test_box.py` covers
its documented behavior directly (padding, background application, child
render caching).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pi_tui.component import Component
from pi_tui.utils import apply_background_to_line, visible_width


@dataclass
class _RenderCache:
    child_lines: list[str]
    width: int
    bg_sample: str | None
    lines: list[str]


class Box(Component):
    """A container that applies padding and background to all children."""

    def __init__(
        self,
        padding_x: int = 1,
        padding_y: int = 1,
        bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.children: list[Component] = []
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._bg_fn = bg_fn
        self._cache: _RenderCache | None = None

    def add_child(self, component: Component) -> None:
        self.children.append(component)
        self._invalidate_cache()

    def remove_child(self, component: Component) -> None:
        if component in self.children:
            self.children.remove(component)
            self._invalidate_cache()

    def clear(self) -> None:
        self.children = []
        self._invalidate_cache()

    def set_bg_fn(self, bg_fn: Callable[[str], str] | None = None) -> None:
        self._bg_fn = bg_fn
        # Don't invalidate here - we'll detect bgFn changes by sampling output.

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _match_cache(self, width: int, child_lines: list[str], bg_sample: str | None) -> bool:
        cache = self._cache
        return (
            cache is not None
            and cache.width == width
            and cache.bg_sample == bg_sample
            and len(cache.child_lines) == len(child_lines)
            and cache.child_lines == child_lines
        )

    def invalidate(self) -> None:
        self._invalidate_cache()
        for child in self.children:
            child.invalidate()

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []

        content_width = max(1, width - self._padding_x * 2)
        left_pad = " " * self._padding_x

        child_lines: list[str] = []
        for child in self.children:
            for line in child.render(content_width):
                child_lines.append(left_pad + line)

        if not child_lines:
            return []

        bg_sample = self._bg_fn("test") if self._bg_fn else None

        if self._match_cache(width, child_lines, bg_sample):
            assert self._cache is not None
            return self._cache.lines

        result: list[str] = []

        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        for line in child_lines:
            result.append(self._apply_bg(line, width))

        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        self._cache = _RenderCache(child_lines, width, bg_sample, result)

        return result

    def _apply_bg(self, line: str, width: int) -> str:
        vis_len = visible_width(line)
        pad_needed = max(0, width - vis_len)
        padded = line + " " * pad_needed

        if self._bg_fn:
            return apply_background_to_line(padded, width, self._bg_fn)
        return padded
