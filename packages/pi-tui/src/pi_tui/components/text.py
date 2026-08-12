"""Word-wrapping multi-line text component.

Ported from ``packages/tui/src/components/text.ts``.
"""

from __future__ import annotations

from collections.abc import Callable

from ..component import Component
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


class Text(Component):
    """Displays multi-line text with word wrapping."""

    def __init__(
        self,
        text: str = "",
        padding_x: int = 1,
        padding_y: int = 1,
        custom_bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.text = text
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.custom_bg_fn = custom_bg_fn
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self.text = text
        self._clear_cache()

    def set_custom_bg_fn(self, custom_bg_fn: Callable[[str], str] | None = None) -> None:
        self.custom_bg_fn = custom_bg_fn
        self._clear_cache()

    def invalidate(self) -> None:
        self._clear_cache()

    def _clear_cache(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_text == self.text and self._cached_width == width:
            return self._cached_lines

        if not self.text or self.text.strip() == "":
            self._cached_text = self.text
            self._cached_width = width
            self._cached_lines = []
            return self._cached_lines

        normalized_text = self.text.replace("\t", "   ")
        content_width = max(1, width - self.padding_x * 2)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)

        margin = " " * self.padding_x
        content_lines: list[str] = []
        for line in wrapped_lines:
            line_with_margins = margin + line + margin
            if self.custom_bg_fn:
                content_lines.append(apply_background_to_line(line_with_margins, width, self.custom_bg_fn))
            else:
                padding_needed = max(0, width - visible_width(line_with_margins))
                content_lines.append(line_with_margins + " " * padding_needed)

        empty_line = " " * width
        empty_lines: list[str] = []
        for _ in range(self.padding_y):
            empty_lines.append(
                apply_background_to_line(empty_line, width, self.custom_bg_fn) if self.custom_bg_fn else empty_line
            )

        result = [*empty_lines, *content_lines, *empty_lines]

        self._cached_text = self.text
        self._cached_width = width
        self._cached_lines = result

        return result if len(result) > 0 else [""]
