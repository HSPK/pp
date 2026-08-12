"""Selectable list component, ported from
`packages/tui/src/components/select-list.ts`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pi_tui.component import Component
from pi_tui.keybindings import get_keybindings
from pi_tui.utils import truncate_to_width, visible_width

_DEFAULT_PRIMARY_COLUMN_WIDTH = 32
_PRIMARY_COLUMN_GAP = 2
_MIN_DESCRIPTION_WIDTH = 10

_MULTILINE_RE = re.compile(r"[\r\n]+")


def _normalize_to_single_line(text: str) -> str:
    return _MULTILINE_RE.sub(" ", text).strip()


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


@dataclass
class SelectItem:
    value: str
    label: str
    description: str | None = None


@dataclass
class SelectListTheme:
    selected_prefix: Callable[[str], str]
    selected_text: Callable[[str], str]
    description: Callable[[str], str]
    scroll_info: Callable[[str], str]
    no_match: Callable[[str], str]


@dataclass
class SelectListTruncatePrimaryContext:
    text: str
    max_width: int
    column_width: int
    item: SelectItem
    is_selected: bool


@dataclass
class SelectListLayoutOptions:
    min_primary_column_width: int | None = None
    max_primary_column_width: int | None = None
    truncate_primary: Callable[[SelectListTruncatePrimaryContext], str] | None = None


class SelectList(Component):
    def __init__(
        self,
        items: list[SelectItem],
        max_visible: int,
        theme: SelectListTheme,
        layout: SelectListLayoutOptions | None = None,
    ) -> None:
        self._items = items
        self._filtered_items = items
        self._selected_index = 0
        self._max_visible = max_visible
        self._theme = theme
        self._layout = layout if layout is not None else SelectListLayoutOptions()

        self.on_select: Callable[[SelectItem], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_selection_change: Callable[[SelectItem], None] | None = None

    def set_filter(self, filter_text: str) -> None:
        filter_lower = filter_text.lower()
        self._filtered_items = [item for item in self._items if item.value.lower().startswith(filter_lower)]
        # Reset selection when filter changes.
        self._selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self._selected_index = max(0, min(index, len(self._filtered_items) - 1))

    def invalidate(self) -> None:
        # No cached state to invalidate currently.
        pass

    def render(self, width: int) -> list[str]:
        lines: list[str] = []

        if not self._filtered_items:
            lines.append(self._theme.no_match("  No matching commands"))
            return lines

        primary_column_width = self._get_primary_column_width()

        start_index = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(self._filtered_items) - self._max_visible,
            ),
        )
        end_index = min(start_index + self._max_visible, len(self._filtered_items))

        for i in range(start_index, end_index):
            item = self._filtered_items[i]
            is_selected = i == self._selected_index
            description_single_line = _normalize_to_single_line(item.description) if item.description else None
            lines.append(self._render_item(item, is_selected, width, description_single_line, primary_column_width))

        if start_index > 0 or end_index < len(self._filtered_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_items)})"
            lines.append(self._theme.scroll_info(truncate_to_width(scroll_text, width - 2, "")))

        return lines

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up"):
            self._selected_index = (
                len(self._filtered_items) - 1 if self._selected_index == 0 else self._selected_index - 1
            )
            self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.down"):
            self._selected_index = (
                0 if self._selected_index == len(self._filtered_items) - 1 else self._selected_index + 1
            )
            self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.confirm"):
            selected_item = (
                self._filtered_items[self._selected_index]
                if 0 <= self._selected_index < len(self._filtered_items)
                else None
            )
            if selected_item and self.on_select:
                self.on_select(selected_item)
        elif kb.matches(key_data, "tui.select.cancel") and self.on_cancel:
            self.on_cancel()

    def _render_item(
        self,
        item: SelectItem,
        is_selected: bool,
        width: int,
        description_single_line: str | None,
        primary_column_width: int,
    ) -> str:
        prefix = "→ " if is_selected else "  "
        prefix_width = visible_width(prefix)

        if description_single_line and width > 40:
            effective_primary_column_width = max(1, min(primary_column_width, width - prefix_width - 4))
            max_primary_width = max(1, effective_primary_column_width - _PRIMARY_COLUMN_GAP)
            truncated_value = self._truncate_primary(
                item, is_selected, max_primary_width, effective_primary_column_width
            )
            truncated_value_width = visible_width(truncated_value)
            spacing = " " * max(1, effective_primary_column_width - truncated_value_width)
            description_start = prefix_width + truncated_value_width + len(spacing)
            remaining_width = width - description_start - 2  # -2 for safety.

            if remaining_width > _MIN_DESCRIPTION_WIDTH:
                truncated_desc = truncate_to_width(description_single_line, remaining_width, "")
                if is_selected:
                    return self._theme.selected_text(f"{prefix}{truncated_value}{spacing}{truncated_desc}")

                desc_text = self._theme.description(spacing + truncated_desc)
                return prefix + truncated_value + desc_text

        max_width = width - prefix_width - 2
        truncated_value = self._truncate_primary(item, is_selected, max_width, max_width)
        if is_selected:
            return self._theme.selected_text(f"{prefix}{truncated_value}")

        return prefix + truncated_value

    def _get_primary_column_width(self) -> int:
        min_width, max_width = self._get_primary_column_bounds()
        widest_primary = 0
        for item in self._filtered_items:
            widest_primary = max(widest_primary, visible_width(self._get_display_value(item)) + _PRIMARY_COLUMN_GAP)

        return _clamp(widest_primary, min_width, max_width)

    def _get_primary_column_bounds(self) -> tuple[int, int]:
        raw_min = (
            self._layout.min_primary_column_width
            if self._layout.min_primary_column_width is not None
            else (
                self._layout.max_primary_column_width
                if self._layout.max_primary_column_width is not None
                else _DEFAULT_PRIMARY_COLUMN_WIDTH
            )
        )
        raw_max = (
            self._layout.max_primary_column_width
            if self._layout.max_primary_column_width is not None
            else (
                self._layout.min_primary_column_width
                if self._layout.min_primary_column_width is not None
                else _DEFAULT_PRIMARY_COLUMN_WIDTH
            )
        )

        return (max(1, min(raw_min, raw_max)), max(1, max(raw_min, raw_max)))

    def _truncate_primary(self, item: SelectItem, is_selected: bool, max_width: int, column_width: int) -> str:
        display_value = self._get_display_value(item)
        if self._layout.truncate_primary:
            truncated_value = self._layout.truncate_primary(
                SelectListTruncatePrimaryContext(display_value, max_width, column_width, item, is_selected)
            )
        else:
            truncated_value = truncate_to_width(display_value, max_width, "")

        return truncate_to_width(truncated_value, max_width, "")

    def _get_display_value(self, item: SelectItem) -> str:
        return item.label or item.value

    def _notify_selection_change(self) -> None:
        selected_item = (
            self._filtered_items[self._selected_index]
            if 0 <= self._selected_index < len(self._filtered_items)
            else None
        )
        if selected_item and self.on_selection_change:
            self.on_selection_change(selected_item)

    def get_selected_item(self) -> SelectItem | None:
        if 0 <= self._selected_index < len(self._filtered_items):
            return self._filtered_items[self._selected_index]
        return None
