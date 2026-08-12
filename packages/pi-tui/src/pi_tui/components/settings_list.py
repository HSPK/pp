"""Settings list component, ported from
`packages/tui/src/components/settings-list.ts`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pi_tui.component import Component
from pi_tui.components.input import Input
from pi_tui.fuzzy import fuzzy_filter
from pi_tui.keybindings import get_keybindings
from pi_tui.utils import truncate_to_width, visible_width, wrap_text_with_ansi


@dataclass
class SettingItem:
    """A single row in a `SettingsList`.

    `submenu`, if provided, is a factory called with the current value and a
    `done` callback; it returns a `Component` to render in place of the main
    list until `done` is invoked.
    """

    id: str
    label: str
    current_value: str
    description: str | None = None
    values: list[str] | None = None
    submenu: Callable[[str, Callable[[str | None], None]], Component] | None = None


@dataclass
class SettingsListTheme:
    label: Callable[[str, bool], str]
    value: Callable[[str, bool], str]
    description: Callable[[str], str]
    cursor: str
    hint: Callable[[str], str]


@dataclass
class SettingsListOptions:
    enable_search: bool = False


class SettingsList(Component):
    """Vertical list of editable settings with optional fuzzy search and
    per-item submenus.
    """

    def __init__(
        self,
        items: list[SettingItem],
        max_visible: int,
        theme: SettingsListTheme,
        on_change: Callable[[str, str], None],
        on_cancel: Callable[[], None],
        options: SettingsListOptions | None = None,
    ) -> None:
        self.items = items
        self.filtered_items: list[SettingItem] = items
        self.max_visible = max_visible
        self.theme = theme
        self.on_change = on_change
        self.on_cancel = on_cancel
        options = options or SettingsListOptions()
        self.search_enabled = options.enable_search
        self.search_input: Input | None = Input() if self.search_enabled else None

        self.selected_index = 0

        # Submenu state.
        self._submenu_component: Component | None = None
        self._submenu_item_index: int | None = None

    def update_value(self, id_: str, new_value: str) -> None:
        """Update an item's `current_value`."""
        for item in self.items:
            if item.id == id_:
                item.current_value = new_value
                return

    def invalidate(self) -> None:
        if self._submenu_component is not None:
            self._submenu_component.invalidate()

    def render(self, width: int) -> list[str]:
        if self._submenu_component is not None:
            return self._submenu_component.render(width)

        return self._render_main_list(width)

    def _render_main_list(self, width: int) -> list[str]:
        lines: list[str] = []

        if self.search_enabled and self.search_input is not None:
            lines.extend(self.search_input.render(width))
            lines.append("")

        if len(self.items) == 0:
            lines.append(self.theme.hint("  No settings available"))
            if self.search_enabled:
                self._add_hint_line(lines, width)
            return lines

        display_items = self.filtered_items if self.search_enabled else self.items
        if len(display_items) == 0:
            lines.append(truncate_to_width(self.theme.hint("  No matching settings"), width))
            self._add_hint_line(lines, width)
            return lines

        # Calculate visible range with scrolling.
        start_index = max(
            0,
            min(self.selected_index - self.max_visible // 2, len(display_items) - self.max_visible),
        )
        end_index = min(start_index + self.max_visible, len(display_items))

        # Calculate max label width for alignment.
        max_label_width = min(30, max(visible_width(item.label) for item in self.items))

        # Render visible items.
        for i in range(start_index, end_index):
            item = display_items[i]

            is_selected = i == self.selected_index
            prefix = self.theme.cursor if is_selected else "  "
            prefix_width = visible_width(prefix)

            # Pad label to align values.
            label_padded = item.label + " " * max(0, max_label_width - visible_width(item.label))
            label_text = self.theme.label(label_padded, is_selected)

            # Calculate space for value.
            separator = "  "
            used_width = prefix_width + max_label_width + visible_width(separator)
            value_max_width = width - used_width - 2

            value_text = self.theme.value(truncate_to_width(item.current_value, value_max_width, ""), is_selected)

            lines.append(truncate_to_width(prefix + label_text + separator + value_text, width))

        # Add scroll indicator if needed.
        if start_index > 0 or end_index < len(display_items):
            scroll_text = f"  ({self.selected_index + 1}/{len(display_items)})"
            lines.append(self.theme.hint(truncate_to_width(scroll_text, width - 2, "")))

        # Add description for selected item.
        selected_item = display_items[self.selected_index] if 0 <= self.selected_index < len(display_items) else None
        if selected_item is not None and selected_item.description:
            lines.append("")
            wrapped_desc = wrap_text_with_ansi(selected_item.description, width - 4)
            for line in wrapped_desc:
                lines.append(self.theme.description(f"  {line}"))

        self._add_hint_line(lines, width)

        return lines

    def handle_input(self, data: str) -> None:
        # If submenu is active, delegate all input to it. The submenu's
        # on_cancel (triggered by escape) will call done() which closes it.
        if self._submenu_component is not None:
            self._submenu_component.handle_input(data)
            return

        kb = get_keybindings()
        display_items = self.filtered_items if self.search_enabled else self.items
        if kb.matches(data, "tui.select.up"):
            if len(display_items) == 0:
                return
            self.selected_index = len(display_items) - 1 if self.selected_index == 0 else self.selected_index - 1
        elif kb.matches(data, "tui.select.down"):
            if len(display_items) == 0:
                return
            self.selected_index = 0 if self.selected_index == len(display_items) - 1 else self.selected_index + 1
        elif kb.matches(data, "tui.select.confirm") or (
            data == " "
            and (not self.search_enabled or (self.search_input is not None and len(self.search_input.get_value()) == 0))
        ):
            self._activate_item()
        elif kb.matches(data, "tui.select.cancel"):
            self.on_cancel()
        elif self.search_enabled and self.search_input is not None:
            self.search_input.handle_input(data)
            self._apply_filter(self.search_input.get_value())

    def _activate_item(self) -> None:
        display_items = self.filtered_items if self.search_enabled else self.items
        if not (0 <= self.selected_index < len(display_items)):
            return
        item = display_items[self.selected_index]

        if item.submenu is not None:
            # Open submenu, passing current value so it can pre-select correctly.
            self._submenu_item_index = self.selected_index

            def done(selected_value: str | None = None) -> None:
                if selected_value is not None:
                    item.current_value = selected_value
                    self.on_change(item.id, selected_value)
                self._close_submenu()

            self._submenu_component = item.submenu(item.current_value, done)
        elif item.values:
            # Cycle through values.
            current_index = item.values.index(item.current_value) if item.current_value in item.values else -1
            next_index = (current_index + 1) % len(item.values)
            new_value = item.values[next_index]
            item.current_value = new_value
            self.on_change(item.id, new_value)

    def _close_submenu(self) -> None:
        self._submenu_component = None
        # Restore selection to the item that opened the submenu.
        if self._submenu_item_index is not None:
            self.selected_index = self._submenu_item_index
            self._submenu_item_index = None

    def _apply_filter(self, query: str) -> None:
        self.filtered_items = fuzzy_filter(self.items, query, lambda item: item.label)
        self.selected_index = 0

    def _add_hint_line(self, lines: list[str], width: int) -> None:
        lines.append("")
        lines.append(
            truncate_to_width(
                self.theme.hint(
                    "  Type to search · Enter/Space to change · Esc to cancel"
                    if self.search_enabled
                    else "  Enter/Space to change · Esc to cancel"
                ),
                width,
            )
        )
