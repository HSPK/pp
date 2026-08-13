"""Bordered select-list dialogs: theme, thinking level, show-images, confirm.

Ported from ``theme-selector.ts``, ``thinking-selector.ts`` and
``show-images-selector.ts`` under
``packages/coding-agent/src/modes/interactive/components/``. ``ConfirmSelector``
covers the Yes/No shape upstream builds on top of ``extension-selector.ts``
(``showExtensionConfirm``), used by the built-in commands that ask for
confirmation and by ``ExtensionUIContext.confirm``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pi_tui.component import Container
from pi_tui.components.select_list import SelectItem, SelectList, SelectListLayoutOptions
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from ..theme.theme import get_available_themes, get_select_list_theme, theme
from .dynamic_border import DynamicBorder

_SELECT_LIST_LAYOUT = SelectListLayoutOptions(min_primary_column_width=12, max_primary_column_width=32)

LEVEL_DESCRIPTIONS: dict[str, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Extra-high reasoning (~32k tokens)",
    "max": "Maximum reasoning",
}


class _BorderedSelector(Container):
    """Top border, select list, bottom border."""

    def __init__(self, items: list[SelectItem], max_visible: int) -> None:
        super().__init__()
        self.add_child(DynamicBorder())
        self.select_list = SelectList(items, max_visible, get_select_list_theme(), _SELECT_LIST_LAYOUT)
        self.add_child(self.select_list)
        self.add_child(DynamicBorder())

    def get_select_list(self) -> SelectList:
        return self.select_list

    def handle_input(self, key_data: str) -> None:
        self.select_list.handle_input(key_data)


class ThemeSelectorComponent(_BorderedSelector):
    def __init__(
        self,
        current_theme: str,
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
        on_preview: Callable[[str], None],
        *,
        custom_themes_dir: str | None = None,
    ) -> None:
        themes = get_available_themes(custom_themes_dir=custom_themes_dir)
        items = [
            SelectItem(
                value=name,
                label=name,
                description="(current)" if name == current_theme else None,
            )
            for name in themes
        ]
        super().__init__(items, 10)
        self.on_preview = on_preview

        if current_theme in themes:
            self.select_list.set_selected_index(themes.index(current_theme))

        self.select_list.on_select = lambda item: on_select(item.value)
        self.select_list.on_cancel = on_cancel
        self.select_list.on_selection_change = lambda item: self.on_preview(item.value)


class ThinkingSelectorComponent(_BorderedSelector):
    def __init__(
        self,
        current_level: str,
        available_levels: Sequence[str],
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
    ) -> None:
        items = [
            SelectItem(value=level, label=level, description=LEVEL_DESCRIPTIONS.get(level))
            for level in available_levels
        ]
        super().__init__(items, len(items))

        for index, item in enumerate(items):
            if item.value == current_level:
                self.select_list.set_selected_index(index)
                break

        self.select_list.on_select = lambda item: on_select(item.value)
        self.select_list.on_cancel = on_cancel


class ShowImagesSelectorComponent(_BorderedSelector):
    def __init__(
        self,
        current_value: bool,
        on_select: Callable[[bool], None],
        on_cancel: Callable[[], None],
    ) -> None:
        items = [
            SelectItem(value="yes", label="Yes", description="Show images inline in terminal"),
            SelectItem(value="no", label="No", description="Show text placeholder instead"),
        ]
        super().__init__(items, 5)
        self.select_list.set_selected_index(0 if current_value else 1)
        self.select_list.on_select = lambda item: on_select(item.value == "yes")
        self.select_list.on_cancel = on_cancel


class ConfirmSelectorComponent(Container):
    """Title plus a Yes/No list.

    Upstream reaches this through ``showExtensionConfirm`` -> ``showExtensionSelector``
    with ``["Yes", "No"]``; the title carries ``f"{title}\\n{message}"``.
    """

    def __init__(
        self,
        title: str,
        message: str,
        on_select: Callable[[bool], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        items = [SelectItem(value="yes", label="Yes"), SelectItem(value="no", label="No")]
        self.select_list = SelectList(items, 5, get_select_list_theme(), _SELECT_LIST_LAYOUT)
        self.select_list.on_select = lambda item: on_select(item.value == "yes")
        self.select_list.on_cancel = on_cancel

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", theme.bold(f"{title}\n{message}")), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(self.select_list)
        self.add_child(DynamicBorder())

    def get_select_list(self) -> SelectList:
        return self.select_list

    def handle_input(self, key_data: str) -> None:
        self.select_list.handle_input(key_data)


__all__ = [
    "LEVEL_DESCRIPTIONS",
    "ConfirmSelectorComponent",
    "ShowImagesSelectorComponent",
    "ThemeSelectorComponent",
    "ThinkingSelectorComponent",
]
