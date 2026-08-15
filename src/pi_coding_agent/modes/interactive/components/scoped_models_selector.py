"""The `/scoped-models` editor.

Ported from ``packages/coding-agent/src/modes/interactive/components/scoped-models-selector.ts``.

``enabled_ids`` is ``None`` when every model is enabled (no filter) and an
explicit ordered list otherwise; the order is the cycling order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pi_ai.models import Model
from pi_tui.component import Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.fuzzy import fuzzy_filter
from pi_tui.keybindings import get_keybindings
from pi_tui.keys import matches_key

from ..model_search import ModelSearchItem, get_model_search_text
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_text

EnabledIds = list[str] | None


def is_enabled(enabled_ids: EnabledIds, model_id: str) -> bool:
    return enabled_ids is None or model_id in enabled_ids


def toggle(enabled_ids: EnabledIds, model_id: str) -> EnabledIds:
    if enabled_ids is None:
        # First toggle narrows the set to just this model.
        return [model_id]
    if model_id in enabled_ids:
        return [item for item in enabled_ids if item != model_id]
    return [*enabled_ids, model_id]


def enable_all(enabled_ids: EnabledIds, all_ids: list[str], target_ids: Sequence[str] | None = None) -> EnabledIds:
    if enabled_ids is None:
        return None
    targets = list(target_ids) if target_ids is not None else all_ids
    result = list(enabled_ids)
    for model_id in targets:
        if model_id not in result:
            result.append(model_id)
    # Everything enabled collapses back to the "no filter" representation.
    if len(result) == len(all_ids) and all(model_id in all_ids for model_id in result):
        return None
    return result


def clear_all(enabled_ids: EnabledIds, all_ids: list[str], target_ids: Sequence[str] | None = None) -> EnabledIds:
    if enabled_ids is None:
        if target_ids is not None:
            return [model_id for model_id in all_ids if model_id not in target_ids]
        return []
    targets = set(target_ids if target_ids is not None else enabled_ids)
    return [model_id for model_id in enabled_ids if model_id not in targets]


def move(enabled_ids: EnabledIds, model_id: str, delta: int) -> EnabledIds:
    if enabled_ids is None:
        return None
    result = list(enabled_ids)
    if model_id not in result:
        return result
    index = result.index(model_id)
    new_index = index + delta
    if new_index < 0 or new_index >= len(result):
        return result
    result[index], result[new_index] = result[new_index], result[index]
    return result


def get_sorted_ids(enabled_ids: EnabledIds, all_ids: list[str]) -> list[str]:
    if enabled_ids is None:
        return all_ids
    enabled_set = set(enabled_ids)
    return [*enabled_ids, *[model_id for model_id in all_ids if model_id not in enabled_set]]


@dataclass
class ModelItem:
    full_id: str
    model: Model | None
    enabled: bool


@dataclass
class ModelsConfig:
    all_models: list[Model] = field(default_factory=list)
    enabled_model_ids: list[str] | None = None
    refresh_status: str | None = None


def _noop(*_args: object) -> None:
    return None


@dataclass
class ModelsCallbacks:
    on_change: Callable[[EnabledIds], Any] = _noop
    """Session-only change to the enabled set or order."""
    on_persist: Callable[[EnabledIds], Any] = _noop
    """Write the current selection to settings."""
    on_cancel: Callable[[], None] = _noop


class ScopedModelsSelectorComponent(Container):
    """Enable/disable and reorder the models that model-cycling walks through.

    Changes are session-only until explicitly persisted.
    """

    MAX_VISIBLE = 8

    def __init__(self, config: ModelsConfig, callbacks: ModelsCallbacks) -> None:
        super().__init__()
        self.callbacks = callbacks
        self.models_by_id: dict[str, Model] = {}
        self.all_ids: list[str] = []
        self.selected_index = 0
        self.is_dirty = False
        self.max_visible = self.MAX_VISIBLE
        self._focused = False

        for model in config.all_models:
            full_id = f"{model.provider}/{model.id}"
            self.models_by_id[full_id] = model
            self.all_ids.append(full_id)

        self.enabled_ids: EnabledIds = None if config.enabled_model_ids is None else list(config.enabled_model_ids)
        self.filtered_items = self._build_items()

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", theme.bold("Model Configuration")), 0, 0))
        self.add_child(
            Text(
                theme.fg("muted", f"Session-only. {key_text('app.models.save')} to save to settings."),
                0,
                0,
            )
        )
        self.add_child(Spacer(1))

        self.search_input = Input()
        self.add_child(self.search_input)
        self.add_child(Spacer(1))

        self.list_container = Container()
        self.add_child(self.list_container)
        self.add_child(Spacer(1))

        self.refresh_status_text: Text | None = None
        if config.refresh_status:
            self.refresh_status_text = Text(theme.fg("muted", f"  {config.refresh_status}"), 0, 0)
            self.add_child(self.refresh_status_text)

        self.footer_text = Text(self._get_footer_text(), 0, 0)
        self.add_child(self.footer_text)
        self.add_child(DynamicBorder())

        self._update_list()

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.search_input.focused = value

    def get_search_input(self) -> Input:
        return self.search_input

    def update_models(self, models: Sequence[Model], enabled_model_ids: list[str] | object | None = ...) -> None:
        selected_id = (
            self.filtered_items[self.selected_index].full_id
            if 0 <= self.selected_index < len(self.filtered_items)
            else None
        )
        if enabled_model_ids is not ...:
            self.enabled_ids = None if enabled_model_ids is None else list(enabled_model_ids)  # type: ignore[arg-type]

        self.models_by_id.clear()
        self.all_ids = []
        for model in models:
            full_id = f"{model.provider}/{model.id}"
            self.models_by_id[full_id] = model
            self.all_ids.append(full_id)

        self._refresh()
        if selected_id is not None:
            for index, item in enumerate(self.filtered_items):
                if item.full_id == selected_id:
                    self.selected_index = index
                    self._update_list()
                    break

    def set_refresh_status(self, message: str, kind: str) -> None:
        if self.refresh_status_text is not None:
            self.refresh_status_text.set_text(theme.fg(kind, f"  {message}"))

    def _build_items(self) -> list[ModelItem]:
        return [
            ModelItem(
                full_id=model_id,
                model=self.models_by_id.get(model_id),
                enabled=is_enabled(self.enabled_ids, model_id),
            )
            for model_id in get_sorted_ids(self.enabled_ids, self.all_ids)
        ]

    def _get_footer_text(self) -> str:
        if self.enabled_ids is None:
            count_text = "all enabled"
        else:
            enabled_count = sum(1 for model_id in self.enabled_ids if model_id in self.models_by_id)
            unavailable_count = sum(1 for model_id in self.enabled_ids if model_id not in self.models_by_id)
            suffix = f" · {unavailable_count} unavailable" if unavailable_count else ""
            count_text = f"{enabled_count}/{len(self.all_ids)} enabled{suffix}"

        parts = [
            f"{key_text('tui.select.confirm')} toggle",
            f"{key_text('app.models.enableAll')} all",
            f"{key_text('app.models.clearAll')} clear",
            f"{key_text('app.models.toggleProvider')} provider",
            f"{key_text('app.models.reorderUp')}/{key_text('app.models.reorderDown')} reorder",
            f"{key_text('app.models.save')} save",
            count_text,
        ]
        joined = " · ".join(parts)
        if self.is_dirty:
            return theme.fg("dim", f"  {joined} ") + theme.fg("warning", "(unsaved)")
        return theme.fg("dim", f"  {joined}")

    def _refresh(self) -> None:
        query = self.search_input.get_value()
        items = self._build_items()
        if query:
            self.filtered_items = fuzzy_filter(
                items,
                query,
                lambda item: (
                    get_model_search_text(
                        ModelSearchItem(id=item.model.id, provider=item.model.provider, name=item.model.name)
                    )
                    if item.model is not None
                    else item.full_id
                ),
            )
        else:
            self.filtered_items = items
        self.selected_index = min(self.selected_index, max(0, len(self.filtered_items) - 1))
        self._update_list()
        self.footer_text.set_text(self._get_footer_text())

    def _notify_change(self) -> None:
        self.callbacks.on_change(None if self.enabled_ids is None else list(self.enabled_ids))

    def _update_list(self) -> None:
        self.list_container.clear()

        if len(self.filtered_items) == 0:
            self.list_container.add_child(Text(theme.fg("muted", "  No matching models"), 0, 0))
            return

        start_index = max(
            0,
            min(
                self.selected_index - math.floor(self.max_visible / 2),
                len(self.filtered_items) - self.max_visible,
            ),
        )
        end_index = min(start_index + self.max_visible, len(self.filtered_items))
        all_enabled = self.enabled_ids is None

        for index in range(start_index, end_index):
            item = self.filtered_items[index]
            is_selected = index == self.selected_index
            prefix = theme.fg("accent", "→ ") if is_selected else "  "
            model_id = item.model.id if item.model is not None else item.full_id
            model_text = theme.fg("accent", model_id) if is_selected else model_id
            provider_badge = theme.fg(
                "muted", f" [{item.model.provider}]" if item.model is not None else " [unavailable]"
            )
            if item.model is None:
                status = theme.fg("dim", " ✗")
            elif all_enabled:
                status = ""
            else:
                status = theme.fg("success", " ✓") if item.enabled else theme.fg("dim", " ✗")
            self.list_container.add_child(Text(f"{prefix}{model_text}{provider_badge}{status}", 0, 0))

        if start_index > 0 or end_index < len(self.filtered_items):
            self.list_container.add_child(
                Text(
                    theme.fg("muted", f"  ({self.selected_index + 1}/{len(self.filtered_items)})"),
                    0,
                    0,
                )
            )

        selected = self.filtered_items[self.selected_index]
        self.list_container.add_child(Spacer(1))
        detail = f"Model Name: {selected.model.name}" if selected.model is not None else "Model unavailable"
        self.list_container.add_child(Text(theme.fg("muted", f"  {detail}"), 0, 0))

    def handle_input(self, data: str) -> None:
        keybindings = get_keybindings()

        if keybindings.matches(data, "tui.select.up"):
            if len(self.filtered_items) == 0:
                return
            self.selected_index = len(self.filtered_items) - 1 if self.selected_index == 0 else self.selected_index - 1
            self._update_list()
            return
        if keybindings.matches(data, "tui.select.down"):
            if len(self.filtered_items) == 0:
                return
            self.selected_index = 0 if self.selected_index == len(self.filtered_items) - 1 else self.selected_index + 1
            self._update_list()
            return

        reorder_up = keybindings.matches(data, "app.models.reorderUp")
        reorder_down = keybindings.matches(data, "app.models.reorderDown")
        if reorder_up or reorder_down:
            if self.enabled_ids is None or not (0 <= self.selected_index < len(self.filtered_items)):
                return
            item = self.filtered_items[self.selected_index]
            if not is_enabled(self.enabled_ids, item.full_id):
                return
            delta = -1 if reorder_up else 1
            new_index = self.enabled_ids.index(item.full_id) + delta
            if 0 <= new_index < len(self.enabled_ids):
                self.enabled_ids = move(self.enabled_ids, item.full_id, delta)
                self.is_dirty = True
                self.selected_index += delta
                self._refresh()
                self._notify_change()
            return

        if keybindings.matches(data, "tui.select.confirm"):
            if 0 <= self.selected_index < len(self.filtered_items):
                self.enabled_ids = toggle(self.enabled_ids, self.filtered_items[self.selected_index].full_id)
                self.is_dirty = True
                self._refresh()
                self._notify_change()
            return

        if keybindings.matches(data, "app.models.enableAll"):
            targets = [item.full_id for item in self.filtered_items] if self.search_input.get_value() else None
            self.enabled_ids = enable_all(self.enabled_ids, self.all_ids, targets)
            self.is_dirty = True
            self._refresh()
            self._notify_change()
            return

        if keybindings.matches(data, "app.models.clearAll"):
            targets = [item.full_id for item in self.filtered_items] if self.search_input.get_value() else None
            self.enabled_ids = clear_all(self.enabled_ids, self.all_ids, targets)
            self.is_dirty = True
            self._refresh()
            self._notify_change()
            return

        if keybindings.matches(data, "app.models.toggleProvider"):
            if 0 <= self.selected_index < len(self.filtered_items):
                item = self.filtered_items[self.selected_index]
                if item.model is not None:
                    provider = item.model.provider
                    provider_ids = [
                        model_id for model_id in self.all_ids if self.models_by_id[model_id].provider == provider
                    ]
                    if all(is_enabled(self.enabled_ids, model_id) for model_id in provider_ids):
                        self.enabled_ids = clear_all(self.enabled_ids, self.all_ids, provider_ids)
                    else:
                        self.enabled_ids = enable_all(self.enabled_ids, self.all_ids, provider_ids)
                    self.is_dirty = True
                    self._refresh()
                    self._notify_change()
            return

        if keybindings.matches(data, "app.models.save"):
            self.callbacks.on_persist(None if self.enabled_ids is None else list(self.enabled_ids))
            self.is_dirty = False
            self.footer_text.set_text(self._get_footer_text())
            return

        if matches_key(data, "ctrl+c"):
            if self.search_input.get_value():
                self.search_input.set_value("")
                self._refresh()
            else:
                self.callbacks.on_cancel()
            return

        if matches_key(data, "escape"):
            self.callbacks.on_cancel()
            return

        self.search_input.handle_input(data)
        self._refresh()


__all__ = [
    "EnabledIds",
    "ModelItem",
    "ModelsCallbacks",
    "ModelsConfig",
    "ScopedModelsSelectorComponent",
    "clear_all",
    "enable_all",
    "get_sorted_ids",
    "is_enabled",
    "move",
    "toggle",
]
