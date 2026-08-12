"""Model picker with fuzzy search and an all/scoped toggle.

Ported from ``packages/coding-agent/src/modes/interactive/components/model-selector.ts``.

The TypeScript version kicks off an asynchronous remote catalog refresh with a
15s timeout and an abort controller. This port's :class:`ModelRuntime` is
local-only and synchronous (see its module docstring: no remote catalog
provider), so the refresh happens inline and the timeout/abort machinery has
nothing to drive. The status and error message paths are kept so the UI reads
the same, and `dispose()` still exists for symmetry with the TypeScript.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pi_ai.models import Model, models_are_equal
from pi_tui.component import Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.fuzzy import fuzzy_filter
from pi_tui.keybindings import get_keybindings

from ..model_search import ModelSearchItem, get_model_selector_search_text
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint

if TYPE_CHECKING:
    from ....core.model_runtime import ModelRuntime
    from ....core.settings_manager import SettingsManager

MAX_VISIBLE = 10


@dataclass
class ModelItem:
    provider: str
    id: str
    model: Model


@dataclass
class ScopedModelItem:
    model: Model
    thinking_level: str | None = None


class ModelSelectorComponent(Container):
    def __init__(
        self,
        tui: Any,
        current_model: Model | None,
        settings_manager: SettingsManager,
        model_runtime: ModelRuntime,
        scoped_models: Sequence[ScopedModelItem],
        on_select: Callable[[Model], None],
        on_cancel: Callable[[], None],
        initial_search_input: str | None = None,
    ) -> None:
        super().__init__()
        self.tui = tui
        self.current_model = current_model
        self.settings_manager = settings_manager
        self.model_runtime = model_runtime
        self.scoped_models: list[ScopedModelItem] = list(scoped_models)
        self.scope = "scoped" if len(self.scoped_models) > 0 else "all"
        self._on_select = on_select
        self._on_cancel = on_cancel

        self.all_models: list[ModelItem] = []
        self.scoped_model_items: list[ModelItem] = []
        self.active_models: list[ModelItem] = []
        self.filtered_models: list[ModelItem] = []
        self.selected_index = 0
        self.error_message: str | None = None
        self.refresh_status_message = "Refreshing model catalogs…"
        self.refresh_status_success = False
        self.closed = False
        self._focused = False

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        self.scope_text: Text | None = None
        self.scope_hint_text: Text | None = None
        if len(self.scoped_models) > 0:
            self.scope_text = Text(self._get_scope_text(), 0, 0)
            self.add_child(self.scope_text)
            self.scope_hint_text = Text(self._get_scope_hint_text(), 0, 0)
            self.add_child(self.scope_hint_text)
        else:
            self.add_child(
                Text(
                    theme.fg(
                        "warning",
                        "Only showing models from configured providers. Use /login to add providers.",
                    ),
                    0,
                    0,
                )
            )
        self.add_child(Spacer(1))

        self.search_input = Input()
        if initial_search_input:
            self.search_input.set_value(initial_search_input)
        self.search_input.on_submit = self._submit_selected
        self.add_child(self.search_input)
        self.add_child(Spacer(1))

        self.list_container = Container()
        self.add_child(self.list_container)
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        self._load_models_from_snapshot()
        if initial_search_input:
            self._filter_models(initial_search_input)
        else:
            self._update_list()
        if self.tui is not None:
            self.tui.request_render()
        self.refresh_models()

    # -- Focusable ----------------------------------------------------------

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.search_input.focused = value

    # -- data ---------------------------------------------------------------

    def _submit_selected(self, _value: str = "") -> None:
        if 0 <= self.selected_index < len(self.filtered_models):
            self._handle_select(self.filtered_models[self.selected_index].model)

    def _load_models_from_snapshot(self) -> None:
        models = [
            ModelItem(provider=model.provider, id=model.id, model=model)
            for model in self.model_runtime.get_available_snapshot()
        ]
        self.all_models = self._sort_models(models)

        refreshed_scoped: list[ScopedModelItem] = []
        for scoped in self.scoped_models:
            refreshed = self.model_runtime.get_model(scoped.model.provider, scoped.model.id)
            refreshed_scoped.append(
                ScopedModelItem(model=refreshed, thinking_level=scoped.thinking_level) if refreshed else scoped
            )
        self.scoped_models = refreshed_scoped
        self.scoped_model_items = [
            ModelItem(provider=scoped.model.provider, id=scoped.model.id, model=scoped.model)
            for scoped in self.scoped_models
        ]

        self.active_models = self.scoped_model_items if self.scope == "scoped" else self.all_models
        self.filtered_models = self.active_models
        current_index = self._index_of_current(self.filtered_models)
        self.selected_index = (
            current_index if current_index >= 0 else min(self.selected_index, max(0, len(self.filtered_models) - 1))
        )

    def _index_of_current(self, items: list[ModelItem]) -> int:
        return next((i for i, item in enumerate(items) if models_are_equal(self.current_model, item.model)), -1)

    def refresh_models(self) -> None:
        try:
            self.model_runtime.refresh()
        except Exception as error:
            self.refresh_status_message = ""
            self.error_message = f"Could not refresh model catalogs: {error}"
            self._update_list()
            if self.tui is not None:
                self.tui.request_render()
            return

        if self.closed:
            return
        self.refresh_status_message = ""
        self.error_message = getattr(self.model_runtime, "get_error", lambda: None)()
        if not self.error_message:
            self.refresh_status_message = "Model catalogs refreshed."
            self.refresh_status_success = True
        self._load_models_from_snapshot()
        self._filter_models(self.search_input.get_value())
        if self.tui is not None:
            self.tui.request_render()

    def dispose(self) -> None:
        self.closed = True

    def _sort_models(self, models: list[ModelItem]) -> list[ModelItem]:
        # Current model first, then by provider name. TS uses `localeCompare`;
        # every provider ID here is ASCII, so a plain comparison matches.
        return sorted(
            models,
            key=lambda item: (0 if models_are_equal(self.current_model, item.model) else 1, item.provider),
        )

    # -- scope --------------------------------------------------------------

    def _get_scope_text(self) -> str:
        all_text = theme.fg("accent" if self.scope == "all" else "muted", "all")
        scoped_text = theme.fg("accent" if self.scope == "scoped" else "muted", "scoped")
        return f"{theme.fg('muted', 'Scope: ')}{all_text}{theme.fg('muted', ' | ')}{scoped_text}"

    def _get_scope_hint_text(self) -> str:
        return key_hint("tui.input.tab", "scope") + theme.fg("muted", " (all/scoped)")

    def _set_scope(self, scope: str) -> None:
        if self.scope == scope:
            return
        self.scope = scope
        self.active_models = self.scoped_model_items if self.scope == "scoped" else self.all_models
        current_index = self._index_of_current(self.active_models)
        self.selected_index = current_index if current_index >= 0 else 0
        self._filter_models(self.search_input.get_value())
        if self.scope_text is not None:
            self.scope_text.set_text(self._get_scope_text())

    # -- filtering / rendering ---------------------------------------------

    def _filter_models(self, query: str) -> None:
        if query:
            self.filtered_models = fuzzy_filter(
                self.active_models,
                query,
                lambda item: get_model_selector_search_text(
                    ModelSearchItem(id=item.id, provider=item.provider, name=item.model.name)
                ),
            )
        else:
            self.filtered_models = self.active_models
        # A query moves the cursor to the best match; clearing it keeps the
        # position, clamped to the restored list.
        self.selected_index = 0 if query else min(self.selected_index, max(0, len(self.filtered_models) - 1))
        self._update_list()

    def _update_list(self) -> None:
        self.list_container.clear()

        start_index = max(
            0,
            min(
                self.selected_index - math.floor(MAX_VISIBLE / 2),
                len(self.filtered_models) - MAX_VISIBLE,
            ),
        )
        end_index = min(start_index + MAX_VISIBLE, len(self.filtered_models))

        for index in range(start_index, end_index):
            item = self.filtered_models[index]
            is_selected = index == self.selected_index
            is_current = models_are_equal(self.current_model, item.model)
            provider_badge = theme.fg("muted", f"[{item.provider}]")
            checkmark = theme.fg("success", " ✓") if is_current else ""
            if is_selected:
                line = f"{theme.fg('accent', '→ ') + theme.fg('accent', item.id)} {provider_badge}{checkmark}"
            else:
                line = f"  {item.id} {provider_badge}{checkmark}"
            self.list_container.add_child(Text(line, 0, 0))

        if start_index > 0 or end_index < len(self.filtered_models):
            self.list_container.add_child(
                Text(theme.fg("muted", f"  ({self.selected_index + 1}/{len(self.filtered_models)})"), 0, 0)
            )

        if self.error_message:
            for line in self.error_message.split("\n"):
                self.list_container.add_child(Text(theme.fg("error", line), 0, 0))
        elif len(self.filtered_models) == 0:
            self.list_container.add_child(Text(theme.fg("muted", "  No matching models"), 0, 0))
        else:
            selected = self.filtered_models[self.selected_index]
            self.list_container.add_child(Spacer(1))
            self.list_container.add_child(Text(theme.fg("muted", f"  Model Name: {selected.model.name}"), 0, 0))

        if self.refresh_status_message:
            self.list_container.add_child(Spacer(1))
            self.list_container.add_child(
                Text(
                    theme.fg(
                        "success" if self.refresh_status_success else "muted",
                        f"  {self.refresh_status_message}",
                    ),
                    0,
                    0,
                )
            )

    # -- input --------------------------------------------------------------

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.input.tab"):
            if len(self.scoped_model_items) > 0:
                self._set_scope("scoped" if self.scope == "all" else "all")
                if self.scope_hint_text is not None:
                    self.scope_hint_text.set_text(self._get_scope_hint_text())
            return

        if keybindings.matches(key_data, "tui.select.up"):
            if len(self.filtered_models) == 0:
                return
            self.selected_index = len(self.filtered_models) - 1 if self.selected_index == 0 else self.selected_index - 1
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.down"):
            if len(self.filtered_models) == 0:
                return
            self.selected_index = 0 if self.selected_index == len(self.filtered_models) - 1 else self.selected_index + 1
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.confirm"):
            self._submit_selected()
        elif keybindings.matches(key_data, "tui.select.cancel"):
            self.dispose()
            self._on_cancel()
        else:
            self.search_input.handle_input(key_data)
            self._filter_models(self.search_input.get_value())

    def _handle_select(self, model: Model) -> None:
        self.dispose()
        self.settings_manager.set_default_model_and_provider(model.provider, model.id)
        self._on_select(model)

    def get_search_input(self) -> Input:
        return self.search_input


__all__ = ["MAX_VISIBLE", "ModelItem", "ModelSelectorComponent", "ScopedModelItem"]
