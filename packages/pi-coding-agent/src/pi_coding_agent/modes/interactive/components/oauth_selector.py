"""Auth provider picker for ``/login`` and ``/logout``.

Ported from ``packages/coding-agent/src/modes/interactive/components/oauth-selector.ts``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pi_ai.auth.types import ApiKeyAuth, AuthCheck, OAuthAuth
from pi_tui.component import Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.truncated_text import TruncatedText
from pi_tui.fuzzy import fuzzy_filter
from pi_tui.keybindings import get_keybindings

from ..theme.theme import theme
from .dynamic_border import DynamicBorder

MAX_VISIBLE = 8
_ENV_VAR_LIST_RE = re.compile(r"^[A-Z][A-Z0-9_]*(?:, [A-Z][A-Z0-9_]*)*$")

AuthType = Literal["oauth", "api_key"]


@dataclass
class AuthSelectorProvider:
    id: str
    name: str
    auth_type: AuthType
    method: ApiKeyAuth | OAuthAuth | None = None
    status: AuthCheck | None = None


def format_auth_selector_provider_type(auth_type: AuthType) -> str:
    return "subscription" if auth_type == "oauth" else "API key"


class OAuthSelectorComponent(Container):
    def __init__(
        self,
        mode: Literal["login", "logout"],
        providers: list[AuthSelectorProvider],
        on_select: Callable[[str, AuthType], None],
        on_cancel: Callable[[], None],
        initial_search_input: str | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.all_providers = providers
        self.filtered_providers = providers
        self.selected_index = 0
        self.show_auth_type_labels = len({provider.auth_type for provider in providers}) > 1
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._focused = False

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        title = "Select provider to configure:" if mode == "login" else "Select provider to logout:"
        self.add_child(TruncatedText(theme.fg("accent", theme.bold(title)), 1, 0))
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

        self._filter_providers(initial_search_input or "")

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.search_input.focused = value

    def _submit_selected(self, _value: str = "") -> None:
        if 0 <= self.selected_index < len(self.filtered_providers):
            provider = self.filtered_providers[self.selected_index]
            self._on_select(provider.id, provider.auth_type)

    def _filter_providers(self, query: str) -> None:
        if query:
            self.filtered_providers = fuzzy_filter(
                self.all_providers,
                query,
                lambda provider: (
                    f"{provider.name} {provider.id} {provider.auth_type} {provider.method.name if provider.method else ''}"
                ),
            )
        else:
            self.filtered_providers = self.all_providers
        self.selected_index = max(0, min(self.selected_index, max(0, len(self.filtered_providers) - 1)))
        self._update_list()

    def _update_list(self) -> None:
        self.list_container.clear()

        start_index = max(
            0,
            min(
                self.selected_index - math.floor(MAX_VISIBLE / 2),
                len(self.filtered_providers) - MAX_VISIBLE,
            ),
        )
        end_index = min(start_index + MAX_VISIBLE, len(self.filtered_providers))

        for index in range(start_index, end_index):
            provider = self.filtered_providers[index]
            status_indicator = self._format_status_indicator(provider)
            auth_type_label = (
                theme.fg("muted", f" [{format_auth_selector_provider_type(provider.auth_type)}]")
                if self.show_auth_type_labels
                else ""
            )
            if index == self.selected_index:
                line = theme.fg("accent", "→ ") + theme.fg("accent", provider.name)
            else:
                line = f"  {theme.fg('text', provider.name)}"
            self.list_container.add_child(TruncatedText(line + auth_type_label + status_indicator, 1, 0))

        if start_index > 0 or end_index < len(self.filtered_providers):
            self.list_container.add_child(
                TruncatedText(theme.fg("muted", f"  ({self.selected_index + 1}/{len(self.filtered_providers)})"), 1, 0)
            )

        if len(self.filtered_providers) == 0:
            if len(self.all_providers) == 0:
                message = (
                    "No providers available" if self.mode == "login" else "No providers logged in. Use /login first."
                )
            else:
                message = "No matching providers"
            self.list_container.add_child(TruncatedText(theme.fg("muted", f"  {message}"), 1, 0))

    def _format_status_indicator(self, provider: AuthSelectorProvider) -> str:
        status = provider.status
        if not status:
            return theme.fg("muted", " • unconfigured")

        if status.type != provider.auth_type:
            label = "subscription configured" if status.type == "oauth" else "API key configured"
            return theme.fg("muted", " • ") + theme.fg("warning", label)

        source = status.source
        if not source or source in ("OAuth", "stored credential"):
            return theme.fg("success", " ✓ configured")

        display_source = f"env: {source}" if _ENV_VAR_LIST_RE.match(source) else source
        return theme.fg("success", f" ✓ {display_source}")

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.select.up"):
            if len(self.filtered_providers) == 0:
                return
            self.selected_index = max(0, self.selected_index - 1)
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.down"):
            if len(self.filtered_providers) == 0:
                return
            self.selected_index = min(len(self.filtered_providers) - 1, self.selected_index + 1)
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.confirm"):
            self._submit_selected()
        elif keybindings.matches(key_data, "tui.select.cancel"):
            self._on_cancel()
        else:
            self.search_input.handle_input(key_data)
            self._filter_providers(self.search_input.get_value())


__all__ = [
    "MAX_VISIBLE",
    "AuthSelectorProvider",
    "AuthType",
    "OAuthSelectorComponent",
    "format_auth_selector_provider_type",
]
