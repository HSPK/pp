"""Project trust dialog.

Ported from ``packages/coding-agent/src/modes/interactive/components/trust-selector.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pi_tui.component import Container
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings

from ....core.trust_manager import (
    ProjectTrustOption,
    ProjectTrustStoreEntry,
    ProjectTrustUpdate,
    get_project_trust_options,
)
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


@dataclass
class TrustSelection:
    trusted: bool
    updates: list[ProjectTrustUpdate] = field(default_factory=list)


def format_decision(trust_path: str | None, decision: ProjectTrustStoreEntry | None) -> str:
    if decision is None:
        return "none"
    label = "trusted" if decision.decision else "untrusted"
    if trust_path is not None and decision.path != trust_path:
        return f"{label} (inherited from {decision.path})"
    return f"{label} ({decision.path})"


class TrustSelectorComponent(Container):
    def __init__(
        self,
        cwd: str,
        saved_decision: ProjectTrustStoreEntry | None,
        project_trusted: bool,
        on_select: Callable[[TrustSelection], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        self.saved_decision = saved_decision
        self.trust_options = get_project_trust_options(cwd)
        self.selected_index = max(
            0,
            next((i for i, option in enumerate(self.trust_options) if self._is_saved_option(option)), -1),
        )
        self._on_select = on_select
        self._on_cancel = on_cancel

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", theme.bold("Project trust")), 1, 0))
        self.add_child(Text(theme.fg("muted", cwd), 1, 0))
        self.add_child(Spacer(1))
        first_saved_path = self.trust_options[0].saved_path if self.trust_options else None
        self.add_child(
            Text(
                theme.fg("muted", f"Saved decision: {format_decision(first_saved_path, saved_decision)}"),
                1,
                0,
            )
        )
        self.add_child(
            Text(
                theme.fg("muted", f"Current session: {'trusted' if project_trusted else 'untrusted'}"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))

        self.list_container = Container()
        self.add_child(self.list_container)
        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "save")
                + "  "
                + key_hint("tui.select.cancel", "cancel"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        self._update_list()

    def _is_saved_option(self, option: ProjectTrustOption) -> bool:
        return (
            option.saved_path is not None
            and self.saved_decision is not None
            and self.saved_decision.decision == option.trusted
            and self.saved_decision.path == option.saved_path
        )

    def _update_list(self) -> None:
        self.list_container.clear()
        for index, option in enumerate(self.trust_options):
            is_selected = index == self.selected_index
            checkmark = theme.fg("success", " ✓") if self._is_saved_option(option) else ""
            prefix = theme.fg("accent", "→ ") if is_selected else "  "
            label = theme.fg("accent", option.label) if is_selected else theme.fg("text", option.label)
            self.list_container.add_child(Text(f"{prefix}{label}{checkmark}", 1, 0))

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.select.up") or key_data == "k":
            self.selected_index = max(0, self.selected_index - 1)
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.down") or key_data == "j":
            self.selected_index = min(len(self.trust_options) - 1, self.selected_index + 1)
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.confirm") or key_data == "\n":
            if 0 <= self.selected_index < len(self.trust_options):
                selected = self.trust_options[self.selected_index]
                self._on_select(TrustSelection(trusted=selected.trusted, updates=selected.updates))
        elif keybindings.matches(key_data, "tui.select.cancel"):
            self._on_cancel()


__all__ = ["TrustSelection", "TrustSelectorComponent", "format_decision"]
