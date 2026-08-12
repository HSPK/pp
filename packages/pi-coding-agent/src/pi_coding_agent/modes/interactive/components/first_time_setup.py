"""First-run onboarding: theme choice and analytics opt-in.

Ported from ``packages/coding-agent/src/modes/interactive/components/first-time-setup.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pi_tui.component import Container
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings

from ....core.config import APP_NAME
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint

TerminalTheme = Literal["dark", "light"]

THEME_OPTIONS: tuple[tuple[TerminalTheme, str], ...] = (("dark", "Dark"), ("light", "Light"))
ANALYTICS_OPTIONS: tuple[tuple[bool, str], ...] = (
    (True, "Share anonymous usage data"),
    (False, "Don't share"),
)

SETUP_LOGO_LINES = ["██████", "██  ██", "████  ██", "██    ██"]

_ANALYTICS_BLURB = (
    "Opting in stores a tracking identifier in settings.json and enables anonymous\n"
    "usage analytics. This helps us to better debug, reproduce, and resolve issues\n"
    "and bugs within Pi. You can observe what is shared using /privacy and make\n"
    "changes anytime in settings.json."
)


@dataclass
class FirstTimeSetupResult:
    theme: TerminalTheme
    share_analytics: bool


class FirstTimeSetupComponent(Container):
    def __init__(
        self,
        detected_theme: TerminalTheme,
        on_theme_preview: Callable[[TerminalTheme], None],
        on_submit: Callable[[FirstTimeSetupResult], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        self.detected_theme = detected_theme
        self._on_theme_preview = on_theme_preview
        self._on_submit = on_submit
        self._on_cancel = on_cancel

        self.step: Literal["theme", "analytics"] = "theme"
        self.theme_index = max(
            0, next((i for i, (value, _) in enumerate(THEME_OPTIONS) if value == detected_theme), -1)
        )
        self.analytics_index = 0
        self._update()

    def _update(self) -> None:
        # Rebuilt in full on every change so a theme preview recolours everything.
        self.clear()
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", "\n".join(SETUP_LOGO_LINES)), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(
            Text(
                theme.fg("accent", theme.bold(f"Welcome to {APP_NAME}, the minimal coding agent.")),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))

        if self.step == "theme":
            self.add_child(Text(theme.fg("text", "Pick a theme."), 1, 0))
            self.add_child(Text(theme.fg("muted", f"Detected system appearance: {self.detected_theme}"), 1, 0))
            self.add_child(Spacer(1))
            self._add_option_list([label for _value, label in THEME_OPTIONS], self.theme_index)
        else:
            self.add_child(Text(theme.fg("text", "Opt-in to anonymous usage data sharing?"), 1, 0))
            self.add_child(Text(theme.fg("muted", _ANALYTICS_BLURB), 1, 0))
            self.add_child(Spacer(1))
            self._add_option_list([label for _value, label in ANALYTICS_OPTIONS], self.analytics_index)

        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "continue" if self.step == "theme" else "finish")
                + "  "
                + key_hint("tui.select.cancel", "skip setup"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    def _add_option_list(self, labels: list[str], selected_index: int) -> None:
        for index, label in enumerate(labels):
            is_selected = index == selected_index
            prefix = theme.fg("accent", "→ ") if is_selected else "  "
            styled = theme.fg("accent", label) if is_selected else theme.fg("text", label)
            self.add_child(Text(f"{prefix}{styled}", 1, 0))

    def _move_selection(self, delta: int) -> None:
        if self.step == "theme":
            next_index = max(0, min(len(THEME_OPTIONS) - 1, self.theme_index + delta))
            if next_index != self.theme_index:
                self.theme_index = next_index
                self._on_theme_preview(THEME_OPTIONS[self.theme_index][0])
        else:
            self.analytics_index = max(0, min(len(ANALYTICS_OPTIONS) - 1, self.analytics_index + delta))
        self._update()

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.select.up") or key_data == "k":
            self._move_selection(-1)
        elif keybindings.matches(key_data, "tui.select.down") or key_data == "j":
            self._move_selection(1)
        elif keybindings.matches(key_data, "tui.select.confirm") or key_data == "\n":
            if self.step == "theme":
                self.step = "analytics"
                self._update()
            else:
                self._on_submit(
                    FirstTimeSetupResult(
                        theme=THEME_OPTIONS[self.theme_index][0],
                        share_analytics=ANALYTICS_OPTIONS[self.analytics_index][0],
                    )
                )
        elif keybindings.matches(key_data, "tui.select.cancel"):
            self._on_cancel()


__all__ = [
    "ANALYTICS_OPTIONS",
    "SETUP_LOGO_LINES",
    "THEME_OPTIONS",
    "FirstTimeSetupComponent",
    "FirstTimeSetupResult",
    "TerminalTheme",
]
