"""Python port of `packages/coding-agent/test/settings-selector.test.ts`."""

from __future__ import annotations

import pytest
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.settings_selector import (
    SettingsCallbacks,
    SettingsConfig,
    SettingsSelectorComponent,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_tui.keybindings import set_keybindings


@pytest.fixture(autouse=True)
def _theme_and_keybindings():
    init_theme("dark")
    set_keybindings(KeybindingsManager())


def test_cycles_through_fullscreen_settings():
    exit_output_calls: list[str] = []
    scrollbar_calls: list[str] = []
    config = SettingsConfig(
        fullscreen_exit_output="transcript",
        fullscreen_scrollbar="auto",
        warnings={},
        available_thinking_levels=[],
        available_themes=[],
    )
    callbacks = SettingsCallbacks(
        on_fullscreen_exit_output_change=exit_output_calls.append,
        on_fullscreen_scrollbar_change=scrollbar_calls.append,
    )

    def cycle(label: str, count: int) -> None:
        settings_list = SettingsSelectorComponent(config, callbacks).get_settings_list()
        for character in label:
            settings_list.handle_input(character)
        for _ in range(count):
            settings_list.handle_input("\r")

    cycle("Fullscreen exit output", 2)
    assert exit_output_calls == ["resume-hint", "transcript"]

    cycle("Fullscreen scrollbar", 3)
    assert scrollbar_calls == ["always", "hidden", "auto"]
