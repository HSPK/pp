"""Python port of `packages/coding-agent/test/custom-editor-history-keybindings.test.ts`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.custom_editor import CustomEditor
from pi_tui.components.editor import EditorTheme
from pi_tui.components.select_list import SelectListTheme
from pi_tui.keybindings import get_keybindings, set_keybindings
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen


def _identity(text: str) -> str:
    return text


DEFAULT_EDITOR_THEME = EditorTheme(
    border_color=_identity,
    select_list=SelectListTheme(
        selected_prefix=_identity,
        selected_text=_identity,
        description=_identity,
        scroll_info=_identity,
        no_match=_identity,
    ),
)


@pytest.fixture(autouse=True)
def _restore_keybindings() -> Iterator[None]:
    previous = get_keybindings()
    yield
    set_keybindings(previous)


def test_explicit_history_binding_wins_over_model_cycling() -> None:
    keybindings = KeybindingsManager(
        {
            "tui.editor.historyPrevious": "ctrl+p",
            "tui.editor.historyNext": "ctrl+n",
        }
    )
    set_keybindings(keybindings)
    editor = CustomEditor(TuiMainScreen(FakeTerminal()), DEFAULT_EDITOR_THEME, keybindings)
    model_cycles = 0

    def on_cycle() -> None:
        nonlocal model_cycles
        model_cycles += 1

    editor.on_action("app.model.cycleForward", on_cycle)
    editor.add_to_history("previous prompt")
    editor.set_text("draft")

    editor.handle_input("\x10")  # Ctrl+P
    assert editor.get_text() == "previous prompt"
    assert model_cycles == 0

    editor.handle_input("\x0e")  # Ctrl+N
    assert editor.get_text() == "draft"
