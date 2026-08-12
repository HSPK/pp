"""Tests ported from packages/tui/test/keybindings.test.ts."""

from pi_tui.keybindings import TUI_KEYBINDINGS, KeybindingConflict, KeybindingsManager


class TestKeybindingsManager:
    def test_binds_ctrl_j_as_default_newline_alias(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)

        assert keybindings.get_keys("tui.input.newLine") == ["shift+enter", "ctrl+j"]
        assert keybindings.matches("\n", "tui.input.newLine") is True
        assert keybindings.matches("\x1b[106;5u", "tui.input.newLine") is True

    def test_binds_modified_and_unmodified_editor_viewport_navigation(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)

        assert keybindings.get_keys("tui.editor.cursorLineStart") == ["home", "ctrl+home", "ctrl+a"]
        assert keybindings.get_keys("tui.editor.cursorLineEnd") == ["end", "ctrl+end", "ctrl+e"]
        assert keybindings.get_keys("tui.editor.pageUp") == ["pageUp", "ctrl+pageUp"]
        assert keybindings.get_keys("tui.editor.pageDown") == ["pageDown", "ctrl+pageDown"]

    def test_leaves_dedicated_prompt_history_navigation_unbound_by_default(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)

        assert keybindings.get_keys("tui.editor.historyPrevious") == []
        assert keybindings.get_keys("tui.editor.historyNext") == []

    def test_binds_unmodified_terminal_viewport_shortcuts_to_alt_screen_navigation(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)

        assert keybindings.get_keys("tui.altScreen.pageUp") == ["pageUp"]
        assert keybindings.get_keys("tui.altScreen.pageDown") == ["pageDown"]
        assert keybindings.get_keys("tui.altScreen.halfPageUp") == []
        assert keybindings.get_keys("tui.altScreen.halfPageDown") == []
        assert keybindings.get_keys("tui.altScreen.previousPrompt") == ["ctrl+shift+up"]
        assert keybindings.get_keys("tui.altScreen.nextPrompt") == ["ctrl+shift+down"]
        assert keybindings.get_keys("tui.altScreen.top") == ["home"]
        assert keybindings.get_keys("tui.altScreen.bottom") == ["end"]

    def test_does_not_evict_selector_confirm_when_input_submit_is_rebound(self):
        keybindings = KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.input.submit": ["enter", "ctrl+enter"]},
        )

        assert keybindings.get_keys("tui.input.submit") == ["enter", "ctrl+enter"]
        assert keybindings.get_keys("tui.select.confirm") == ["enter"]

    def test_does_not_evict_cursor_bindings_when_another_action_reuses_same_key(self):
        keybindings = KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.select.up": ["up", "ctrl+p"]},
        )

        assert keybindings.get_keys("tui.select.up") == ["up", "ctrl+p"]
        assert keybindings.get_keys("tui.editor.cursorUp") == ["up"]

    def test_still_reports_direct_user_binding_conflicts_without_evicting_defaults(self):
        keybindings = KeybindingsManager(
            TUI_KEYBINDINGS,
            {
                "tui.input.submit": "ctrl+x",
                "tui.select.confirm": "ctrl+x",
            },
        )

        assert keybindings.get_conflicts() == [
            KeybindingConflict("ctrl+x", ["tui.input.submit", "tui.select.confirm"]),
        ]
        assert keybindings.get_keys("tui.editor.cursorLeft") == ["left", "ctrl+b"]

    def test_ignores_user_bindings_for_unknown_keybinding_ids(self):
        keybindings = KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.does.not.exist": "ctrl+z"},
        )

        assert keybindings.get_conflicts() == []
        assert keybindings.get_keys("tui.editor.cursorUp") == ["up"]

    def test_deduplicates_repeated_keys_in_a_single_binding(self):
        keybindings = KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.editor.undo": ["ctrl+-", "ctrl+-", "ctrl+z"]},
        )

        assert keybindings.get_keys("tui.editor.undo") == ["ctrl+-", "ctrl+z"]

    def test_get_definition_returns_the_registered_definition(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)

        definition = keybindings.get_definition("tui.editor.cursorUp")

        assert definition.default_keys == "up"
        assert definition.description == "Move cursor up"

    def test_set_user_bindings_rebuilds_keys_and_conflicts(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)
        assert keybindings.get_keys("tui.editor.cursorUp") == ["up"]

        keybindings.set_user_bindings({"tui.editor.cursorUp": "ctrl+p"})

        assert keybindings.get_keys("tui.editor.cursorUp") == ["ctrl+p"]
        assert keybindings.get_user_bindings() == {"tui.editor.cursorUp": "ctrl+p"}

    def test_get_user_bindings_returns_a_copy(self):
        user_bindings = {"tui.editor.cursorUp": "ctrl+p"}
        keybindings = KeybindingsManager(TUI_KEYBINDINGS, user_bindings)

        result = keybindings.get_user_bindings()
        result["tui.editor.cursorUp"] = "ctrl+z"

        assert keybindings.get_user_bindings() == {"tui.editor.cursorUp": "ctrl+p"}

    def test_get_resolved_bindings_returns_scalar_for_single_key_and_list_otherwise(self):
        keybindings = KeybindingsManager(TUI_KEYBINDINGS)

        resolved = keybindings.get_resolved_bindings()

        assert resolved["tui.editor.cursorUp"] == "up"
        assert resolved["tui.editor.cursorLeft"] == ["left", "ctrl+b"]
        assert resolved["tui.editor.historyPrevious"] == []
