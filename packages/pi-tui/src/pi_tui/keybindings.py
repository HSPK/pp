"""Keybinding registry for `packages/tui/src/keybindings.ts`.

`Keybinding` is a plain `str` type alias here (rather than a TypeScript
`keyof Keybindings` union produced via declaration merging): the module still
exposes `TUI_KEYBINDINGS` as the canonical set of built-in keybinding IDs, but
`KeybindingsManager` accepts arbitrary string IDs so downstream packages can
register their own without needing a Python equivalent of TS interface
merging.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_tui.keys import KeyId, matches_key

Keybinding = str


@dataclass
class KeybindingDefinition:
    default_keys: KeyId | list[KeyId]
    description: str | None = None


KeybindingDefinitions = dict[str, KeybindingDefinition]
KeybindingsConfig = dict[str, "KeyId | list[KeyId] | None"]


TUI_KEYBINDINGS: KeybindingDefinitions = {
    "tui.editor.cursorUp": KeybindingDefinition("up", "Move cursor up"),
    "tui.editor.cursorDown": KeybindingDefinition("down", "Move cursor down"),
    "tui.editor.historyPrevious": KeybindingDefinition([], "Select previous prompt history entry"),
    "tui.editor.historyNext": KeybindingDefinition([], "Select next prompt history entry"),
    "tui.editor.cursorLeft": KeybindingDefinition(["left", "ctrl+b"], "Move cursor left"),
    "tui.editor.cursorRight": KeybindingDefinition(["right", "ctrl+f"], "Move cursor right"),
    "tui.editor.cursorWordLeft": KeybindingDefinition(["alt+left", "ctrl+left", "alt+b"], "Move cursor word left"),
    "tui.editor.cursorWordRight": KeybindingDefinition(["alt+right", "ctrl+right", "alt+f"], "Move cursor word right"),
    "tui.editor.cursorLineStart": KeybindingDefinition(["home", "ctrl+home", "ctrl+a"], "Move to line start"),
    "tui.editor.cursorLineEnd": KeybindingDefinition(["end", "ctrl+end", "ctrl+e"], "Move to line end"),
    "tui.editor.jumpForward": KeybindingDefinition("ctrl+]", "Jump forward to character"),
    "tui.editor.jumpBackward": KeybindingDefinition("ctrl+alt+]", "Jump backward to character"),
    "tui.editor.pageUp": KeybindingDefinition(["pageUp", "ctrl+pageUp"], "Page up"),
    "tui.editor.pageDown": KeybindingDefinition(["pageDown", "ctrl+pageDown"], "Page down"),
    "tui.editor.deleteCharBackward": KeybindingDefinition("backspace", "Delete character backward"),
    "tui.editor.deleteCharForward": KeybindingDefinition(["delete", "ctrl+d"], "Delete character forward"),
    "tui.editor.deleteWordBackward": KeybindingDefinition(["ctrl+w", "alt+backspace"], "Delete word backward"),
    "tui.editor.deleteWordForward": KeybindingDefinition(["alt+d", "alt+delete"], "Delete word forward"),
    "tui.editor.deleteToLineStart": KeybindingDefinition("ctrl+u", "Delete to line start"),
    "tui.editor.deleteToLineEnd": KeybindingDefinition("ctrl+k", "Delete to line end"),
    "tui.editor.yank": KeybindingDefinition("ctrl+y", "Yank"),
    "tui.editor.yankPop": KeybindingDefinition("alt+y", "Yank pop"),
    "tui.editor.undo": KeybindingDefinition("ctrl+-", "Undo"),
    "tui.input.newLine": KeybindingDefinition(["shift+enter", "ctrl+j"], "Insert newline"),
    "tui.input.submit": KeybindingDefinition("enter", "Submit input"),
    "tui.input.tab": KeybindingDefinition("tab", "Tab / autocomplete"),
    "tui.input.copy": KeybindingDefinition("ctrl+c", "Copy selection"),
    "tui.select.up": KeybindingDefinition("up", "Move selection up"),
    "tui.select.down": KeybindingDefinition("down", "Move selection down"),
    "tui.select.pageUp": KeybindingDefinition("pageUp", "Selection page up"),
    "tui.select.pageDown": KeybindingDefinition("pageDown", "Selection page down"),
    "tui.select.confirm": KeybindingDefinition("enter", "Confirm selection"),
    "tui.select.cancel": KeybindingDefinition(["escape", "ctrl+c"], "Cancel selection"),
    # These intentionally shadow the unmodified editor bindings in fullscreen mode.
    "tui.altScreen.pageUp": KeybindingDefinition("pageUp", "Scroll viewport up one page"),
    "tui.altScreen.pageDown": KeybindingDefinition("pageDown", "Scroll viewport down one page"),
    "tui.altScreen.halfPageUp": KeybindingDefinition([], "Scroll viewport up half a page"),
    "tui.altScreen.halfPageDown": KeybindingDefinition([], "Scroll viewport down half a page"),
    # Unbound by default, like upstream: they exist so a user can bind
    # single-line scrolling, which no default key occupies.
    "tui.altScreen.lineUp": KeybindingDefinition([], "Scroll viewport up one line"),
    "tui.altScreen.lineDown": KeybindingDefinition([], "Scroll viewport down one line"),
    "tui.altScreen.previousPrompt": KeybindingDefinition("ctrl+shift+up", "Jump to previous semantic prompt"),
    "tui.altScreen.nextPrompt": KeybindingDefinition("ctrl+shift+down", "Jump to next semantic prompt"),
    "tui.altScreen.top": KeybindingDefinition("home", "Scroll viewport to top"),
    "tui.altScreen.bottom": KeybindingDefinition("end", "Scroll viewport to bottom"),
}


@dataclass
class KeybindingConflict:
    key: KeyId
    keybindings: list[str]


def _normalize_keys(keys: KeyId | list[KeyId] | None) -> list[KeyId]:
    if keys is None:
        return []
    key_list = keys if isinstance(keys, list) else [keys]
    seen: set[KeyId] = set()
    result: list[KeyId] = []
    for key in key_list:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


class KeybindingsManager:
    def __init__(self, definitions: KeybindingDefinitions, user_bindings: KeybindingsConfig | None = None) -> None:
        self._definitions = definitions
        self._user_bindings: KeybindingsConfig = user_bindings if user_bindings is not None else {}
        self._keys_by_id: dict[Keybinding, list[KeyId]] = {}
        self._conflicts: list[KeybindingConflict] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._keys_by_id.clear()
        self._conflicts = []

        user_claims: dict[KeyId, dict[Keybinding, None]] = {}
        for keybinding, keys in self._user_bindings.items():
            if keybinding not in self._definitions:
                continue
            for key in _normalize_keys(keys):
                claimants = user_claims.setdefault(key, {})
                claimants[keybinding] = None

        for key, keybindings in user_claims.items():
            if len(keybindings) > 1:
                self._conflicts.append(KeybindingConflict(key, list(keybindings)))

        for id_, definition in self._definitions.items():
            user_keys = self._user_bindings.get(id_)
            keys = _normalize_keys(definition.default_keys) if user_keys is None else _normalize_keys(user_keys)
            self._keys_by_id[id_] = keys

    def matches(self, data: str, keybinding: Keybinding) -> bool:
        keys = self._keys_by_id.get(keybinding, [])
        return any(matches_key(data, key) for key in keys)

    def get_keys(self, keybinding: Keybinding) -> list[KeyId]:
        return list(self._keys_by_id.get(keybinding, []))

    def get_definition(self, keybinding: Keybinding) -> KeybindingDefinition:
        return self._definitions[keybinding]

    def get_conflicts(self) -> list[KeybindingConflict]:
        return [KeybindingConflict(c.key, list(c.keybindings)) for c in self._conflicts]

    def set_user_bindings(self, user_bindings: KeybindingsConfig) -> None:
        self._user_bindings = user_bindings
        self._rebuild()

    def get_user_bindings(self) -> KeybindingsConfig:
        return dict(self._user_bindings)

    def get_resolved_bindings(self) -> KeybindingsConfig:
        resolved: KeybindingsConfig = {}
        for id_ in self._definitions:
            keys = self._keys_by_id.get(id_, [])
            resolved[id_] = keys[0] if len(keys) == 1 else list(keys)
        return resolved


_global_keybindings: KeybindingsManager | None = None


def set_keybindings(keybindings: KeybindingsManager) -> None:
    global _global_keybindings
    _global_keybindings = keybindings


def get_keybindings() -> KeybindingsManager:
    global _global_keybindings
    if _global_keybindings is None:
        _global_keybindings = KeybindingsManager(TUI_KEYBINDINGS)
    return _global_keybindings
