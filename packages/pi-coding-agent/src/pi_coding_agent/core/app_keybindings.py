"""Application keybindings.

Python port of `packages/coding-agent/src/core/keybindings.ts`. Extends the TUI
keybinding definitions with the application-level bindings and loads user
overrides from `keybindings.json`, migrating the legacy flat names.

Key strings and descriptions are user-visible and are copied verbatim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pi_tui.keybindings import TUI_KEYBINDINGS, KeybindingDefinition
from pi_tui.keybindings import KeybindingsManager as TuiKeybindingsManager

from .config import get_agent_dir

_IS_WINDOWS = sys.platform == "win32"
_IS_MACOS = sys.platform == "darwin"


def _definition(default_keys: Any, description: str) -> KeybindingDefinition:
    return KeybindingDefinition(default_keys, description)


APP_KEYBINDINGS: dict[str, KeybindingDefinition] = {
    "app.interrupt": _definition("escape", "Cancel or abort"),
    "app.clear": _definition("ctrl+c", "Clear editor"),
    "app.exit": _definition("ctrl+d", "Exit when editor is empty"),
    "app.suspend": _definition([] if _IS_WINDOWS else "ctrl+z", "Suspend to background"),
    "app.thinking.cycle": _definition("shift+tab", "Cycle thinking level"),
    "app.model.cycleForward": _definition("ctrl+p", "Cycle to next model"),
    "app.model.cycleBackward": _definition("shift+ctrl+p", "Cycle to previous model"),
    "app.model.select": _definition("ctrl+l", "Open model selector"),
    "app.tools.expand": _definition("ctrl+o", "Toggle tool output"),
    "app.thinking.toggle": _definition("ctrl+t", "Toggle thinking blocks"),
    "app.session.toggleNamedFilter": _definition("ctrl+n", "Toggle named session filter"),
    "app.editor.external": _definition("ctrl+g", "Open external editor"),
    "app.message.copy": _definition("ctrl+x", "Copy message to clipboard"),
    "app.message.followUp": _definition("alt+enter", "Queue follow-up message"),
    "app.message.dequeue": _definition("alt+up", "Restore queued messages"),
    "app.clipboard.pasteImage": _definition(
        "alt+v" if _IS_WINDOWS else "ctrl+v", "Paste image from clipboard (text fallback)"
    ),
    "app.session.new": _definition([], "Start a new session"),
    "app.session.tree": _definition([], "Open session tree"),
    "app.session.fork": _definition([], "Fork current session"),
    "app.session.resume": _definition([], "Resume a session"),
    "app.tree.foldOrUp": _definition(
        ["alt+left", "ctrl+left"] if _IS_MACOS else ["ctrl+left", "alt+left"],
        "Fold tree branch or move up",
    ),
    "app.tree.unfoldOrDown": _definition(
        ["alt+right", "ctrl+right"] if _IS_MACOS else ["ctrl+right", "alt+right"],
        "Unfold tree branch or move down",
    ),
    "app.tree.editLabel": _definition("shift+l", "Edit tree label"),
    "app.tree.toggleLabelTimestamp": _definition("shift+t", "Toggle tree label timestamps"),
    "app.session.togglePath": _definition("ctrl+p", "Toggle session path display"),
    "app.session.toggleSort": _definition("ctrl+s", "Toggle session sort mode"),
    "app.session.rename": _definition("ctrl+r", "Rename session"),
    "app.session.delete": _definition("ctrl+d", "Delete session"),
    "app.session.deleteNoninvasive": _definition("ctrl+backspace", "Delete session when query is empty"),
    "app.models.save": _definition("ctrl+s", "Save model selection"),
    "app.models.enableAll": _definition("ctrl+a", "Enable all models"),
    "app.models.clearAll": _definition("ctrl+x", "Clear all models"),
    "app.models.toggleProvider": _definition("ctrl+p", "Toggle all models for provider"),
    "app.models.reorderUp": _definition("alt+up", "Move model up in order"),
    "app.models.reorderDown": _definition("alt+down", "Move model down in order"),
    "app.tree.filter.default": _definition("ctrl+d", "Tree filter: default view"),
    "app.tree.filter.noTools": _definition("ctrl+t", "Tree filter: hide tool results"),
    "app.tree.filter.userOnly": _definition("ctrl+u", "Tree filter: user messages only"),
    "app.tree.filter.labeledOnly": _definition("ctrl+l", "Tree filter: labeled entries only"),
    "app.tree.filter.all": _definition("ctrl+a", "Tree filter: show all entries"),
    "app.tree.filter.cycleForward": _definition("ctrl+o", "Tree filter: cycle forward"),
    "app.tree.filter.cycleBackward": _definition("shift+ctrl+o", "Tree filter: cycle backward"),
}

KEYBINDINGS: dict[str, KeybindingDefinition] = {**TUI_KEYBINDINGS, **APP_KEYBINDINGS}

KEYBINDING_NAME_MIGRATIONS: dict[str, str] = {
    "cursorUp": "tui.editor.cursorUp",
    "cursorDown": "tui.editor.cursorDown",
    "cursorLeft": "tui.editor.cursorLeft",
    "cursorRight": "tui.editor.cursorRight",
    "cursorWordLeft": "tui.editor.cursorWordLeft",
    "cursorWordRight": "tui.editor.cursorWordRight",
    "cursorLineStart": "tui.editor.cursorLineStart",
    "cursorLineEnd": "tui.editor.cursorLineEnd",
    "jumpForward": "tui.editor.jumpForward",
    "jumpBackward": "tui.editor.jumpBackward",
    "pageUp": "tui.editor.pageUp",
    "pageDown": "tui.editor.pageDown",
    "deleteCharBackward": "tui.editor.deleteCharBackward",
    "deleteCharForward": "tui.editor.deleteCharForward",
    "deleteWordBackward": "tui.editor.deleteWordBackward",
    "deleteWordForward": "tui.editor.deleteWordForward",
    "deleteToLineStart": "tui.editor.deleteToLineStart",
    "deleteToLineEnd": "tui.editor.deleteToLineEnd",
    "yank": "tui.editor.yank",
    "yankPop": "tui.editor.yankPop",
    "undo": "tui.editor.undo",
    "newLine": "tui.input.newLine",
    "submit": "tui.input.submit",
    "tab": "tui.input.tab",
    "copy": "tui.input.copy",
    "selectUp": "tui.select.up",
    "selectDown": "tui.select.down",
    "selectPageUp": "tui.select.pageUp",
    "selectPageDown": "tui.select.pageDown",
    "selectConfirm": "tui.select.confirm",
    "selectCancel": "tui.select.cancel",
    "interrupt": "app.interrupt",
    "clear": "app.clear",
    "exit": "app.exit",
    "suspend": "app.suspend",
    "cycleThinkingLevel": "app.thinking.cycle",
    "cycleModelForward": "app.model.cycleForward",
    "cycleModelBackward": "app.model.cycleBackward",
    "selectModel": "app.model.select",
    "expandTools": "app.tools.expand",
    "toggleThinking": "app.thinking.toggle",
    "toggleSessionNamedFilter": "app.session.toggleNamedFilter",
    "externalEditor": "app.editor.external",
    "followUp": "app.message.followUp",
    "dequeue": "app.message.dequeue",
    "pasteImage": "app.clipboard.pasteImage",
    "newSession": "app.session.new",
    "tree": "app.session.tree",
    "fork": "app.session.fork",
    "resume": "app.session.resume",
    "treeFoldOrUp": "app.tree.foldOrUp",
    "treeUnfoldOrDown": "app.tree.unfoldOrDown",
    "treeEditLabel": "app.tree.editLabel",
    "treeToggleLabelTimestamp": "app.tree.toggleLabelTimestamp",
    "toggleSessionPath": "app.session.togglePath",
    "toggleSessionSort": "app.session.toggleSort",
    "renameSession": "app.session.rename",
    "deleteSession": "app.session.delete",
    "deleteSessionNoninvasive": "app.session.deleteNoninvasive",
}


def _to_keybindings_config(value: dict[str, Any]) -> dict[str, Any]:
    """Keep only entries that are a key string or a list of key strings."""
    config: dict[str, Any] = {}
    for key, binding in value.items():
        if isinstance(binding, str) or (isinstance(binding, list) and all(isinstance(entry, str) for entry in binding)):
            config[key] = binding
    return config


def _order_keybindings_config(config: dict[str, Any]) -> dict[str, Any]:
    """Order known bindings by definition order, then unknown ones alphabetically."""
    ordered: dict[str, Any] = {}
    for keybinding in KEYBINDINGS:
        if keybinding in config:
            ordered[keybinding] = config[keybinding]
    for key in sorted(k for k in config if k not in ordered):
        ordered[key] = config[key]
    return ordered


def migrate_keybindings_config(raw_config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Rename legacy flat binding names. Returns `(config, migrated)`.

    When both the legacy and the current name are present the legacy entry is
    dropped, so an already-migrated file does not lose its current value.
    """
    config: dict[str, Any] = {}
    migrated = False

    for key, value in raw_config.items():
        next_key = KEYBINDING_NAME_MIGRATIONS.get(key, key)
        if next_key != key:
            migrated = True
            if next_key in raw_config:
                continue
        config[next_key] = value

    return _order_keybindings_config(config), migrated


def _load_raw_config(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        parsed = json.loads(file_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return parsed if isinstance(parsed, dict) else None


class KeybindingsManager(TuiKeybindingsManager):
    """TUI keybindings plus the application bindings and a user config file."""

    def __init__(self, user_bindings: dict[str, Any] | None = None, config_path: str | None = None) -> None:
        super().__init__(KEYBINDINGS, user_bindings or {})
        self.config_path = config_path

    @staticmethod
    def load_from_file(path: str | Path) -> dict[str, Any]:
        raw_config = _load_raw_config(path)
        if not raw_config:
            return {}
        config, _migrated = migrate_keybindings_config(raw_config)
        return _to_keybindings_config(config)

    @classmethod
    def create(cls, agent_dir: str | None = None) -> KeybindingsManager:
        directory = agent_dir if agent_dir is not None else get_agent_dir()
        config_path = str(Path(directory) / "keybindings.json")
        return cls(cls.load_from_file(config_path), config_path)

    def reload(self) -> None:
        if not self.config_path:
            return
        self.set_user_bindings(self.load_from_file(self.config_path))

    def get_effective_config(self) -> dict[str, Any]:
        return self.get_resolved_bindings()
