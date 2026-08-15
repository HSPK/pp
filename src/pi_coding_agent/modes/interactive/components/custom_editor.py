"""Editor subclass that dispatches app-level keybindings.

Ported from ``packages/coding-agent/src/modes/interactive/components/custom-editor.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pi_tui.components.editor import Editor, EditorOptions, EditorTheme


class CustomEditor(Editor):
    def __init__(
        self,
        tui: Any,
        theme: EditorTheme,
        keybindings: Any,
        options: EditorOptions | None = None,
    ) -> None:
        super().__init__(tui, theme, options)
        self.keybindings = keybindings
        self.action_handlers: dict[str, Callable[[], None]] = {}

        # Dynamically replaceable handlers.
        self.on_escape: Callable[[], None] | None = None
        self.on_ctrl_d: Callable[[], None] | None = None
        self.on_paste_image: Callable[[], None] | None = None
        self.on_extension_shortcut: Callable[[str], bool] | None = None

    def on_action(self, action: str, handler: Callable[[], None]) -> None:
        self.action_handlers[action] = handler

    def handle_input(self, data: str) -> None:
        if self.on_extension_shortcut is not None and self.on_extension_shortcut(data):
            return

        if self.keybindings.matches(data, "app.clipboard.pasteImage"):
            if self.on_paste_image is not None:
                self.on_paste_image()
            return

        # Interrupt only fires when the autocomplete popup is not open; otherwise
        # escape must reach the editor so it can dismiss the popup.
        if self.keybindings.matches(data, "app.interrupt"):
            if not self.is_showing_autocomplete():
                handler = self.on_escape or self.action_handlers.get("app.interrupt")
                if handler is not None:
                    handler()
                    return
            super().handle_input(data)
            return

        # Exit only fires on an empty editor; otherwise the key keeps its
        # delete-char-forward meaning.
        if self.keybindings.matches(data, "app.exit") and len(self.get_text()) == 0:
            handler = self.on_ctrl_d or self.action_handlers.get("app.exit")
            if handler is not None:
                handler()
            return

        # Explicit history bindings win over app actions while the editor is
        # focused, so a user can bind ctrl+p even though it cycles models.
        if self.keybindings.matches(data, "tui.editor.historyPrevious") or self.keybindings.matches(
            data, "tui.editor.historyNext"
        ):
            super().handle_input(data)
            return

        for action, handler in self.action_handlers.items():
            if action not in ("app.interrupt", "app.exit") and self.keybindings.matches(data, action):
                handler()
                return

        super().handle_input(data)


__all__ = ["CustomEditor"]
