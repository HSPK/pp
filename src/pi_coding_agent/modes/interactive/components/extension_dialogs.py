"""Extension-facing dialogs: input, selector, editor.

Ported from ``extension-input.ts``, ``extension-selector.ts`` and
``extension-editor.ts`` under
``packages/coding-agent/src/modes/interactive/components/``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pi_tui.component import Container
from pi_tui.components.editor import Editor, EditorOptions
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings
from pi_tui.tasks import spawn

from ..external_editor import (
    ExternalEditorOptions,
    default_external_editor_command,
    edit_in_external_editor,
)
from ..theme.theme import get_editor_theme, theme
from .countdown_timer import CountdownTimer
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint

if TYPE_CHECKING:
    from pi_tui.tui import TuiBase


class ExtensionInputComponent(Container):
    """Single-line text input for extensions."""

    def __init__(
        self,
        title: str,
        _placeholder: str | None,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None],
        tui: TuiBase | None = None,
        timeout: int | None = None,
    ) -> None:
        super().__init__()
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self.base_title = title
        self._focused = False

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.title_text = Text(theme.fg("accent", title), 1, 0)
        self.add_child(self.title_text)
        self.add_child(Spacer(1))

        self.countdown: CountdownTimer | None = None
        if timeout and timeout > 0 and tui is not None:
            self.countdown = CountdownTimer(
                timeout,
                tui,
                lambda seconds: self.title_text.set_text(theme.fg("accent", f"{self.base_title} ({seconds}s)")),
                self._on_cancel,
            )

        self.input = Input()
        self.add_child(self.input)
        self.add_child(Spacer(1))
        self.add_child(
            Text(
                f"{key_hint('tui.select.confirm', 'submit')}  {key_hint('tui.select.cancel', 'cancel')}",
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        # Propagated so the input can place the hardware cursor for IME.
        self._focused = value
        self.input.focused = value

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.select.confirm") or key_data == "\n":
            self._on_submit(self.input.get_value())
        elif keybindings.matches(key_data, "tui.select.cancel"):
            self._on_cancel()
        else:
            self.input.handle_input(key_data)

    def dispose(self) -> None:
        if self.countdown is not None:
            self.countdown.dispose()


class ExtensionSelectorComponent(Container):
    """List of string options with keyboard navigation."""

    def __init__(
        self,
        title: str,
        options: list[str],
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
        tui: TuiBase | None = None,
        timeout: int | None = None,
        on_toggle_tools_expanded: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.options = options
        self.selected_index = 0
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._on_toggle_tools_expanded = on_toggle_tools_expanded
        self.base_title = title

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.title_text = Text(theme.fg("accent", theme.bold(title)), 1, 0)
        self.add_child(self.title_text)
        self.add_child(Spacer(1))

        self.countdown: CountdownTimer | None = None
        if timeout and timeout > 0 and tui is not None:
            self.countdown = CountdownTimer(
                timeout,
                tui,
                lambda seconds: self.title_text.set_text(
                    theme.fg("accent", theme.bold(f"{self.base_title} ({seconds}s)"))
                ),
                self._on_cancel,
            )

        self.list_container = Container()
        self.add_child(self.list_container)
        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "select")
                + "  "
                + key_hint("tui.select.cancel", "cancel"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        self._update_list()

    def _update_list(self) -> None:
        self.list_container.clear()
        for index, option in enumerate(self.options):
            if index == self.selected_index:
                text = theme.fg("accent", "→ ") + theme.fg("accent", option)
            else:
                text = f"  {theme.fg('text', option)}"
            self.list_container.add_child(Text(text, 1, 0))

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "app.tools.expand"):
            if self._on_toggle_tools_expanded is not None:
                self._on_toggle_tools_expanded()
        elif keybindings.matches(key_data, "tui.select.up") or key_data == "k":
            self.selected_index = max(0, self.selected_index - 1)
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.down") or key_data == "j":
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            self._update_list()
        elif keybindings.matches(key_data, "tui.select.confirm") or key_data == "\n":
            if 0 <= self.selected_index < len(self.options):
                self._on_select(self.options[self.selected_index])
        elif keybindings.matches(key_data, "tui.select.cancel"):
            self._on_cancel()

    def dispose(self) -> None:
        if self.countdown is not None:
            self.countdown.dispose()


class ExtensionEditorComponent(Container):
    """Multi-line editor for extensions, with external-editor support."""

    def __init__(
        self,
        tui: Any,
        keybindings: Any,
        title: str,
        prefill: str | None,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None],
        options: EditorOptions | None = None,
        external_editor_command: str | None = None,
    ) -> None:
        super().__init__()
        self.tui = tui
        self.keybindings = keybindings
        self.external_editor_command = external_editor_command or default_external_editor_command()
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._focused = False

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", title), 1, 0))
        self.add_child(Spacer(1))

        self.editor = Editor(tui, get_editor_theme(), options)
        if prefill:
            self.editor.set_text(prefill)
        self.editor.on_submit = self._on_submit
        self.add_child(self.editor)

        self.add_child(Spacer(1))
        self.add_child(
            Text(
                key_hint("tui.select.confirm", "submit")
                + "  "
                + key_hint("tui.input.newLine", "newline")
                + "  "
                + key_hint("tui.select.cancel", "cancel")
                + f"  {key_hint('app.editor.external', 'external editor')}",
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.editor.focused = value

    def handle_input(self, key_data: str) -> None:
        if get_keybindings().matches(key_data, "tui.select.cancel"):
            self._on_cancel()
            return
        if self.keybindings.matches(key_data, "app.editor.external"):
            # TS uses `void this.handleOpenExternalEditor()`; `spawn` is the
            # equivalent that keeps a strong reference so the task is not
            # garbage collected mid-flight.
            spawn(self.handle_open_external_editor())
            return
        self.editor.handle_input(key_data)

    async def handle_open_external_editor(self) -> None:
        content = self.editor.get_text()
        self.tui.stop()
        try:
            result = await edit_in_external_editor(
                ExternalEditorOptions(command=self.external_editor_command, content=content)
            )
            if result.status == "complete":
                self.editor.set_text(result.content)
        finally:
            self.tui.start()
            self.tui.request_render(True)


__all__ = [
    "ExtensionEditorComponent",
    "ExtensionInputComponent",
    "ExtensionSelectorComponent",
]
