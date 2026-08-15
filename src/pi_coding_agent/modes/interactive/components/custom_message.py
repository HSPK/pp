"""Extension-provided message and session entry rendering.

Ported from ``custom-message.ts`` and ``custom-entry.ts`` under
``packages/coding-agent/src/modes/interactive/components/``.

The renderer callbacks come from extensions, so both components treat a
renderer raising as a recoverable condition: ``CustomMessageComponent`` falls
back to its own default rendering, and ``CustomEntryComponent`` shows the error
inline instead of taking down the transcript.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.harness.messages import CustomMessage
from pi_tui.component import Component, Container
from pi_tui.components.box import Box
from pi_tui.components.markdown import DefaultTextStyle, Markdown, MarkdownTheme
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from ....core.session_manager import CustomEntry
from ..theme.theme import get_markdown_theme, theme


@dataclass
class MessageRenderContext:
    expanded: bool
    output_pad: int


@dataclass
class EntryRenderContext:
    expanded: bool


MessageRenderer = Callable[[CustomMessage, MessageRenderContext, Any], Component | None]
EntryRenderer = Callable[[CustomEntry, EntryRenderContext, Any], Component | None]


class CustomMessageComponent(Container):
    """Renders a custom message entry from extensions."""

    def __init__(
        self,
        message: CustomMessage,
        custom_renderer: MessageRenderer | None = None,
        markdown_theme: MarkdownTheme | None = None,
        output_pad: int = 1,
    ) -> None:
        super().__init__()
        self.message = message
        self.custom_renderer = custom_renderer
        self.markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self.output_pad = output_pad
        self._expanded = False
        self._custom_component: Component | None = None

        self.add_child(Spacer(1))
        self.box = Box(1, 1, lambda text: theme.bg("customMessageBg", text))
        self._rebuild()

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._expanded = expanded
            self._rebuild()

    def set_output_pad(self, output_pad: int) -> None:
        if self.output_pad != output_pad:
            self.output_pad = output_pad
            self._rebuild()

    def invalidate(self) -> None:
        super().invalidate()
        self._rebuild()

    def _message_text(self) -> str:
        content = self.message.content
        if isinstance(content, str):
            return content
        return "\n".join(part.text for part in content if getattr(part, "type", None) == "text")

    def _rebuild(self) -> None:
        if self._custom_component is not None:
            self.remove_child(self._custom_component)
            self._custom_component = None
        self.remove_child(self.box)

        if self.custom_renderer is not None:
            try:
                component = self.custom_renderer(
                    self.message,
                    MessageRenderContext(expanded=self._expanded, output_pad=self.output_pad),
                    theme,
                )
            except Exception:
                component = None
            if component is not None:
                self._custom_component = component
                self.add_child(component)
                return

        self.add_child(self.box)
        self.box.clear()

        label = theme.fg("customMessageLabel", f"\x1b[1m[{self.message.custom_type}]\x1b[22m")
        self.box.add_child(Text(label, 0, 0))
        self.box.add_child(Spacer(1))
        self.box.add_child(
            Markdown(
                self._message_text(),
                0,
                0,
                self.markdown_theme,
                DefaultTextStyle(color=lambda text: theme.fg("customMessageText", text)),
            )
        )


class CustomEntryComponent(Container):
    """Renders a custom session entry from extensions.

    The host owns transcript spacing, so the renderer supplies content only.
    """

    def __init__(self, entry: CustomEntry, renderer: EntryRenderer) -> None:
        super().__init__()
        self.entry = entry
        self.renderer = renderer
        self._custom_component: Component | None = None
        self._expanded = False
        self._rebuild()

    def has_content(self) -> bool:
        return self._custom_component is not None

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._expanded = expanded
            self._rebuild()

    def invalidate(self) -> None:
        super().invalidate()
        self._rebuild()

    def _rebuild(self) -> None:
        self.clear()
        self._custom_component = None

        try:
            component = self.renderer(self.entry, EntryRenderContext(expanded=self._expanded), theme)
        except Exception as error:
            box = Box(1, 1, lambda text: theme.bg("customMessageBg", text))
            box.add_child(Text(theme.fg("error", f"[{self.entry.custom_type}] renderer failed: {error}"), 0, 0))
            component = box

        if component is None:
            return

        self._custom_component = component
        self.add_child(Spacer(1))
        self.add_child(component)
