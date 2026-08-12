"""User message rendering.

Ported from ``packages/coding-agent/src/modes/interactive/components/user-message.ts``.
"""

from __future__ import annotations

from collections.abc import Sequence

from pi_tui.component import Container
from pi_tui.components.box import Box
from pi_tui.components.markdown import DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme

from ..theme.theme import get_markdown_theme, theme
from .markdown_transform import MarkdownTransformer, create_markdown_transform

OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"


class UserMessageComponent(Container):
    def __init__(
        self,
        text: str,
        markdown_theme: MarkdownTheme | None = None,
        output_pad: int = 1,
        markdown_transformers: Sequence[MarkdownTransformer] = (),
    ) -> None:
        super().__init__()
        self.text = text
        self.markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self.output_pad = output_pad
        self.markdown_transformers = markdown_transformers
        self._rebuild()

    def set_output_pad(self, padding: int) -> None:
        self.output_pad = padding
        self._rebuild()

    def _rebuild(self) -> None:
        self.clear()
        content_box = Box(self.output_pad, 1, lambda content: theme.bg("userMessageBg", content))
        content_box.add_child(
            Markdown(
                self.text,
                0,
                0,
                self.markdown_theme,
                DefaultTextStyle(color=lambda content: theme.fg("userMessageText", content)),
                MarkdownOptions(
                    preserve_ordered_list_markers=True,
                    preserve_backslash_escapes=True,
                    transform=create_markdown_transform("user", False, self.markdown_transformers),
                ),
            )
        )
        self.add_child(content_box)

    def render(self, width: int) -> list[str]:
        lines = super().render(width)
        if len(lines) == 0:
            return lines

        # OSC 133 shell-integration markers let terminals treat each user
        # message as one navigable prompt zone.
        lines[0] = OSC133_ZONE_START + lines[0]
        lines[-1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[-1]
        return lines
