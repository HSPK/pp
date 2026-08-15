"""Assistant message rendering.

Ported from ``packages/coding-agent/src/modes/interactive/components/assistant-message.ts``.
"""

from __future__ import annotations

from collections.abc import Sequence

from pi_ai.types import AssistantMessage
from pi_tui.component import Container
from pi_tui.components.markdown import DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from ..theme.theme import get_markdown_theme, theme
from .markdown_transform import MarkdownTransformer, create_markdown_transform

OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"


def _is_visible(content: object) -> bool:
    content_type = getattr(content, "type", None)
    if content_type == "text":
        return bool(content.text.strip())  # type: ignore[attr-defined]
    if content_type == "thinking":
        return bool(content.thinking.strip())  # type: ignore[attr-defined]
    return False


class AssistantMessageComponent(Container):
    def __init__(
        self,
        message: AssistantMessage | None = None,
        hide_thinking_block: bool = False,
        markdown_theme: MarkdownTheme | None = None,
        hidden_thinking_label: str = "Thinking...",
        output_pad: int = 1,
        markdown_transformers: Sequence[MarkdownTransformer] = (),
    ) -> None:
        super().__init__()
        self.hide_thinking_block = hide_thinking_block
        self.markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self.hidden_thinking_label = hidden_thinking_label
        self.output_pad = output_pad
        self.markdown_transformers = markdown_transformers
        self.last_message: AssistantMessage | None = None
        self.has_tool_calls = False
        self.is_streaming = False

        self.content_container = Container()
        self.add_child(self.content_container)

        if message is not None:
            self.update_content(message)

    def invalidate(self) -> None:
        super().invalidate()
        if self.last_message is not None:
            self.update_content(self.last_message)

    def set_hide_thinking_block(self, hide: bool) -> None:
        self.hide_thinking_block = hide
        if self.last_message is not None:
            self.update_content(self.last_message)

    def set_hidden_thinking_label(self, label: str) -> None:
        self.hidden_thinking_label = label
        if self.last_message is not None:
            self.update_content(self.last_message)

    def set_output_pad(self, padding: int) -> None:
        self.output_pad = padding
        if self.last_message is not None:
            self.update_content(self.last_message)

    def render(self, width: int) -> list[str]:
        lines = super().render(width)
        if self.has_tool_calls or len(lines) == 0:
            return lines

        lines[0] = OSC133_ZONE_START + lines[0]
        lines[-1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[-1]
        return lines

    def _add_stop_reason_notice(self, message: AssistantMessage) -> None:
        # Aborted/errored *tool* calls are surfaced by the tool execution rows,
        # but a length stop can land mid-tool-call so it is always shown here.
        self.has_tool_calls = any(getattr(c, "type", None) == "toolCall" for c in message.content)
        stop_reason = getattr(message, "stop_reason", None)
        error_message = getattr(message, "error_message", None)

        if stop_reason == "length":
            self.content_container.add_child(Spacer(1))
            self.content_container.add_child(
                Text(theme.fg("error", "Response was truncated before completion."), self.output_pad, 0)
            )
        elif not self.has_tool_calls:
            if stop_reason == "aborted":
                abort_message = (
                    error_message if error_message and error_message != "Request was aborted" else "Operation aborted"
                )
                self.content_container.add_child(Spacer(1))
                self.content_container.add_child(Text(theme.fg("error", abort_message), self.output_pad, 0))
            elif stop_reason == "error":
                self.content_container.add_child(Spacer(1))
                self.content_container.add_child(
                    Text(theme.fg("error", f"Error: {error_message or 'Unknown error'}"), self.output_pad, 0)
                )

    def update_content(self, message: AssistantMessage, is_streaming: bool | None = None) -> None:
        self.last_message = message
        self.is_streaming = self.is_streaming if is_streaming is None else is_streaming

        self.content_container.clear()

        if any(_is_visible(content) for content in message.content):
            self.content_container.add_child(Spacer(1))

        index = 0
        while index < len(message.content):
            content = message.content[index]
            content_type = getattr(content, "type", None)

            if content_type == "text" and content.text.strip():
                # Assistant text has no background; padding_y=0 avoids an extra
                # blank line before a following tool execution row.
                self.content_container.add_child(
                    Markdown(
                        content.text.strip(),
                        self.output_pad,
                        0,
                        self.markdown_theme,
                        None,
                        MarkdownOptions(
                            transform=create_markdown_transform(
                                "assistant", self.is_streaming, self.markdown_transformers
                            )
                        ),
                    )
                )
            elif content_type == "thinking":
                thinking_blocks: list[str] = []
                while index < len(message.content):
                    thinking_content = message.content[index]
                    if getattr(thinking_content, "type", None) != "thinking":
                        break
                    thinking = thinking_content.thinking.strip()
                    if thinking:
                        thinking_blocks.append(thinking)
                    index += 1
                index -= 1

                if len(thinking_blocks) == 0:
                    index += 1
                    continue

                # Only pad when more visible assistant content follows, so a
                # separately rendered tool execution block is not preceded by a
                # stray blank line.
                has_visible_content_after = any(_is_visible(c) for c in message.content[index + 1 :])

                if self.hide_thinking_block:
                    self.content_container.add_child(
                        Text(
                            theme.italic(theme.fg("thinkingText", self.hidden_thinking_label)),
                            self.output_pad,
                            0,
                        )
                    )
                else:
                    self.content_container.add_child(
                        Markdown(
                            "\n\n".join(thinking_blocks),
                            self.output_pad,
                            0,
                            self.markdown_theme,
                            DefaultTextStyle(
                                color=lambda text: theme.fg("thinkingText", text),
                                italic=True,
                            ),
                            MarkdownOptions(
                                transform=create_markdown_transform(
                                    "assistant-thinking", self.is_streaming, self.markdown_transformers
                                )
                            ),
                        )
                    )
                if has_visible_content_after:
                    self.content_container.add_child(Spacer(1))

            index += 1

        self._add_stop_reason_notice(message)


__all__ = ["AssistantMessageComponent"]
