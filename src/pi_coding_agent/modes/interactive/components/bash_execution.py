"""Bash command execution row with streaming output.

Ported from ``packages/coding-agent/src/modes/interactive/components/bash-execution.ts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pi_tui.component import Component, Container
from pi_tui.components.loader import Loader
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from ....tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail
from ....utils.ansi import strip_ansi
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, key_text
from .visual_truncate import truncate_to_visual_lines

if TYPE_CHECKING:
    from pi_tui.tui import TuiBase

# Preview line limit when collapsed (matches tool execution behaviour).
PREVIEW_LINES = 20

BashExecutionStatus = Literal["running", "complete", "cancelled", "error"]


class _VisualTruncatedOutput(Component):
    """Renders the collapsed preview, caching per width.

    TypeScript builds this as an inline object literal implementing
    ``Component``; Python needs a named class for the same thing.
    """

    def __init__(self, styled_input: str) -> None:
        self._styled_input = styled_input
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def invalidate(self) -> None:
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is None or self._cached_width != width:
            self._cached_lines = truncate_to_visual_lines(self._styled_input, PREVIEW_LINES, width, 1).visual_lines
            self._cached_width = width
        return self._cached_lines


class BashExecutionComponent(Container):
    def __init__(self, command: str, ui: TuiBase | None, exclude_from_context: bool = False) -> None:
        super().__init__()
        self.command = command
        self.output_lines: list[str] = []
        self.status: BashExecutionStatus = "running"
        self.exit_code: int | None = None
        self.truncation_result: TruncationResult | None = None
        self.full_output_path: str | None = None
        self.expanded = False

        # Commands excluded from context (the `!!` prefix) get a dim border.
        color_key = "dim" if exclude_from_context else "bashMode"
        self._color_key = color_key

        def border_color(text: str) -> str:
            return theme.fg(color_key, text)

        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(border_color))

        self.content_container = Container()
        self.add_child(self.content_container)
        self.content_container.add_child(Text(theme.fg(color_key, theme.bold(f"$ {command}")), 1, 0))

        self.loader = Loader(
            ui,
            lambda spinner: theme.fg(color_key, spinner),
            lambda text: theme.fg("muted", text),
            f"Running... ({key_text('tui.select.cancel')} to cancel)",
        )
        self.content_container.add_child(self.loader)

        self.add_child(DynamicBorder(border_color))

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def append_output(self, chunk: str) -> None:
        # Binary data is already sanitized by the bash executor; here we only
        # strip ANSI and normalize line endings.
        clean = strip_ansi(chunk).replace("\r\n", "\n").replace("\r", "\n")
        new_lines = clean.split("\n")
        if len(self.output_lines) > 0 and len(new_lines) > 0:
            # The first chunk continues the previous (incomplete) line.
            self.output_lines[-1] += new_lines[0]
            self.output_lines.extend(new_lines[1:])
        else:
            self.output_lines.extend(new_lines)
        self._update_display()

    def set_complete(
        self,
        exit_code: int | None,
        cancelled: bool,
        truncation_result: TruncationResult | None = None,
        full_output_path: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        if cancelled:
            self.status = "cancelled"
        elif exit_code is not None and exit_code != 0:
            self.status = "error"
        else:
            self.status = "complete"
        self.truncation_result = truncation_result
        self.full_output_path = full_output_path
        self.loader.stop()
        self._update_display()

    def _update_display(self) -> None:
        # Same limits as the bash tool, so what is shown matches what the LLM sees.
        context_truncation = truncate_tail(
            "\n".join(self.output_lines), max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES
        )
        available_lines = context_truncation.content.split("\n") if context_truncation.content else []
        preview_logical_lines = available_lines[-PREVIEW_LINES:]
        hidden_line_count = len(available_lines) - len(preview_logical_lines)

        self.content_container.clear()
        self.content_container.add_child(Text(theme.fg("bashMode", theme.bold(f"$ {self.command}")), 1, 0))

        if len(available_lines) > 0:
            if self.expanded:
                display_text = "\n".join(theme.fg("muted", line) for line in available_lines)
                self.content_container.add_child(Text(f"\n{display_text}", 1, 0))
            else:
                styled_output = "\n".join(theme.fg("muted", line) for line in preview_logical_lines)
                self.content_container.add_child(_VisualTruncatedOutput(f"\n{styled_output}"))

        if self.status == "running":
            self.content_container.add_child(self.loader)
            return

        status_parts: list[str] = []
        if hidden_line_count > 0:
            if self.expanded:
                status_parts.append(
                    f"{theme.fg('muted', '(')}{key_hint('app.tools.expand', 'to collapse')}{theme.fg('muted', ')')}"
                )
            else:
                status_parts.append(
                    f"{theme.fg('muted', f'... {hidden_line_count} more lines (')}"
                    f"{key_hint('app.tools.expand', 'to expand')}{theme.fg('muted', ')')}"
                )

        if self.status == "cancelled":
            status_parts.append(theme.fg("warning", "(cancelled)"))
        elif self.status == "error":
            status_parts.append(theme.fg("error", f"(exit {self.exit_code})"))

        was_truncated = (
            self.truncation_result.truncated if self.truncation_result is not None else False
        ) or context_truncation.truncated
        if was_truncated and self.full_output_path:
            status_parts.append(theme.fg("warning", f"Output truncated. Full output: {self.full_output_path}"))

        if len(status_parts) > 0:
            self.content_container.add_child(Text("\n" + "\n".join(status_parts), 1, 0))

    def get_output(self) -> str:
        """Raw output, for building a ``BashExecutionMessage``."""
        return "\n".join(self.output_lines)

    def get_command(self) -> str:
        return self.command


__all__ = ["PREVIEW_LINES", "BashExecutionComponent"]
