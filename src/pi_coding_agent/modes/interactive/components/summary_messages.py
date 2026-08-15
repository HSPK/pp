"""Collapsible branch/compaction/skill summary messages.

Ported from ``branch-summary-message.ts``, ``compaction-summary-message.ts``
and ``skill-invocation-message.ts`` under
``packages/coding-agent/src/modes/interactive/components/``.

All three are the same shape: a ``[label]`` line plus either the full Markdown
body (expanded) or a one-line "press X to expand" hint (collapsed).
"""

from __future__ import annotations

from pi_agent.harness.messages import BranchSummaryMessage, CompactionSummaryMessage
from pi_tui.components.box import Box
from pi_tui.components.markdown import DefaultTextStyle, Markdown, MarkdownTheme
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from ....core.agent_session import ParsedSkillBlock
from ..theme.theme import get_markdown_theme, theme
from .keybinding_hints import key_text

_EXPAND_KEYBINDING = "app.tools.expand"


def _custom_message_text(text: str) -> str:
    return theme.fg("customMessageText", text)


def _label(name: str) -> str:
    return theme.fg("customMessageLabel", f"\x1b[1m[{name}]\x1b[22m")


def _format_token_count(value: int) -> str:
    """JS ``Number.prototype.toLocaleString()`` under the default en-US locale."""
    return f"{value:,}"


class _CollapsibleSummaryBox(Box):
    def __init__(self, markdown_theme: MarkdownTheme | None = None) -> None:
        super().__init__(1, 1, lambda text: theme.bg("customMessageBg", text))
        self.expanded = False
        self.markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def _add_markdown(self, body: str) -> None:
        self.add_child(
            Markdown(
                body,
                0,
                0,
                self.markdown_theme,
                DefaultTextStyle(color=_custom_message_text),
            )
        )

    def _update_display(self) -> None:
        raise NotImplementedError


class BranchSummaryMessageComponent(_CollapsibleSummaryBox):
    def __init__(self, message: BranchSummaryMessage, markdown_theme: MarkdownTheme | None = None) -> None:
        super().__init__(markdown_theme)
        self.message = message
        self._update_display()

    def _update_display(self) -> None:
        self.clear()
        self.add_child(Text(_label("branch"), 0, 0))
        self.add_child(Spacer(1))

        if self.expanded:
            self._add_markdown(f"**Branch Summary**\n\n{self.message.summary}")
        else:
            self.add_child(
                Text(
                    _custom_message_text("Branch summary (")
                    + theme.fg("dim", key_text(_EXPAND_KEYBINDING))
                    + _custom_message_text(" to expand)"),
                    0,
                    0,
                )
            )


class CompactionSummaryMessageComponent(_CollapsibleSummaryBox):
    def __init__(self, message: CompactionSummaryMessage, markdown_theme: MarkdownTheme | None = None) -> None:
        super().__init__(markdown_theme)
        self.message = message
        self._update_display()

    def _update_display(self) -> None:
        self.clear()
        token_str = _format_token_count(self.message.tokens_before)
        self.add_child(Text(_label("compaction"), 0, 0))
        self.add_child(Spacer(1))

        if self.expanded:
            self._add_markdown(f"**Compacted from {token_str} tokens**\n\n{self.message.summary}")
        else:
            self.add_child(
                Text(
                    _custom_message_text(f"Compacted from {token_str} tokens (")
                    + theme.fg("dim", key_text(_EXPAND_KEYBINDING))
                    + _custom_message_text(" to expand)"),
                    0,
                    0,
                )
            )


class SkillInvocationMessageComponent(_CollapsibleSummaryBox):
    """Renders only the skill block; the user message is rendered separately."""

    def __init__(self, skill_block: ParsedSkillBlock, markdown_theme: MarkdownTheme | None = None) -> None:
        super().__init__(markdown_theme)
        self.skill_block = skill_block
        self._update_display()

    def _update_display(self) -> None:
        self.clear()
        if self.expanded:
            self.add_child(Text(_label("skill"), 0, 0))
            self._add_markdown(f"**{self.skill_block.name}**\n\n{self.skill_block.content}")
        else:
            self.add_child(
                Text(
                    theme.fg("customMessageLabel", "\x1b[1m[skill]\x1b[22m ")
                    + _custom_message_text(self.skill_block.name)
                    + theme.fg("dim", f" ({key_text(_EXPAND_KEYBINDING)} to expand)"),
                    0,
                    0,
                )
            )
