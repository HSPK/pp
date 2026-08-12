"""Python port of `packages/coding-agent/test/user-message.test.ts`."""

from __future__ import annotations

import pytest
from pi_coding_agent.modes.interactive.components.markdown_transform import MarkdownTransformContext
from pi_coding_agent.modes.interactive.components.user_message import UserMessageComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi

OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"
BG_RESET = "\x1b[49m"


@pytest.fixture(autouse=True)
def _theme():
    init_theme("dark")


def test_keeps_user_message_height_stable_while_moving_closing_osc_markers_off_line_end():
    component = UserMessageComponent("hello")

    lines = component.render(20)

    assert len(lines) == 3
    assert OSC133_ZONE_START in lines[0]
    assert lines[0].endswith(BG_RESET)
    assert OSC133_ZONE_END not in lines[0]
    assert "hello" in lines[1]
    assert lines[2].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)
    assert lines[2].endswith(BG_RESET)


def test_chains_markdown_transformers_with_user_message_context():
    calls: list[str] = []
    contexts: list[MarkdownTransformContext] = []

    def formula(markdown: str, context: MarkdownTransformContext) -> str:
        calls.append("formula")
        contexts.append(context)
        return markdown.replace("$x^2$", "x\u00b2")

    def suffix(markdown: str, _context: MarkdownTransformContext) -> str:
        calls.append("suffix")
        return f"{markdown} Done."

    component = UserMessageComponent("The input is $x^2$.", None, 1, [formula, suffix])

    assert "The input is x\u00b2. Done." in strip_ansi("\n".join(component.render(80)))
    assert calls == ["formula", "suffix"]
    assert contexts[0] == MarkdownTransformContext(message_type="user", is_streaming=False, available_width=78)


def test_reapplies_markdown_transformers_when_invalidated():
    suffix = "before"

    def transformer(markdown: str, _context: MarkdownTransformContext) -> str:
        return f"{markdown} {suffix}"

    component = UserMessageComponent("Message", None, 1, [transformer])

    assert "Message before" in strip_ansi("\n".join(component.render(80)))

    suffix = "after"
    component.invalidate()

    assert "Message after" in strip_ansi("\n".join(component.render(80)))
