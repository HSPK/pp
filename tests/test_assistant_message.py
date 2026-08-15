"""Python port of `packages/coding-agent/test/assistant-message.test.ts`."""

from __future__ import annotations

import time

from pi_ai.types import AssistantMessage, Cost, TextContent, ThinkingContent, ToolCall, Usage

from pi_coding_agent.modes.interactive.components.assistant_message import (
    OSC133_ZONE_END,
    OSC133_ZONE_FINAL,
    OSC133_ZONE_START,
    AssistantMessageComponent,
)
from pi_coding_agent.modes.interactive.components.markdown_transform import MarkdownTransformContext
from pi_coding_agent.modes.interactive.components.user_message import UserMessageComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi


def create_assistant_message(content: list[object], stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,  # type: ignore[arg-type]
        api="openai-responses",
        provider="openai",
        model="gpt-4o-mini",
        usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost()),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=int(time.time() * 1000),
    )


def test_adds_osc133_zone_markers_without_tool_calls() -> None:
    init_theme("dark")

    component = AssistantMessageComponent(create_assistant_message([TextContent(text="hello")]))
    lines = component.render(40)

    assert len(lines) != 0
    assert OSC133_ZONE_START in lines[0]
    assert lines[-1].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)


def test_does_not_add_osc133_markers_when_message_has_tool_calls() -> None:
    init_theme("dark")

    component = AssistantMessageComponent(
        create_assistant_message(
            [
                TextContent(text="calling tool"),
                ToolCall(id="tool-1", name="read", arguments={"path": "file.txt"}),
            ]
        )
    )
    rendered = "\n".join(component.render(60))

    assert OSC133_ZONE_START not in rendered
    assert OSC133_ZONE_END not in rendered
    assert OSC133_ZONE_FINAL not in rendered


def test_renders_length_stops_with_neutral_truncation_wording() -> None:
    init_theme("dark")

    component = AssistantMessageComponent(
        create_assistant_message([ThinkingContent(thinking="private reasoning")], stop_reason="length"),
        True,
    )
    rendered = "\n".join(component.render(80))

    assert "Thinking..." in rendered
    assert "Response was truncated before completion." in rendered


def test_coalesces_adjacent_thinking_blocks_into_one_hidden_label() -> None:
    init_theme("dark")

    component = AssistantMessageComponent(
        create_assistant_message(
            [
                ThinkingContent(thinking="first thought"),
                ThinkingContent(thinking=""),
                ThinkingContent(thinking="second thought"),
                TextContent(text="answer"),
            ]
        ),
        True,
    )
    rendered = strip_ansi("\n".join(component.render(80)))

    assert rendered.count("Thinking...") == 1
    assert "answer" in rendered


def test_uses_configured_output_padding_for_text_and_thinking() -> None:
    init_theme("dark")

    component = AssistantMessageComponent(
        create_assistant_message([TextContent(text="hello"), ThinkingContent(thinking="reasoning")]),
        False,
        None,
        "Thinking...",
        1,
    )
    lines = [strip_ansi(line) for line in component.render(80)]

    assert any(" hello" in line for line in lines)
    assert any(" reasoning" in line for line in lines)

    component.set_output_pad(0)
    updated_lines = [strip_ansi(line) for line in component.render(80)]

    assert any(line.startswith("hello") for line in updated_lines)
    assert any(line.startswith("reasoning") for line in updated_lines)


def test_chains_markdown_transformers_in_registration_order() -> None:
    init_theme("dark")
    calls: list[str] = []
    message = create_assistant_message([TextContent(text="The result is $x^2$.")])

    def formula(markdown: str, context: MarkdownTransformContext) -> str:
        calls.append("formula")
        assert context == MarkdownTransformContext(message_type="assistant", is_streaming=False, available_width=78)
        return markdown.replace("$x^2$", "x²")

    def suffix(markdown: str, _context: MarkdownTransformContext) -> str:
        calls.append("suffix")
        return f"{markdown} Done."

    component = AssistantMessageComponent(message, False, None, "Thinking...", 1, [formula, suffix])

    assert "The result is x². Done." in strip_ansi("\n".join(component.render(80)))
    assert calls == ["formula", "suffix"]


def test_identifies_partial_assistant_markdown_as_streaming() -> None:
    init_theme("dark")
    streaming_states: list[bool] = []
    message = create_assistant_message([TextContent(text="partial")])

    def transformer(markdown: str, context: MarkdownTransformContext) -> str:
        streaming_states.append(context.is_streaming)
        return markdown if context.is_streaming else f"{markdown} transformed"

    component = AssistantMessageComponent(None, False, None, "Thinking...", 1, [transformer])

    component.update_content(message, True)
    assert "transformed" not in strip_ansi("\n".join(component.render(80)))

    component.update_content(message, False)
    assert "partial transformed" in strip_ansi("\n".join(component.render(80)))
    assert streaming_states == [True, False]


def test_reapplies_markdown_transformers_when_available_width_changes() -> None:
    init_theme("dark")
    available_widths: list[int] = []

    def transformer(markdown: str, context: MarkdownTransformContext) -> str:
        available_widths.append(context.available_width)
        return f"{markdown} ({context.available_width})"

    component = AssistantMessageComponent(
        create_assistant_message([TextContent(text="answer")]),
        False,
        None,
        "Thinking...",
        1,
        [transformer],
    )

    assert "answer (78)" in strip_ansi("\n".join(component.render(80)))
    component.render(80)
    assert "answer (58)" in strip_ansi("\n".join(component.render(60)))
    assert available_widths == [78, 58]


def test_continues_the_transformer_chain_when_a_transformer_throws() -> None:
    init_theme("dark")
    calls: list[str] = []

    def first(markdown: str, _context: MarkdownTransformContext) -> str:
        calls.append("first")
        return markdown.replace("still", "remains")

    def broken(_markdown: str, _context: MarkdownTransformContext) -> str:
        calls.append("throw")
        raise RuntimeError("broken transformer")

    def last(markdown: str, _context: MarkdownTransformContext) -> str:
        calls.append("last")
        return f"{markdown} after error"

    component = AssistantMessageComponent(
        create_assistant_message([TextContent(text="still visible")]),
        False,
        None,
        "Thinking...",
        1,
        [first, broken, last],
    )

    assert "remains visible after error" in strip_ansi("\n".join(component.render(80)))
    assert calls == ["first", "throw", "last"]


def test_transforms_text_and_thinking_without_mutating_the_message() -> None:
    init_theme("dark")
    message = create_assistant_message([TextContent(text="answer"), ThinkingContent(thinking="reasoning")])

    def transformer(markdown: str, context: MarkdownTransformContext) -> str:
        return f"{context.message_type}:{markdown}"

    component = AssistantMessageComponent(message, False, None, "Thinking...", 1, [transformer])

    rendered = strip_ansi("\n".join(component.render(80)))

    assert "assistant:answer" in rendered
    assert "assistant-thinking:reasoning" in rendered
    assert message.content == [TextContent(text="answer"), ThinkingContent(thinking="reasoning")]


def test_uses_configured_output_padding_for_user_messages() -> None:
    init_theme("dark")

    padded = UserMessageComponent("hello", None, 1)
    padded_lines = [strip_ansi(line) for line in padded.render(40)]
    assert any(line.startswith(" hello") for line in padded_lines)

    unpadded = UserMessageComponent("hello", None, 0)
    unpadded_lines = [strip_ansi(line) for line in unpadded.render(40)]
    assert any(line.startswith("hello") for line in unpadded_lines)
