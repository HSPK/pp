"""Python port of `packages/coding-agent/test/suite/lax-message-content.test.ts`.

The message types require `content` to always be present, but untyped
extension tools, hand-built histories, and old or hand-edited session files can
violate that contract. The ingestion boundaries are intentionally lax and
normalize null/missing content to an empty list so it never reaches rendering,
compaction, or provider request conversion (issues #6259, #6276).
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from pathlib import Path

from harness import create_harness
from pi_agent.types import AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import AssistantMessage, Cost, ToolResultMessage, Usage, UserMessage
from pi_ai.types import now_ms as _now_ms
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import MessageEndEventResult, ToolDefinition
from pi_coding_agent.core.session_manager import (
    CustomMessageEntry,
    SessionMessageEntry,
    session_entry_to_context_messages,
)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _message_entry(message: object) -> SessionMessageEntry:
    return SessionMessageEntry(id="entry-1", parent_id=None, timestamp=_iso_now(), message=message)


async def test_normalizes_tool_results_from_untyped_tools_that_omit_content(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def execute(tool_call_id, params, signal, on_update, ctx) -> AgentToolResult:
            # Simulate an untyped extension tool that omits content.
            return AgentToolResult(details={})

        pi.register_tool(
            ToolDefinition(
                name="web_search",
                label="Web Search",
                description="Custom tool that returns a result without content",
                execute=execute,
            )
        )

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("web_search", {})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("search something"), timeout=10)

        tool_results = [m for m in harness.session.messages if getattr(m, "role", "") == "toolResult"]
        assert len(tool_results) == 1
        assert tool_results[0].content == []
        # The follow-up turn consumed the normalized tool result without crashing.
        assert harness.get_pending_response_count() == 0
    finally:
        harness.cleanup()


async def test_normalizes_null_content_in_message_end_extension_replacements(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def on_message_end(event, ctx) -> MessageEndEventResult | None:
            if getattr(event.message, "role", "") != "assistant":
                return None
            # Simulate an untyped extension replacing a message without content.
            return MessageEndEventResult(message=dataclasses.replace(event.message, content=None))

        pi.on("message_end", on_message_end)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.set_responses([faux_assistant_message("hello")])
        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assistant_messages = [m for m in harness.session.messages if getattr(m, "role", "") == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].content == []
    finally:
        harness.cleanup()


async def test_normalizes_null_content_in_custom_messages_from_extensions(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        await harness.session.send_custom_message("test", None, False, None)

        custom_messages = [m for m in harness.session.messages if getattr(m, "role", "") == "custom"]
        assert len(custom_messages) == 1
        assert custom_messages[0].content == []
    finally:
        harness.cleanup()


def test_normalizes_null_or_missing_content_when_loading_session_message_entries() -> None:
    # TypeScript's third case omits `content` entirely; Python messages are
    # dataclasses with a declared field, so `None` is the only representable
    # form of "the file did not carry usable content".
    bad_messages = [
        UserMessage(content=None, timestamp=_now_ms()),
        AssistantMessage(
            content=None,
            api="openai-completions",
            provider="openai",
            model="test-model",
            usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost()),
            stop_reason="stop",
            timestamp=_now_ms(),
        ),
        ToolResultMessage(
            content=None,
            tool_call_id="call_1",
            tool_name="web_search",
            is_error=False,
            timestamp=_now_ms(),
        ),
    ]

    for bad_message in bad_messages:
        message = session_entry_to_context_messages(_message_entry(bad_message))[0]
        assert message.role == bad_message.role
        assert message.content == []


def test_normalizes_null_content_when_loading_custom_message_entries() -> None:
    entry = CustomMessageEntry(
        id="entry-1",
        parent_id=None,
        timestamp=_iso_now(),
        custom_type="test",
        content=None,
        display=False,
        details=None,
    )

    message = session_entry_to_context_messages(entry)[0]
    assert message.role == "custom"
    assert message.content == []


def test_keeps_valid_message_content_untouched_when_loading_session_entries() -> None:
    message = session_entry_to_context_messages(_message_entry(UserMessage(content="hello", timestamp=_now_ms())))[0]
    assert message.role == "user"
    assert message.content == "hello"
