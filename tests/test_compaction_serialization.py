"""Python port of `packages/coding-agent/test/compaction-serialization.test.ts`.

`serializeConversation` lives in `coding-agent/src/core/compaction/utils.ts`
upstream; in this port it is shared code in `pi_agent.harness.compaction.utils`
and re-used by `pi_coding_agent.core.compaction`.
"""

from __future__ import annotations

import time

from pi_agent.harness.compaction.utils import serialize_conversation
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Message,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _now() -> int:
    return int(time.time() * 1000)


def test_truncates_long_tool_results() -> None:
    long_content = "x" * 5000
    messages: list[Message] = [
        ToolResultMessage(
            tool_call_id="tc1",
            tool_name="read",
            content=[TextContent(text=long_content)],
            is_error=False,
            timestamp=_now(),
        )
    ]

    result = serialize_conversation(messages)

    assert "[Tool result]:" in result
    assert "[... 3000 more characters truncated]" in result
    assert "x" * 3000 not in result
    assert "x" * 2000 in result


def test_does_not_truncate_short_tool_results() -> None:
    short_content = "x" * 1500
    messages: list[Message] = [
        ToolResultMessage(
            tool_call_id="tc1",
            tool_name="read",
            content=[TextContent(text=short_content)],
            is_error=False,
            timestamp=_now(),
        )
    ]

    result = serialize_conversation(messages)

    assert result == f"[Tool result]: {short_content}"
    assert "truncated" not in result


def test_does_not_truncate_assistant_or_user_messages() -> None:
    long_text = "y" * 5000
    messages: list[Message] = [
        UserMessage(content=[TextContent(text=long_text)], timestamp=_now()),
        AssistantMessage(
            content=[TextContent(text=long_text)],
            api="anthropic-messages",
            provider="anthropic",
            model="test",
            usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost()),
            stop_reason="stop",
            timestamp=_now(),
        ),
    ]

    result = serialize_conversation(messages)

    assert "truncated" not in result
    assert long_text in result
