"""Python port of `packages/coding-agent/test/suite/regressions/4167-thinking-toggle-pending-tool-render.test.ts`.

The TypeScript test calls `InteractiveMode.prototype.renderSessionEntries`
with a hand-built `this`. This port drives the real `InteractiveMode` against
a `FakeTerminal` (the style `tests/test_interactive_mode.py` already
establishes) and replays the transcript through `_render_initial_messages`,
which is this port's `renderSessionEntries`/`renderSessionItems`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from interactive_harness import make_interactive_mode, strip_ansi_lines
from pi_agent.types import AgentToolResult, ToolExecutionEndEvent
from pi_ai.types import (
    AssistantMessage,
    Cost,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    now_ms,
)
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode

TOOL_CALL_ID = "tool-4167"
TOOL_NAME = "slow_tool"


def _empty_usage() -> Usage:
    return Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost())


def _assistant_tool_call_message() -> AssistantMessage:
    return AssistantMessage(
        api="test-api",
        provider="test-provider",
        model="test-model",
        content=[ToolCall(id=TOOL_CALL_ID, name=TOOL_NAME, arguments={"delayMs": 10_000})],
        usage=_empty_usage(),
        stop_reason="toolUse",
        timestamp=now_ms(),
    )


def _tool_result_message(text: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=TOOL_CALL_ID,
        tool_name=TOOL_NAME,
        content=[TextContent(text=text)],
        is_error=False,
        timestamp=now_ms(),
    )


def _render_chat(mode: InteractiveMode) -> str:
    return strip_ansi_lines(mode.chat_container.render(120))


async def test_keeps_unresolved_rendered_tool_calls_registered_for_live_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    mode.session_manager.append_message(_assistant_tool_call_message())

    mode._render_initial_messages()

    assert TOOL_CALL_ID in mode.pending_tools

    mode.is_initialized = True
    await mode._handle_event(
        ToolExecutionEndEvent(
            tool_call_id=TOOL_CALL_ID,
            tool_name=TOOL_NAME,
            result=AgentToolResult(content=[TextContent(text="FINAL_RESULT")], details=None),
            is_error=False,
        )
    )

    assert TOOL_CALL_ID not in mode.pending_tools
    assert "FINAL_RESULT" in _render_chat(mode)


async def test_does_not_keep_completed_historical_tool_calls_registered_as_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    mode.session_manager.append_message(_assistant_tool_call_message())
    mode.session_manager.append_message(_tool_result_message("HISTORICAL_RESULT"))

    mode._render_initial_messages()

    assert len(mode.pending_tools) == 0
    assert "HISTORICAL_RESULT" in _render_chat(mode)
