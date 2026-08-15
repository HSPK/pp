"""Python port of `packages/coding-agent/test/suite/regressions/5998-blocked-tool-terminate.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import create_harness, get_assistant_texts
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import ToolCallEventResult


async def test_lets_a_tool_call_handler_terminate_the_run_after_blocking_execution(tmp_path: Path) -> None:
    async def execute(_tool_call_id: str, _params: dict, _signal=None, _on_update=None) -> AgentToolResult:
        raise AssertionError("tool should have been blocked")

    echo_tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )

    def factory(pi: ExtensionAPI) -> None:
        async def on_tool_call(_event, _ctx) -> ToolCallEventResult:
            return ToolCallEventResult(block=True, reason="Blocked by terminating policy", terminate=True)

        pi.on("tool_call", on_tool_call)

    harness = await create_harness(tmp_path, tools=[echo_tool], extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                faux_assistant_message("should not run"),
            ]
        )

        await harness.session.prompt("hi")

        assert harness.get_pending_response_count() == 1
        assert "should not run" not in get_assistant_texts(harness)
        tool_end_events = harness.events_of_type("tool_execution_end")
        assert tool_end_events
        assert getattr(tool_end_events[0].result, "terminate", None) is True
        assert any(
            getattr(message, "role", None) == "toolResult" and message.is_error for message in harness.session.messages
        )
    finally:
        harness.cleanup()
