"""Python port of `packages/coding-agent/test/suite/regressions/1717-2113-agent-session-event-settlement.test.ts`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness import Harness, create_harness
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.loader import ExtensionAPI


def create_echo_tool() -> AgentTool:
    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        text = str(params.get("text", "")) if isinstance(params, dict) else ""
        return AgentToolResult(content=[TextContent(text=text)], details={"text": text})

    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )


def _branch_messages(harness: Harness) -> list[object]:
    return [entry.message for entry in harness.session_manager.get_branch() if entry.type == "message"]


async def test_keeps_persisted_message_order_when_message_end_handlers_yield(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def on_message_end(event, ctx) -> None:
            if getattr(event.message, "role", None) == "assistant":
                await asyncio.sleep(0.02)

        pi.on("message_end", on_message_end)

    harness = await create_harness(tmp_path, tools=[create_echo_tool()], extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("echo", {"text": "one"}), faux_tool_call("echo", {"text": "two"})],
                    stop_reason="toolUse",
                ),
                faux_assistant_message("done"),
            ]
        )
        await harness.session.prompt("run tools")

        messages = _branch_messages(harness)
        roles = [getattr(message, "role", "") for message in messages]
        assert roles == ["user", "assistant", "toolResult", "toolResult", "assistant"]
        first_tool_result_index = roles.index("toolResult")
        assert first_tool_result_index > 0
        assert roles[first_tool_result_index - 1] == "assistant"
    finally:
        harness.cleanup()


async def test_runs_tool_call_handlers_after_assistant_message_is_settled(tmp_path: Path) -> None:
    branch_roles_at_tool_call: list[list[str]] = []
    harness_holder: dict[str, Harness] = {}

    def factory(pi: ExtensionAPI) -> None:
        def on_tool_call(event, ctx) -> None:
            harness = harness_holder["harness"]
            branch_roles_at_tool_call.append(
                [getattr(message, "role", "") for message in _branch_messages(harness)],
            )

        pi.on("tool_call", on_tool_call)

    harness = await create_harness(tmp_path, tools=[create_echo_tool()], extension_factories=[factory])
    harness_holder["harness"] = harness
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )
        await harness.session.prompt("run tool")

        assert branch_roles_at_tool_call == [["user", "assistant"]]
    finally:
        harness.cleanup()
