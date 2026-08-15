"""Python port of `packages/coding-agent/test/suite/regressions/2023-queued-slash-command-followup.test.ts`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness import create_harness, drain_extension_actions, get_assistant_texts, get_user_texts
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.loader import ExtensionAPI


async def test_queued_extension_slash_command_followup_is_raw_user_text(tmp_path: Path) -> None:
    extension_api: dict[str, ExtensionAPI] = {}
    command_runs: list[str] = []
    tool_release = asyncio.Event()

    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        await tool_release.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    wait_tool = AgentTool(
        name="wait",
        label="Wait",
        description="Wait for the test to release execution",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    def factory(pi: ExtensionAPI) -> None:
        extension_api["pi"] = pi

        async def handler(args: str, ctx) -> None:
            command_runs.append(args)

        pi.register_command("testcmd", handler=handler, description="Test command")

    harness = await create_harness(tmp_path, tools=[wait_tool], extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("first turn complete"),
                faux_assistant_message("queued follow-up handled by model"),
            ]
        )

        saw_tool_start: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_event(event) -> None:
            if event.type == "tool_execution_start" and event.tool_name == "wait" and not saw_tool_start.done():
                saw_tool_start.set_result(None)

        harness.session.subscribe(on_event)

        prompt_task = asyncio.ensure_future(harness.session.prompt("start"))
        await saw_tool_start
        await asyncio.sleep(0)

        extension_api["pi"].send_user_message("/testcmd queued", {"deliverAs": "followUp"})
        await drain_extension_actions()
        tool_release.set()
        await prompt_task

        assert command_runs == []
        assert get_user_texts(harness) == ["start", "/testcmd queued"]
        assert "queued follow-up handled by model" in get_assistant_texts(harness)
    finally:
        harness.cleanup()
