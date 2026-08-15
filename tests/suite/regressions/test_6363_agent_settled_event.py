"""Python port of `packages/coding-agent/test/suite/regressions/6363-agent-settled-event.test.ts`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness import create_harness, drain_extension_actions, get_user_texts
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.loader import ExtensionAPI

RETRY_SETTINGS = {"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}}


def create_wait_tool(released: asyncio.Event) -> AgentTool:
    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        await released.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    return AgentTool(
        name="wait",
        label="Wait",
        description="Wait until released",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


async def test_emits_one_agent_settled_after_automatic_retry(tmp_path: Path) -> None:
    extension_events: list[str] = []
    public_events: list[str] = []

    def factory(pi: ExtensionAPI) -> None:
        def on_agent_end(event, ctx) -> None:
            extension_events.append("agent_end")

        def on_agent_settled(event, ctx) -> None:
            extension_events.append(f"agent_settled:{ctx.is_idle()}")

        pi.on("agent_end", on_agent_end)
        pi.on("agent_settled", on_agent_settled)

    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS, extension_factories=[factory])
    try:

        def on_event(event) -> None:
            if event.type == "agent_settled":
                public_events.append("agent_settled")

        harness.session.subscribe(on_event)
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("recovered"),
            ]
        )

        await harness.session.prompt("test")

        assert [event.will_retry for event in harness.events_of_type("agent_end")] == [True, False]
        assert len(harness.events_of_type("agent_settled")) == 1
        assert extension_events == ["agent_end", "agent_end", "agent_settled:True"]
        assert public_events == ["agent_settled"]
    finally:
        harness.cleanup()


async def test_settles_only_after_agent_end_follow_ups_run(tmp_path: Path) -> None:
    state = {"queued": False}
    settled_idle_states: list[bool] = []

    def factory(pi: ExtensionAPI) -> None:
        def on_agent_end(event, ctx) -> None:
            if state["queued"]:
                return
            state["queued"] = True
            pi.send_user_message("status follow-up", {"deliverAs": "followUp"})

        def on_agent_settled(event, ctx) -> None:
            settled_idle_states.append(ctx.is_idle())

        pi.on("agent_end", on_agent_end)
        pi.on("agent_settled", on_agent_settled)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

        await harness.session.prompt("hello")

        assert get_user_texts(harness) == ["hello", "status follow-up"]
        assert len(harness.events_of_type("agent_end")) == 2
        assert len(harness.events_of_type("agent_settled")) == 1
        assert settled_idle_states == [True]
    finally:
        harness.cleanup()


async def test_extension_command_wait_for_idle_waits_for_session_settlement(tmp_path: Path) -> None:
    released = asyncio.Event()
    command_started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    command_results: list[bool] = []

    def factory(pi: ExtensionAPI) -> None:
        async def handler(args: str, ctx) -> None:
            if not command_started.done():
                command_started.set_result(None)
            await ctx.wait_for_idle()
            command_results.append(ctx.is_idle())

        pi.register_command("after-idle", handler=handler, description="Wait for idle")

    harness = await create_harness(
        tmp_path,
        tools=[create_wait_tool(released)],
        extension_factories=[factory],
    )
    try:
        tool_started: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_event(event) -> None:
            if event.type == "tool_execution_start" and event.tool_name == "wait" and not tool_started.done():
                tool_started.set_result(None)

        harness.session.subscribe(on_event)
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        prompt_task = asyncio.ensure_future(harness.session.prompt("start"))
        await tool_started
        command_task = asyncio.ensure_future(harness.session.prompt("/after-idle"))
        await command_started
        await asyncio.sleep(0)
        assert command_task.done() is False

        released.set()
        await asyncio.gather(prompt_task, command_task)
        await drain_extension_actions()

        assert command_results == [True]
        assert len(harness.events_of_type("agent_settled")) == 1
    finally:
        harness.cleanup()
