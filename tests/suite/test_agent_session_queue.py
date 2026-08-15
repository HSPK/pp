"""Python port of `packages/coding-agent/test/suite/agent-session-queue.test.ts`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from harness import (
    Harness,
    create_harness,
    drain_extension_actions,
    get_assistant_texts,
    get_message_text,
    get_user_texts,
    wait_until,
)
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.loader import ExtensionAPI


@dataclass
class WaitingHarness:
    harness: Harness
    release_tool_execution: Any
    prompt_task: asyncio.Task[None]
    wait_for_tool_start: asyncio.Future[None]


async def create_waiting_harness(
    tmp_path: Path,
    *,
    tools: list[AgentTool] | None = None,
    extension_factories: list[Any] | None = None,
) -> WaitingHarness:
    """Port of `createWaitingHarness`: a harness stalled inside a `wait` tool."""
    tool_release = asyncio.Event()

    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        await tool_release.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    wait_tool = AgentTool(
        name="wait",
        label="Wait",
        description="Wait for release",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(
        tmp_path,
        tools=[wait_tool, *(tools or [])],
        extension_factories=extension_factories,
    )

    started: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def on_event(event) -> None:
        if event.type == "tool_execution_start" and event.tool_name == "wait" and not started.done():
            started.set_result(None)

    harness.session.subscribe(on_event)

    return WaitingHarness(
        harness=harness,
        release_tool_execution=tool_release.set,
        prompt_task=asyncio.ensure_future(harness.session.prompt("start")),
        wait_for_tool_start=started,
    )


async def test_dispatches_extension_commands_immediately_when_prompted_while_idle(tmp_path: Path) -> None:
    command_runs: list[str] = []

    def factory(pi: ExtensionAPI) -> None:
        async def handler(args: str, ctx) -> None:
            command_runs.append(args)

        pi.register_command("testcmd", handler=handler, description="Test command")

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        await asyncio.wait_for(harness.session.prompt("/testcmd hello world"), timeout=10)

        assert command_runs == ["hello world"]
        assert harness.get_pending_response_count() == 0
        assert harness.session.messages == []
    finally:
        harness.cleanup()


async def test_delivers_extension_origin_steering_messages_before_the_next_llm_call(tmp_path: Path) -> None:
    seen_api: list[ExtensionAPI] = []

    def factory(pi: ExtensionAPI) -> None:
        seen_api.append(pi)

    waiting = await create_waiting_harness(tmp_path, extension_factories=[factory])
    harness = waiting.harness
    try:

        def second(context, options, state, model):
            saw_steer = any(
                getattr(m, "role", "") == "user" and get_message_text(m) == "steer now" for m in context.messages
            )
            return faux_assistant_message("saw steer" if saw_steer else "missing steer")

        harness.set_responses([faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"), second])

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await asyncio.sleep(0)

        assert seen_api
        seen_api[0].send_user_message("steer now", {"deliverAs": "steer"})
        # `pi.send_user_message` is fire-and-forget; wait for the message to
        # actually reach the steering queue rather than sleeping a fixed delay
        # a loaded host could overshoot (which would release the tool first and
        # turn a scheduling stall into a spurious "missing steer").
        await wait_until(
            lambda: "steer now" in harness.session.get_steering_messages(),
            what="the steer message to reach the steering queue",
        )
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert get_user_texts(harness) == ["start", "steer now"]
        assert "saw steer" in get_assistant_texts(harness)
    finally:
        harness.cleanup()


async def test_delivers_follow_up_messages_only_after_the_current_run_finishes(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        assistant_seen: list[str] = []

        def second(context, options, state, model):
            assistant_seen.extend(
                "\n".join(part.text for part in m.content if isinstance(part, TextContent))
                for m in context.messages
                if getattr(m, "role", "") == "assistant"
            )
            return faux_assistant_message("follow-up response")

        harness.set_responses([faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"), second])

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.follow_up("after current run")
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert get_user_texts(harness) == ["start", "after current run"]
        assert "" in assistant_seen
        assert "follow-up response" in get_assistant_texts(harness)
    finally:
        harness.cleanup()


async def test_delivers_multiple_steering_messages_in_order_in_one_at_a_time_mode(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("handled steer 1"),
                faux_assistant_message("handled steer 2"),
            ]
        )

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.steer("steer 1")
        await harness.session.steer("steer 2")
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert get_user_texts(harness) == ["start", "steer 1", "steer 2"]
        assert get_assistant_texts(harness) == ["", "handled steer 1", "handled steer 2"]
    finally:
        harness.cleanup()


async def test_delivers_multiple_follow_up_messages_in_order_in_one_at_a_time_mode(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("original turn complete"),
                faux_assistant_message("handled follow-up 1"),
                faux_assistant_message("handled follow-up 2"),
            ]
        )

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.follow_up("follow-up 1")
        await harness.session.follow_up("follow-up 2")
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert get_user_texts(harness) == ["start", "follow-up 1", "follow-up 2"]
        assert get_assistant_texts(harness) == [
            "",
            "original turn complete",
            "handled follow-up 1",
            "handled follow-up 2",
        ]
    finally:
        harness.cleanup()


async def test_delivers_all_steering_messages_in_one_batch_in_all_mode(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        harness.session.set_steering_mode("all")
        batched: list[list[str]] = []

        def second(context, options, state, model):
            batched.append([get_message_text(m) for m in context.messages if getattr(m, "role", "") == "user"])
            return faux_assistant_message("batched steer response")

        harness.set_responses([faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"), second])

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.steer("steer 1")
        await harness.session.steer("steer 2")
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert batched[-1] == ["start", "steer 1", "steer 2"]
        assert get_assistant_texts(harness) == ["", "batched steer response"]
    finally:
        harness.cleanup()


async def test_delivers_all_follow_up_messages_in_one_batch_in_all_mode(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        harness.session.set_follow_up_mode("all")
        batched: list[list[str]] = []

        def third(context, options, state, model):
            batched.append([get_message_text(m) for m in context.messages if getattr(m, "role", "") == "user"])
            return faux_assistant_message("batched follow-up response")

        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("original turn complete"),
                third,
            ]
        )

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.follow_up("follow-up 1")
        await harness.session.follow_up("follow-up 2")
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert batched[-1] == ["start", "follow-up 1", "follow-up 2"]
        assert get_assistant_texts(harness) == ["", "original turn complete", "batched follow-up response"]
    finally:
        harness.cleanup()


def _saw_user_text(context, text: str) -> bool:
    return any(
        getattr(message, "role", "") == "user"
        and not isinstance(message.content, str)
        and any(isinstance(part, TextContent) and part.text == text for part in message.content)
        for message in context.messages
    )


async def test_queues_custom_messages_with_deliver_as_steer_while_streaming(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        saw: list[bool] = []

        def second(context, options, state, model):
            saw.append(_saw_user_text(context, "steer custom"))
            return faux_assistant_message("done")

        harness.set_responses([faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"), second])

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.send_custom_message("queue-test", "steer custom", True, {"value": 1}, deliver_as="steer")
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert saw[-1] is True
        assert any(
            getattr(message, "role", "") == "custom" and getattr(message, "custom_type", "") == "queue-test"
            for message in harness.session.messages
        )
    finally:
        harness.cleanup()


async def test_queues_custom_messages_with_deliver_as_follow_up_while_streaming(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        saw: list[bool] = []

        def third(context, options, state, model):
            saw.append(_saw_user_text(context, "follow-up custom"))
            return faux_assistant_message("done")

        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("original turn complete"),
                third,
            ]
        )

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.send_custom_message(
            "queue-test", "follow-up custom", True, {"value": 1}, deliver_as="followUp"
        )
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert saw[-1] is True
        assert any(
            getattr(message, "role", "") == "custom" and getattr(message, "custom_type", "") == "queue-test"
            for message in harness.session.messages
        )
    finally:
        harness.cleanup()


async def test_injects_next_turn_custom_messages_into_the_next_prompt(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        saw: list[bool] = []

        await harness.session.send_custom_message("next-turn", "carry this", True, {}, deliver_as="nextTurn")

        def respond(context, options, state, model):
            saw.append(_saw_user_text(context, "carry this"))
            return faux_assistant_message("done")

        harness.set_responses([respond])

        await asyncio.wait_for(harness.session.prompt("normal prompt"), timeout=10)

        assert saw[-1] is True
        assert [getattr(m, "role", "") for m in harness.session.messages] == ["user", "custom", "assistant"]
    finally:
        harness.cleanup()


async def test_updates_pending_message_count_and_removes_queued_text_before_message_start(tmp_path: Path) -> None:
    waiting = await create_waiting_harness(tmp_path)
    harness = waiting.harness
    try:
        counts: list[int] = []

        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        def on_event(event) -> None:
            if (
                event.type == "message_start"
                and getattr(event.message, "role", "") == "user"
                and get_message_text(event.message) == "queued"
            ):
                counts.append(harness.session.pending_message_count)

        harness.session.subscribe(on_event)

        await asyncio.wait_for(waiting.wait_for_tool_start, timeout=10)
        await harness.session.steer("queued")
        assert harness.session.pending_message_count == 1
        waiting.release_tool_execution()
        await asyncio.wait_for(waiting.prompt_task, timeout=10)

        assert counts == [0]
        assert harness.session.pending_message_count == 0
    finally:
        harness.cleanup()


def _command_factory(pi: ExtensionAPI) -> None:
    async def handler(args: str, ctx) -> None:
        return None

    pi.register_command("testcmd", handler=handler, description="Test command")


async def test_throws_when_queueing_an_extension_command_with_steer(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, extension_factories=[_command_factory])
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await harness.session.steer("/testcmd queued")
        assert (
            'Extension command "/testcmd" cannot be queued. Use prompt() or execute the command when not streaming.'
            in str(excinfo.value)
        )
    finally:
        harness.cleanup()


async def test_throws_when_queueing_an_extension_command_with_follow_up(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, extension_factories=[_command_factory])
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await harness.session.follow_up("/testcmd queued")
        assert (
            'Extension command "/testcmd" cannot be queued. Use prompt() or execute the command when not streaming.'
            in str(excinfo.value)
        )
    finally:
        harness.cleanup()


async def test_delivers_follow_ups_queued_during_agent_end(tmp_path: Path) -> None:
    sent: list[bool] = []

    def factory(pi: ExtensionAPI) -> None:
        async def on_agent_end(event, ctx) -> None:
            if sent:
                return
            sent.append(True)
            pi.send_user_message("conflict report", {"deliverAs": "followUp"})

        pi.on("agent_end", on_agent_end)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.set_responses([faux_assistant_message("reply"), faux_assistant_message("follow-up reply")])

        await asyncio.wait_for(harness.session.prompt("hello"), timeout=10)
        # The extension's `send_user_message` is a spawned task here (see
        # `drain_extension_actions`), so it has to be drained before the agent
        # can go idle for good.
        await asyncio.wait_for(drain_extension_actions(), timeout=10)
        await asyncio.wait_for(harness.session.agent.wait_for_idle(), timeout=10)

        assert get_user_texts(harness) == ["hello", "conflict report"]
    finally:
        harness.cleanup()
