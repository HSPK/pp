"""Python port of `packages/coding-agent/test/suite/agent-session-retry-events.test.ts`."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from harness import Harness, create_harness
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_thinking, faux_tool_call
from pi_ai.types import TextContent

RETRY_SETTINGS = {"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}}


def normalize_event_order(harness: Harness) -> list[str]:
    """Port of `normalizeEventOrder`: collapses runs of `message_update`."""
    normalized: list[str] = []
    for event in harness.events:
        if event.type in ("message_start", "message_end"):
            label = f"{event.type}:{getattr(event.message, 'role', '')}"
        elif event.type in ("tool_execution_start", "tool_execution_end"):
            label = f"{event.type}:{event.tool_name}"
        else:
            label = event.type
        if label == "message_update" and normalized and normalized[-1] == "message_update":
            continue
        normalized.append(label)
    return normalized


def _echo_tool(tool_runs: list[str]) -> AgentTool:
    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        text = str(params.get("text", "")) if isinstance(params, dict) else ""
        tool_runs.append(text)
        return AgentToolResult(content=[TextContent(text=f"echo:{text}")], details={"text": text})

    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )


def _collect_retry_events(harness: Harness) -> list[str]:
    retry_events: list[str] = []

    def on_event(event) -> None:
        if event.type == "auto_retry_start":
            retry_events.append(f"start:{event.attempt}")
        if event.type == "auto_retry_end":
            retry_events.append(f"end:{event.success}")

    harness.session.subscribe(on_event)
    return retry_events


async def test_retries_after_a_transient_error_and_succeeds(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS)
    try:
        retry_events = _collect_retry_events(harness)
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("recovered"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert retry_events == ["start:1", "end:True"]
        assert [event.will_retry for event in harness.events_of_type("agent_end")] == [True, False]
        assert harness.faux.state.call_count == 2
        assert harness.session.is_retrying is False
    finally:
        harness.cleanup()


async def test_retries_multiple_transient_failures_and_succeeds_on_the_final_attempt(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS)
    try:
        retry_events = _collect_retry_events(harness)
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("success"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert retry_events == ["start:1", "start:2", "end:True"]
        assert harness.faux.state.call_count == 3
    finally:
        harness.cleanup()


async def test_exhausts_max_retries_and_emits_a_failure_event(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 1}})
    try:
        retry_events = _collect_retry_events(harness)
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert retry_events == ["start:1", "start:2", "end:False"]
        assert [event.will_retry for event in harness.events_of_type("agent_end")] == [True, True, False]
        assert harness.faux.state.call_count == 3
        assert harness.session.is_retrying is False
    finally:
        harness.cleanup()


async def test_prompt_waits_for_retry_completion_when_message_end_handling_is_delayed(tmp_path: Path) -> None:
    def factory(pi) -> None:
        async def on_message_end(event, ctx) -> None:
            if getattr(event.message, "role", "") == "assistant":
                await asyncio.sleep(0.04)

        pi.on("message_end", on_message_end)

    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS, extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message("recovered"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert harness.faux.state.call_count == 2
        assert harness.session.is_retrying is False
    finally:
        harness.cleanup()


async def test_does_not_retry_when_retry_is_disabled(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings={"retry": {"enabled": False}})
    try:
        harness.set_responses([faux_assistant_message("", stop_reason="error", error_message="overloaded_error")])

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert harness.faux.state.call_count == 1
        assert harness.events_of_type("auto_retry_start") == []
    finally:
        harness.cleanup()


async def test_does_not_retry_non_retryable_errors(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS)
    try:
        harness.set_responses([faux_assistant_message("", stop_reason="error", error_message="invalid_api_key")])

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert harness.faux.state.call_count == 1
        assert harness.events_of_type("auto_retry_start") == []
    finally:
        harness.cleanup()


async def test_cancels_retry_sleep_when_abort_retry_is_called(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 100}})
    try:
        harness.set_responses([faux_assistant_message("", stop_reason="error", error_message="overloaded_error")])

        saw_retry_start: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_event(event) -> None:
            if event.type == "auto_retry_start" and not saw_retry_start.done():
                saw_retry_start.set_result(None)

        harness.session.subscribe(on_event)

        prompt_task = asyncio.ensure_future(harness.session.prompt("test"))
        await asyncio.wait_for(saw_retry_start, timeout=10)
        harness.session.abort_retry()
        await asyncio.wait_for(prompt_task, timeout=10)

        assert harness.session.is_retrying is False
        assert "Retry cancelled" in [event.final_error for event in harness.events_of_type("auto_retry_end")]
        assert harness.faux.state.call_count == 1
    finally:
        harness.cleanup()


async def test_waits_for_the_full_loop_when_retry_recovery_produces_tool_calls(tmp_path: Path) -> None:
    tool_runs: list[str] = []
    harness = await create_harness(tmp_path, tools=[_echo_tool(tool_runs)], settings=RETRY_SETTINGS)
    try:
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                faux_assistant_message("final answer"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("test"), timeout=10)

        assert harness.faux.state.call_count == 3
        assert tool_runs == ["hello"]
        assert harness.session.is_streaming is False
        await asyncio.wait_for(harness.session.prompt("follow-up"), timeout=10)
        assert harness.faux.state.call_count == 4
    finally:
        harness.cleanup()


async def test_emits_extension_events_before_public_event_subscribers(tmp_path: Path) -> None:
    order: list[str] = []

    def factory(pi) -> None:
        async def on_message(event, ctx) -> None:
            order.append(f"extension:{event.type}:{getattr(event.message, 'role', '')}")

        pi.on("message_start", on_message)
        pi.on("message_end", on_message)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:

        def on_event(event) -> None:
            if event.type in ("message_start", "message_end"):
                order.append(f"public:{event.type}:{getattr(event.message, 'role', '')}")

        harness.session.subscribe(on_event)
        harness.set_responses([faux_assistant_message("done")])

        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assert order == [
            "extension:message_start:user",
            "public:message_start:user",
            "extension:message_end:user",
            "public:message_end:user",
            "extension:message_start:assistant",
            "public:message_start:assistant",
            "extension:message_end:assistant",
            "public:message_end:assistant",
        ]
    finally:
        harness.cleanup()


async def test_emits_the_expected_event_order_for_a_single_prompt(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.set_responses([faux_assistant_message("hello")])

        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assert normalize_event_order(harness) == [
            "agent_start",
            "turn_start",
            "message_start:user",
            "message_end:user",
            "message_start:assistant",
            "message_update",
            "message_end:assistant",
            "turn_end",
            "agent_end",
            "agent_settled",
        ]
    finally:
        harness.cleanup()


async def test_emits_the_expected_event_order_for_a_tool_call_turn(tmp_path: Path) -> None:
    tool_runs: list[str] = []
    harness = await create_harness(tmp_path, tools=[_echo_tool(tool_runs)])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assert tool_runs == ["hello"]
        assert normalize_event_order(harness) == [
            "agent_start",
            "turn_start",
            "message_start:user",
            "message_end:user",
            "message_start:assistant",
            "message_update",
            "message_end:assistant",
            "tool_execution_start:echo",
            "tool_execution_end:echo",
            "message_start:toolResult",
            "message_end:toolResult",
            "turn_end",
            "turn_start",
            "message_start:assistant",
            "message_update",
            "message_end:assistant",
            "turn_end",
            "agent_end",
            "agent_settled",
        ]
    finally:
        harness.cleanup()


async def test_emits_streaming_deltas_for_text_thinking_and_tool_calls(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.set_responses(
            [
                faux_assistant_message(
                    [
                        faux_thinking("plan"),
                        TextContent(text="answer"),
                        faux_tool_call("echo", {"text": "hello"}),
                    ],
                    stop_reason="toolUse",
                )
            ]
        )

        with contextlib.suppress(Exception):
            await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        update_types = [event.assistant_message_event.type for event in harness.events_of_type("message_update")]
        assert "thinking_delta" in update_types
        assert "text_delta" in update_types
        assert "toolcall_delta" in update_types
    finally:
        harness.cleanup()


async def test_emits_agent_end_for_error_responses(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.set_responses([faux_assistant_message("", stop_reason="error", error_message="broken")])

        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assert len(harness.events_of_type("agent_end")) == 1
        assert harness.events[-1].type == "agent_settled"
    finally:
        harness.cleanup()


async def test_emits_agent_end_for_aborted_runs_and_persists_the_aborted_message(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.set_responses([faux_assistant_message("x" * 20_000)])

        saw_update: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_event(event) -> None:
            if event.type == "message_update" and not saw_update.done():
                saw_update.set_result(None)

        harness.session.subscribe(on_event)

        prompt_task = asyncio.ensure_future(harness.session.prompt("hi"))
        await asyncio.wait_for(saw_update, timeout=10)
        await asyncio.wait_for(harness.session.abort(), timeout=10)
        await asyncio.wait_for(prompt_task, timeout=10)

        assert len(harness.events_of_type("agent_end")) == 1
        assert harness.events[-1].type == "agent_settled"
        last_message = harness.session.messages[-1]
        assert getattr(last_message, "role", "") == "assistant"
        assert last_message.stop_reason == "aborted"
    finally:
        harness.cleanup()
