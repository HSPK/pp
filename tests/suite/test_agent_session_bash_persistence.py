"""Python port of `packages/coding-agent/test/suite/agent-session-bash-persistence.test.ts`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from harness import Harness, create_harness
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.bash_executor import BashResult


def get_entry_types(harness: Harness) -> list[str]:
    return [entry.type for entry in harness.session_manager.get_entries()]


@dataclass
class ControlledBashInvocation:
    signal: AbortSignal | None
    future: asyncio.Future[int | None]

    def finish(self) -> None:
        if not self.future.done():
            self.future.set_result(0)


@dataclass
class ControlledBashOperations:
    """Port of `createControlledBashOperations`: an exec that blocks until told to finish."""

    invocations: list[ControlledBashInvocation] = field(default_factory=list)

    async def exec(
        self,
        command: str,
        cwd: str,
        on_data: Callable[[bytes], None],
        signal: AbortSignal | None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> int | None:
        future: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()
        self.invocations.append(ControlledBashInvocation(signal=signal, future=future))
        return await future


async def test_records_bash_results_immediately_while_idle(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.session.record_bash_result(
            "echo hi",
            BashResult(output="hi", exit_code=0, cancelled=False, truncated=False),
        )

        assert harness.session.has_pending_bash_messages is False
        assert harness.session.messages[-1].role == "bashExecution"
        assert "message" in get_entry_types(harness)
    finally:
        harness.cleanup()


async def test_defers_bash_results_while_streaming_and_flushes_before_next_prompt(tmp_path: Path) -> None:
    tool_release = asyncio.Event()

    async def execute(tool_call_id, params, signal=None, on_update=None) -> AgentToolResult:
        await tool_release.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    wait_tool = AgentTool(
        name="wait",
        label="Wait",
        description="Wait for release",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(tmp_path, tools=[wait_tool])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"),
                faux_assistant_message("done"),
                faux_assistant_message("after flush"),
            ]
        )

        saw_tool_start = asyncio.Event()

        def on_event(event) -> None:
            if event.type == "tool_execution_start":
                saw_tool_start.set()

        harness.session.subscribe(on_event)

        first_prompt = asyncio.ensure_future(harness.session.prompt("start"))
        await asyncio.wait_for(saw_tool_start.wait(), timeout=5)
        harness.session.record_bash_result(
            "echo hi",
            BashResult(output="hi", exit_code=0, cancelled=False, truncated=False),
        )

        assert harness.session.has_pending_bash_messages is True
        assert not any(message.role == "bashExecution" for message in harness.session.messages)

        tool_release.set()
        await asyncio.wait_for(first_prompt, timeout=5)

        assert harness.session.has_pending_bash_messages is False
        assert any(message.role == "bashExecution" for message in harness.session.messages)

        await asyncio.wait_for(harness.session.prompt("next turn"), timeout=5)

        assert harness.session.has_pending_bash_messages is False
        assert any(message.role == "bashExecution" for message in harness.session.messages)
        assert len([entry for entry in get_entry_types(harness) if entry == "message"]) > 0
    finally:
        harness.cleanup()


async def test_executes_bash_commands_and_records_the_result(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        result = await asyncio.wait_for(harness.session.execute_bash("printf 'hello'"), timeout=10)

        assert "hello" in result.output
        assert harness.session.messages[-1].role == "bashExecution"
    finally:
        harness.cleanup()


async def test_cancels_running_bash_commands_with_abort_bash(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)

    class RejectingOperations:
        async def exec(self, command, cwd, on_data, signal, timeout=None, env=None) -> int | None:
            # TS attaches an `abort` listener and rejects; this port's
            # `AbortSignal` exposes `wait()` instead of DOM event listeners.
            if signal is not None:
                await signal.wait()
            raise RuntimeError("aborted")

    try:
        bash_task = asyncio.ensure_future(harness.session.execute_bash("sleep", operations=RejectingOperations()))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert harness.session.is_bash_running is True
        harness.session.abort_bash()

        result = await asyncio.wait_for(bash_task, timeout=5)
        assert result.cancelled is True
        assert harness.session.is_bash_running is False
    finally:
        harness.cleanup()


async def test_keeps_newer_bash_execution_tracked_when_older_finishes(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        operations = ControlledBashOperations()

        first_bash = asyncio.ensure_future(harness.session.execute_bash("first", operations=operations))
        second_bash = asyncio.ensure_future(harness.session.execute_bash("second", operations=operations))
        while len(operations.invocations) < 2:
            await asyncio.sleep(0)

        operations.invocations[0].finish()
        first_result = await asyncio.wait_for(first_bash, timeout=5)
        running_after_first_settles = harness.session.is_bash_running

        harness.session.abort_bash()
        second_was_aborted = operations.invocations[1].signal is not None and operations.invocations[1].signal.aborted
        operations.invocations[1].finish()
        second_result = await asyncio.wait_for(second_bash, timeout=5)

        assert first_result.cancelled is False
        assert running_after_first_settles is True
        assert second_was_aborted is True
        assert second_result.cancelled is True
        assert harness.session.is_bash_running is False
    finally:
        harness.cleanup()


async def test_aborts_all_active_bash_executions(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        operations = ControlledBashOperations()

        first_bash = asyncio.ensure_future(harness.session.execute_bash("first", operations=operations))
        second_bash = asyncio.ensure_future(harness.session.execute_bash("second", operations=operations))
        while len(operations.invocations) < 2:
            await asyncio.sleep(0)

        harness.session.abort_bash()
        aborted_signals = [
            invocation.signal is not None and invocation.signal.aborted for invocation in operations.invocations
        ]
        for invocation in operations.invocations:
            invocation.finish()
        results = await asyncio.wait_for(asyncio.gather(first_bash, second_bash), timeout=5)

        assert aborted_signals == [True, True]
        assert [result.cancelled for result in results] == [True, True]
        assert harness.session.is_bash_running is False
    finally:
        harness.cleanup()


async def test_persists_messages_in_order(tmp_path: Path) -> None:
    async def execute(tool_call_id, params, signal=None, on_update=None) -> AgentToolResult:
        text = str(params.get("text", "")) if isinstance(params, dict) else ""
        return AgentToolResult(content=[TextContent(text=f"echo:{text}")], details={"text": text})

    echo_tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )
    harness = await create_harness(tmp_path, tools=[echo_tool])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        await harness.session.send_custom_message("note", "hello", True, {"a": 1})
        await asyncio.wait_for(harness.session.prompt("start"), timeout=5)

        entries = harness.session_manager.get_entries()
        assert [entry.type for entry in entries] == [
            "custom_message",
            "message",
            "message",
            "message",
            "message",
        ]
        assert [message.role for message in harness.session.messages] == [
            "custom",
            "user",
            "assistant",
            "toolResult",
            "assistant",
        ]
    finally:
        harness.cleanup()


async def test_does_not_emit_message_end_for_bash_execution_messages(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        message_end_roles: list[str] = []

        def on_event(event) -> None:
            if event.type == "message_end":
                message_end_roles.append(event.message.role)

        harness.session.subscribe(on_event)

        harness.session.record_bash_result(
            "echo hi",
            BashResult(output="hi", exit_code=0, cancelled=False, truncated=False),
        )

        assert message_end_roles == []
    finally:
        harness.cleanup()


async def test_persists_aborted_assistant_messages(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.set_responses([faux_assistant_message("x" * 20_000)])

        saw_message_update = asyncio.Event()

        def on_event(event) -> None:
            if event.type == "message_update":
                saw_message_update.set()

        harness.session.subscribe(on_event)

        prompt_task = asyncio.ensure_future(harness.session.prompt("hi"))
        await asyncio.wait_for(saw_message_update.wait(), timeout=5)
        await asyncio.wait_for(harness.session.abort(), timeout=5)
        await asyncio.wait_for(prompt_task, timeout=5)

        entries = harness.session_manager.get_entries()
        last_entry = entries[-1]
        assert last_entry.type == "message"
        assert last_entry.message.role == "assistant"
        assert last_entry.message.stop_reason == "aborted"
    finally:
        harness.cleanup()


async def test_records_bash_output_through_custom_operations(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)

    class CustomOperations:
        async def exec(self, command, cwd, on_data, signal, timeout=None, env=None) -> int | None:
            on_data(b"hello from custom ops")
            return 0

    try:
        result = await asyncio.wait_for(
            harness.session.execute_bash("custom", operations=CustomOperations()), timeout=5
        )

        assert "hello from custom ops" in result.output
        assert harness.session.messages[-1].role == "bashExecution"
    finally:
        harness.cleanup()


async def test_streams_bash_output_to_callback_and_session_events(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)

    class CustomOperations:
        async def exec(self, command, cwd, on_data, signal, timeout=None, env=None) -> int | None:
            on_data(b"hello ")
            on_data(b"world")
            return 0

    try:
        callback_deltas: list[str] = []
        event_updates: list[tuple[str | None, str]] = []

        def on_event(event) -> None:
            if event.type == "bash_execution_update":
                event_updates.append((event.id, event.delta))

        unsubscribe = harness.session.subscribe(on_event)

        await asyncio.wait_for(
            harness.session.execute_bash(
                "custom",
                callback_deltas.append,
                id="bash-1",
                operations=CustomOperations(),
            ),
            timeout=5,
        )
        unsubscribe()

        assert callback_deltas == ["hello ", "world"]
        assert event_updates == [("bash-1", "hello "), ("bash-1", "world")]
    finally:
        harness.cleanup()
