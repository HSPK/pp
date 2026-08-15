"""Python port of `packages/coding-agent/test/agent-session-retry.test.ts`.

The TypeScript test builds its `Agent` by hand around a `MockAssistantStream`
whose `streamFn` counts calls and decides per call whether to emit an error or a
success message. This port uses the shared `tests/suite/harness.py` + the faux
provider instead: the faux provider is a scripted response queue, so "fail the
first N calls" becomes "queue N error messages followed by a success message",
and `harness.faux.state.call_count` is the direct counterpart of the TS
`callCount` closure.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.agent_session import AgentSession

sys.path.insert(0, str(Path(__file__).resolve().parent / "suite"))

from harness import create_harness

RETRY_SETTINGS: dict[str, Any] = {"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}}


def _error_message(text: str = "overloaded_error"):
    return faux_assistant_message("", stop_reason="error", error_message=text)


def _retry_trace(session: AgentSession) -> list[str]:
    """Port of the TS `events` array collected via `session.subscribe(...)`."""
    trace: list[str] = []

    def on_event(event: Any) -> None:
        if event.type == "auto_retry_start":
            trace.append(f"start:{event.attempt}")
        if event.type == "auto_retry_end":
            trace.append(f"end:success={str(event.success).lower()}")

    session.subscribe(on_event)
    return trace


async def test_retries_after_a_transient_error_and_succeeds(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS)
    try:
        harness.set_responses([_error_message(), faux_assistant_message("Success")])
        events = _retry_trace(harness.session)

        await harness.session.prompt("Test")

        assert harness.faux.state.call_count == 2
        assert events == ["start:1", "end:success=true"]
        assert harness.session.is_retrying is False
    finally:
        harness.cleanup()


async def test_exhausts_max_retries_and_emits_failure(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 1}},
    )
    try:
        # TS uses `failCount: 99`, i.e. "always fail"; the scripted provider only
        # needs enough failures to outlast `maxRetries: 2` (3 calls total).
        harness.set_responses([_error_message() for _ in range(6)])
        events = _retry_trace(harness.session)

        await harness.session.prompt("Test")

        assert harness.faux.state.call_count == 3
        assert "start:1" in events
        assert "start:2" in events
        assert "end:success=false" in events
        assert harness.session.is_retrying is False
    finally:
        harness.cleanup()


async def test_prompt_waits_for_retry_when_assistant_message_end_is_delayed(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS)
    try:
        harness.set_responses([_error_message(), faux_assistant_message("Success")])

        # TS wraps `_emitExtensionEvent` with a real 40ms `setTimeout` on assistant
        # `message_end`. A wall-clock sleep is banned here, so the delay is
        # expressed as event-loop yields instead: it interleaves the extension
        # emit with the retry scheduling exactly the same way without burning time.
        original = harness.session._emit_extension_event

        async def delayed(event: Any) -> None:
            if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
                for _ in range(20):
                    await asyncio.sleep(0)
            await original(event)

        harness.session._emit_extension_event = delayed  # type: ignore[method-assign]

        await harness.session.prompt("Test")

        assert harness.faux.state.call_count == 2
        assert harness.session.is_retrying is False
    finally:
        harness.cleanup()


async def test_retries_provider_network_error_failures(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS)
    try:
        harness.set_responses(
            [
                _error_message("Provider finish_reason: network_error"),
                faux_assistant_message("Recovered after retry"),
            ]
        )
        events = _retry_trace(harness.session)

        await harness.session.prompt("Test")

        assert harness.faux.state.call_count == 2
        assert events == ["start:1", "end:success=true"]
    finally:
        harness.cleanup()


async def test_prompt_waits_for_full_agent_loop_when_retry_produces_tool_calls(tmp_path: Path) -> None:
    # Regression: when auto-retry fires and the retry response includes tool_use,
    # session.prompt() must wait for the entire tool loop to finish before returning.
    tool_executed = {"value": False}

    async def execute(tool_call_id, params, signal=None, on_update=None) -> AgentToolResult:
        tool_executed["value"] = True
        return AgentToolResult(content=[TextContent(text="echoed")], details=None)

    echo_tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )

    harness = await create_harness(tmp_path, settings=RETRY_SETTINGS, tools=[echo_tool])
    try:
        harness.set_responses(
            [
                _error_message(),
                faux_assistant_message(
                    [
                        TextContent(text="Looking that up now."),
                        faux_tool_call("echo", {"text": "hello"}, id="call_1"),
                    ],
                    stop_reason="toolUse",
                ),
                faux_assistant_message("Final answer."),
                faux_assistant_message("Follow-up answer."),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("Test"), timeout=5)

        assert harness.faux.state.call_count == 3
        assert tool_executed["value"] is True
        assert harness.session.is_streaming is False

        await asyncio.wait_for(harness.session.prompt("Follow-up"), timeout=5)
        assert harness.faux.state.call_count == 4
    finally:
        harness.cleanup()
