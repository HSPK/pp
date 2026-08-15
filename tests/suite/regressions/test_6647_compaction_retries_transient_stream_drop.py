"""Python port of `packages/coding-agent/test/suite/regressions/6647-compaction-retries-transient-stream-drop.test.ts`.

Compaction used to run a single non-retried summarization call, so a transient
mid-stream socket death (`terminated`) failed the whole compaction. These
cases pin that summarization reuses `settings.retry`, emits the
`summarization_retry_*` events, and that aborts / non-retryable errors are not
retried.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from harness import Harness, create_harness
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    TextContent,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


def create_usage(total_tokens: int) -> Usage:
    return Usage(
        input=total_tokens,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=total_tokens,
        cost=Cost(),
    )


def seed_compactable_session(harness: Harness) -> None:
    harness.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    now = now_ms()
    harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
    )
    model = harness.get_model()
    assert model is not None
    assistant = faux_assistant_message("", stop_reason="stop", timestamp=now - 500)
    assistant.api = model.api
    assistant.provider = model.provider
    assistant.model = model.id
    assistant.usage = create_usage(100)
    assistant.content = [TextContent(text="assistant response to compact")]
    harness.session_manager.append_message(assistant)
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages


def use_scripted_stream_fn(harness: Harness, script: list[AssistantMessage]) -> list[int]:
    """Port of `useScriptedStreamFn`: answer with `script[n]` on call n.

    Returns a single-element list holding the call count (TS returns a closure).
    """
    call_count = [0]

    def stream_fn(model, context, options=None, **kwargs) -> AssistantMessageEventStream:
        message = script[call_count[0]] if call_count[0] < len(script) else script[-1]
        call_count[0] += 1
        stream = AssistantMessageEventStream()

        async def push() -> None:
            response = _clone_for_model(message, model)
            if response.stop_reason in ("error", "aborted"):
                stream.push(ErrorEvent(reason=response.stop_reason, error=response))
                stream.end(response)
            else:
                stream.push(DoneEvent(reason=response.stop_reason, message=response))
                stream.end(response)

        # TS defers the push with `queueMicrotask`; the closest asyncio analogue
        # is a task that runs on the next loop iteration.
        asyncio.get_running_loop().create_task(push())
        return stream

    harness.session.agent.stream_function = stream_fn
    return call_count


def _clone_for_model(message: AssistantMessage, model: Any) -> AssistantMessage:
    clone = AssistantMessage(
        content=list(message.content),
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=message.usage,
        stop_reason=message.stop_reason,
        error_message=message.error_message,
        timestamp=message.timestamp,
    )
    return clone


def error_message(text: str) -> AssistantMessage:
    message = faux_assistant_message("", stop_reason="error", error_message=text)
    message.usage = create_usage(10)
    return message


async def test_retries_transient_terminated_error_and_compacts(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        harness.settings_manager.apply_overrides({"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 0}})

        success = faux_assistant_message("recovered summary")
        success.usage = create_usage(10)
        call_count = use_scripted_stream_fn(
            harness, [error_message("terminated"), error_message("terminated"), success]
        )

        result = await harness.session.compact()

        assert "recovered summary" in result.summary
        assert call_count[0] == 3  # 1 initial + 2 retries
        starts = harness.events_of_type("summarization_retry_scheduled")
        ends = harness.events_of_type("summarization_retry_finished")
        assert len(starts) == 2
        assert len(ends) == 1
        assert starts[0].attempt == 1
        assert starts[0].max_attempts == 3
        assert starts[0].error_message == "terminated"
        assert starts[1].attempt == 2
        assert starts[1].max_attempts == 3
        assert ends[0].type == "summarization_retry_finished"
    finally:
        harness.cleanup()


async def test_does_not_retry_non_retryable_error(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        harness.settings_manager.apply_overrides({"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 0}})

        call_count = use_scripted_stream_fn(harness, [error_message("insufficient_quota")])

        with pytest.raises(Exception, match="insufficient_quota"):
            await harness.session.compact()
        assert call_count[0] == 1
        assert harness.events_of_type("summarization_retry_scheduled") == []
    finally:
        harness.cleanup()


async def test_does_not_retry_when_retry_disabled(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        harness.settings_manager.apply_overrides({"retry": {"enabled": False, "maxRetries": 3, "baseDelayMs": 0}})

        call_count = use_scripted_stream_fn(harness, [error_message("terminated")])

        with pytest.raises(Exception, match="terminated"):
            await harness.session.compact()
        assert call_count[0] == 1
        assert harness.events_of_type("summarization_retry_scheduled") == []
    finally:
        harness.cleanup()


async def test_stops_retrying_after_max_retries(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        harness.settings_manager.apply_overrides({"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 0}})

        errors = [error_message("terminated") for _ in range(3)]
        call_count = use_scripted_stream_fn(harness, errors)

        with pytest.raises(Exception, match="terminated"):
            await harness.session.compact()
        assert call_count[0] == 3  # 1 initial + 2 retries
        starts = harness.events_of_type("summarization_retry_scheduled")
        ends = harness.events_of_type("summarization_retry_finished")
        assert len(starts) == 2
        assert len(ends) == 1
        assert ends[0].type == "summarization_retry_finished"
    finally:
        harness.cleanup()


async def test_aborts_in_flight_retry_backoff(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        harness.settings_manager.apply_overrides({"retry": {"enabled": True, "maxRetries": 5, "baseDelayMs": 30_000}})

        use_scripted_stream_fn(harness, [error_message("terminated") for _ in range(3)])

        # TS lets the first error resolve with a bare `setTimeout(0)` before
        # aborting. A fixed number of event-loop turns is not the same thing --
        # under parallel load the backoff may not have been scheduled yet, and
        # the abort then lands somewhere else entirely. Wait for the retry to
        # actually be scheduled instead: `retry_assistant_message` emits
        # `on_retry_scheduled` immediately before `_sleep`, and `_sleep` raises
        # `RetrySleepAbortError` for a signal that is already aborted, so an
        # abort at this point always lands in the backoff.
        retry_scheduled = asyncio.Event()

        def watch(event: object) -> None:
            if getattr(event, "type", None) == "summarization_retry_scheduled":
                retry_scheduled.set()

        unsubscribe = harness.session.subscribe(watch)
        try:
            compact_task = asyncio.ensure_future(harness.session.compact())
            await asyncio.wait_for(retry_scheduled.wait(), timeout=10)
        finally:
            unsubscribe()
        harness.session.abort_compaction()

        # TS asserts only `rejects.toThrow()` here: any rejection counts.
        raised: Exception | None = None
        try:
            await compact_task
        except Exception as error:
            raised = error
        assert raised is not None

        compaction_end = harness.events_of_type("compaction_end")[-1]
        assert compaction_end.aborted is True
    finally:
        harness.cleanup()
