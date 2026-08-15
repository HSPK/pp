"""Python port of `packages/coding-agent/test/agent-session-auto-compaction-queue.test.ts`.

Drives `AgentSession._check_compaction` / `_run_auto_compaction` directly, as
the TypeScript does through its `as unknown as {...}` casts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_agent.agent import Agent, MutableAgentState
from pi_agent.harness.messages import CustomMessage
from pi_ai.auth.types import Credential
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    TextContent,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream

from pi_coding_agent.core.agent_session import AgentSession, CompactionEndEvent
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

ZERO_COST = Cost(input=0, output=0, cache_read=0, cache_write=0, total=0)


def usage(input_tokens: int = 0, output_tokens: int = 0, total: int | None = None) -> Usage:
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=0,
        cache_write=0,
        total_tokens=input_tokens + output_tokens if total is None else total,
        cost=ZERO_COST,
    )


def assistant(
    model: Any,
    text: str,
    *,
    stop_reason: str = "stop",
    error_message: str | None = None,
    timestamp: int | None = None,
    message_usage: Usage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[TextContent(text=text)],
        usage=message_usage if message_usage is not None else usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        timestamp=now_ms() if timestamp is None else timestamp,
    )


class SessionFixture:
    def __init__(
        self, session: AgentSession, session_manager: SessionManager, settings_manager: SettingsManager
    ) -> None:
        self.session = session
        self.session_manager = session_manager
        self.settings_manager = settings_manager


@pytest.fixture
async def fixture(tmp_path: Path) -> Any:
    """Port of the TypeScript `beforeEach`/`afterEach` pair."""
    temp_dir = tmp_path / "auto-compaction-queue"
    temp_dir.mkdir(parents=True, exist_ok=True)

    auth_storage = AuthStorage.create(str(temp_dir / "auth.json"))
    await auth_storage.set("anthropic", Credential(type="api_key", key="test-key"))
    model_runtime = await ModelRuntime.create(
        agent_dir=str(temp_dir),
        credentials=auth_storage,
        models_path=str(temp_dir / "models.json"),
    )
    model = model_runtime.get_model("anthropic", "claude-sonnet-4-5")
    assert model is not None

    agent = Agent(
        model_runtime.stream_simple,
        initial_state=MutableAgentState(model=model, system_prompt="Test"),
    )

    session_manager = SessionManager.in_memory()
    settings_manager = SettingsManager.create(str(temp_dir), str(temp_dir))
    resource_loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(temp_dir),
            agent_dir=str(temp_dir),
            no_skills=True,
            no_prompt_templates=True,
            no_context_files=True,
        )
    )
    resource_loader.reload()

    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=settings_manager,
        cwd=str(temp_dir),
        model_runtime=model_runtime,
        resource_loader=resource_loader,
    )
    try:
        yield SessionFixture(session, session_manager, settings_manager)
    finally:
        session.dispose()


async def test_resumes_after_threshold_compaction_with_only_agent_level_queued_messages(fixture) -> None:
    session = fixture.session
    fixture.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    model = session.model
    assert model is not None
    now = now_ms()

    fixture.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
    )
    fixture.session_manager.append_message(
        assistant(
            model,
            "assistant response to compact",
            timestamp=now - 500,
            message_usage=usage(100, 0),
        )
    )
    session.agent.state.messages = fixture.session_manager.build_session_context().messages

    def stream_function(summary_model, context=None, options=None, **kwargs):
        stream = AssistantMessageEventStream()
        message = assistant(summary_model, "compacted", message_usage=usage(10, 0))
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end(message)
        return stream

    session.agent.stream_function = stream_function

    session.agent.follow_up(
        CustomMessage(
            custom_type="test",
            content=[TextContent(text="Queued custom")],
            display=False,
            timestamp=now_ms(),
        )
    )

    assert session.pending_message_count == 0
    assert session.agent.has_queued_messages() is True

    continue_calls: list[None] = []

    async def continue_spy() -> None:
        continue_calls.append(None)

    session.agent.continue_ = continue_spy

    assert await session._run_auto_compaction("threshold", False) is True

    assert continue_calls == []


async def test_does_not_compact_repeatedly_after_overflow_recovery_already_attempted(fixture) -> None:
    session = fixture.session
    model = session.model
    assert model is not None

    overflow_message = assistant(
        model,
        "",
        stop_reason="error",
        error_message="prompt is too long",
    )

    run_calls: list[tuple[str, bool]] = []

    async def run_auto_compaction_spy(reason: str, will_retry: bool) -> bool:
        run_calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = run_auto_compaction_spy

    events: list[dict[str, Any]] = []

    def on_event(event: Any) -> None:
        if isinstance(event, CompactionEndEvent):
            events.append({"type": event.type, "reason": event.reason, "error_message": event.error_message})

    session.subscribe(on_event)

    await session._check_compaction(overflow_message)
    await session._check_compaction(
        assistant(model, "", stop_reason="error", error_message="prompt is too long", timestamp=now_ms() + 1)
    )

    assert len(run_calls) == 1
    assert {
        "type": "compaction_end",
        "reason": "overflow",
        "error_message": (
            "Context overflow recovery failed after one compact-and-retry attempt. "
            "Try reducing context or switching to a larger-context model."
        ),
    } in events


async def test_ignores_stale_pre_compaction_assistant_usage_on_pre_prompt_checks(fixture) -> None:
    session = fixture.session
    model = session.model
    assert model is not None
    stale_timestamp = now_ms() - 10_000

    stale_assistant = assistant(
        model,
        "large response before compaction",
        timestamp=stale_timestamp,
        message_usage=usage(600_000, 10_000),
    )

    fixture.session_manager.append_message(
        UserMessage(content=[TextContent(text="before compaction")], timestamp=stale_timestamp - 1000)
    )
    fixture.session_manager.append_message(stale_assistant)

    first_kept_entry_id = fixture.session_manager.get_entries()[0].id
    fixture.session_manager.append_compaction(
        "summary", first_kept_entry_id, stale_assistant.usage.total_tokens, None, False
    )

    fixture.session_manager.append_message(
        UserMessage(content=[TextContent(text="session recovery payload")], timestamp=now_ms())
    )

    run_calls: list[tuple[str, bool]] = []

    async def run_auto_compaction_spy(reason: str, will_retry: bool) -> bool:
        run_calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = run_auto_compaction_spy

    await session._check_compaction(stale_assistant, False)

    assert run_calls == []


async def test_triggers_threshold_compaction_for_error_messages_using_last_successful_usage(fixture) -> None:
    session = fixture.session
    model = session.model
    assert model is not None

    compaction_settings = fixture.settings_manager.get_compaction_settings()
    threshold_tokens = (model.context_window or 200_000) - compaction_settings["reserveTokens"] + 1
    successful_assistant = assistant(
        model,
        "large successful response",
        message_usage=usage(threshold_tokens - 10_000, 10_000, total=threshold_tokens),
    )

    error_assistant = assistant(
        model,
        "",
        stop_reason="error",
        error_message="529 overloaded",
        timestamp=now_ms() + 1000,
    )

    session.agent.state.messages = [
        UserMessage(content=[TextContent(text="hello")], timestamp=now_ms() - 1000),
        successful_assistant,
        UserMessage(content=[TextContent(text="another prompt")], timestamp=now_ms() + 500),
        error_assistant,
    ]

    run_calls: list[tuple[str, bool]] = []

    async def run_auto_compaction_spy(reason: str, will_retry: bool) -> bool:
        run_calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = run_auto_compaction_spy

    await session._check_compaction(error_assistant)

    assert run_calls == [("threshold", False)]


async def test_does_not_trigger_threshold_compaction_for_errors_without_prior_usage(fixture) -> None:
    session = fixture.session
    model = session.model
    assert model is not None

    error_assistant = assistant(model, "", stop_reason="error", error_message="529 overloaded")

    session.agent.state.messages = [
        UserMessage(content=[TextContent(text="hello")], timestamp=now_ms() - 1000),
        error_assistant,
    ]

    run_calls: list[tuple[str, bool]] = []

    async def run_auto_compaction_spy(reason: str, will_retry: bool) -> bool:
        run_calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = run_auto_compaction_spy

    await session._check_compaction(error_assistant)

    assert run_calls == []


async def test_does_not_trigger_threshold_compaction_when_only_kept_pre_compaction_usage_exists(fixture) -> None:
    session = fixture.session
    model = session.model
    assert model is not None
    pre_compaction_timestamp = now_ms() - 10_000

    kept_assistant = assistant(
        model,
        "kept response from before compaction",
        timestamp=pre_compaction_timestamp,
        message_usage=usage(180_000, 10_000),
    )

    fixture.session_manager.append_message(
        UserMessage(content=[TextContent(text="before compaction")], timestamp=pre_compaction_timestamp - 1000)
    )
    fixture.session_manager.append_message(kept_assistant)
    first_kept_entry_id = fixture.session_manager.get_entries()[0].id
    fixture.session_manager.append_compaction(
        "summary", first_kept_entry_id, kept_assistant.usage.total_tokens, None, False
    )

    error_assistant = assistant(model, "", stop_reason="error", error_message="529 overloaded")

    session.agent.state.messages = [
        UserMessage(content=[TextContent(text="kept user msg")], timestamp=pre_compaction_timestamp - 1000),
        kept_assistant,
        UserMessage(content=[TextContent(text="new prompt")], timestamp=now_ms() - 500),
        error_assistant,
    ]

    run_calls: list[tuple[str, bool]] = []

    async def run_auto_compaction_spy(reason: str, will_retry: bool) -> bool:
        run_calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = run_auto_compaction_spy

    await session._check_compaction(error_assistant)

    # Should NOT compact because the only usage data is from a kept
    # pre-compaction message.
    assert run_calls == []
