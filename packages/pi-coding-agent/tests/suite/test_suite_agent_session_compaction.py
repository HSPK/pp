"""Python port of `packages/coding-agent/test/suite/agent-session-compaction.test.ts`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from harness import Harness, create_harness, get_user_texts
from pi_ai.auth.types import ApiKeyAuth, AuthResult, ProviderAuth, ResolvedAuth
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.registry import Provider, create_provider
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
from pi_coding_agent.core.compaction import CompactionResult, estimate_tokens
from pi_coding_agent.core.extensions.types import SessionBeforeCompactResult
from pi_coding_agent.core.session_manager import CompactionEntry


def create_usage(total_tokens: int) -> Usage:
    return Usage(
        input=total_tokens,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=total_tokens,
        cost=Cost(),
    )


def create_assistant(
    harness: Harness,
    *,
    stop_reason: str = "stop",
    error_message: str | None = None,
    total_tokens: int = 0,
    timestamp: int | None = None,
) -> AssistantMessage:
    model = harness.get_model()
    assert model is not None
    message = faux_assistant_message("", stop_reason=stop_reason, error_message=error_message, timestamp=timestamp)
    message.api = model.api
    message.provider = model.provider
    message.model = model.id
    message.usage = create_usage(total_tokens)
    return message


def use_summary_stream_fn(harness: Harness, summary: str) -> list[int]:
    """Port of `useSummaryStreamFn`: swap the agent's stream function for one
    that always answers with `summary`. The returned single-element list holds
    the call count (TS returns a closure over a counter)."""
    call_count = [0]

    def stream_fn(model, context, options=None, **kwargs):
        call_count[0] += 1
        stream = AssistantMessageEventStream()
        message = faux_assistant_message(summary)
        message.api = model.api
        message.provider = model.provider
        message.model = model.id
        message.usage = create_usage(10)
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end(message)
        return stream

    harness.session.agent.stream_function = stream_fn
    return call_count


def seed_compactable_session(harness: Harness) -> None:
    harness.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    now = now_ms()
    harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
    )
    assistant = create_assistant(harness, stop_reason="stop", total_tokens=100, timestamp=now - 500)
    assistant.content = [TextContent(text="assistant response to compact")]
    harness.session_manager.append_message(assistant)
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages


def extension_summary_factory(summary: str, **extra: Any):
    """An extension that answers `session_before_compact` with a fixed summary."""

    def factory(pi) -> None:
        async def on_before_compact(event, ctx) -> SessionBeforeCompactResult:
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary=summary,
                    first_kept_entry_id=event.preparation.first_kept_entry_id,
                    tokens_before=event.preparation.tokens_before,
                    details={},
                    **extra,
                )
            )

        pi.on("session_before_compact", on_before_compact)

    return factory


async def test_manually_compacts_using_an_extension_provided_summary(tmp_path: Path) -> None:
    summary_usage = Usage(
        input=10,
        output=20,
        cache_read=30,
        cache_write=40,
        total_tokens=100,
        cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )

    def factory(pi) -> None:
        async def on_before_compact(event, ctx) -> SessionBeforeCompactResult:
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary="summary from extension",
                    first_kept_entry_id=event.preparation.first_kept_entry_id,
                    tokens_before=event.preparation.tokens_before,
                    usage=summary_usage,
                    details={"source": "extension"},
                )
            )

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[factory],
    )
    try:
        harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
        await asyncio.wait_for(harness.session.prompt("one"), timeout=5)
        await asyncio.wait_for(harness.session.prompt("two"), timeout=5)
        stats_before = harness.session.get_session_stats()

        result = await asyncio.wait_for(harness.session.compact(), timeout=5)
        compaction_entries = [entry for entry in harness.session_manager.get_entries() if entry.type == "compaction"]
        estimated_tokens_after = sum(estimate_tokens(message) for message in harness.session.messages)

        assert result.summary == "summary from extension"
        assert result.usage == summary_usage
        assert result.estimated_tokens_after == estimated_tokens_after
        assert len(compaction_entries) == 1
        assert compaction_entries[0].usage == summary_usage

        stats_after = harness.session.get_session_stats()
        assert stats_after.tokens.input == stats_before.tokens.input + summary_usage.input
        assert stats_after.tokens.output == stats_before.tokens.output + summary_usage.output
        assert stats_after.tokens.cache_read == stats_before.tokens.cache_read + summary_usage.cache_read
        assert stats_after.tokens.cache_write == stats_before.tokens.cache_write + summary_usage.cache_write
        assert stats_after.cost == stats_before.cost + summary_usage.cost.total
        assert harness.session.messages[0].role == "compactionSummary"
    finally:
        harness.cleanup()


async def test_allows_a_queued_prompt_to_start_when_manual_compaction_ends(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[extension_summary_factory("manual compacted")],
    )
    try:
        seed_compactable_session(harness)
        harness.set_responses([faux_assistant_message("queued response")])

        queued: list[asyncio.Future[None]] = []
        compacting_flags: list[bool] = []

        def on_event(event) -> None:
            if event.type == "compaction_end" and event.reason == "manual" and event.result is not None:
                compacting_flags.append(harness.session.is_compacting)
                queued.append(asyncio.ensure_future(harness.session.prompt("queued after compaction")))

        harness.session.subscribe(on_event)

        await asyncio.wait_for(harness.session.compact(), timeout=5)
        assert queued, "compaction_end did not start the queued prompt"
        await asyncio.wait_for(queued[0], timeout=5)

        assert compacting_flags == [False]
        assert "queued after compaction" in get_user_texts(harness)
        assert harness.session.get_last_assistant_text() == "queued response"
    finally:
        harness.cleanup()


async def test_throws_when_compacting_without_a_model(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.session.agent.state.model = None
        with pytest.raises(RuntimeError, match="No model selected"):
            await asyncio.wait_for(harness.session.compact(), timeout=5)
    finally:
        harness.cleanup()


async def test_throws_when_compacting_without_configured_auth(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        model = harness.get_model()
        assert model is not None
        with pytest.raises(RuntimeError, match=f"No API key found for {model.provider}\\."):
            await asyncio.wait_for(harness.session.compact(), timeout=5)
    finally:
        harness.cleanup()


async def test_manually_compacts_with_a_custom_stream_fn_when_registry_auth_is_absent(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        call_count = use_summary_stream_fn(harness, "summary from custom stream")

        result = await asyncio.wait_for(harness.session.compact(), timeout=5)

        assert "summary from custom stream" in result.summary
        assert call_count[0] == 1
    finally:
        harness.cleanup()


async def test_manually_compacts_with_provider_resolved_bearer_auth(tmp_path: Path) -> None:
    seen: list[Any] = []

    class BearerResolve:
        def __call__(self, *, credential: Any = None, env: Any = None) -> AuthResult:
            return AuthResult(
                auth=ResolvedAuth(headers={"Authorization": "******"}),
                source="ambient bearer token",
            )

    def to_bearer_provider(provider: Provider) -> Provider:
        # TS calls `modelRuntime.registerNativeProvider` with a provider whose
        # auth resolves to headers only; this port re-registers the faux
        # provider with that auth instead.
        return create_provider(
            id=provider.id,
            name="Faux bearer provider",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Faux bearer token", resolve=BearerResolve())),
            api=provider.api,
            models=provider.models,
        )

    harness = await create_harness(tmp_path, with_configured_auth=False, provider_override=to_bearer_provider)
    try:
        seed_compactable_session(harness)

        def respond(context, options, state, model) -> AssistantMessage:
            seen.append(options)
            return faux_assistant_message("summary with bearer auth")

        harness.set_responses([respond])

        result = await asyncio.wait_for(harness.session.compact(), timeout=5)

        assert "summary with bearer auth" in result.summary
        assert harness.faux.state.call_count == 1
        assert seen[0].api_key is None
        assert seen[0].headers == {"Authorization": "******"}
    finally:
        harness.cleanup()


async def test_persists_usage_from_pi_generated_manual_compaction(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        use_summary_stream_fn(harness, "summary from custom stream")

        result = await asyncio.wait_for(harness.session.compact(), timeout=5)

        compaction_entries = [
            entry for entry in harness.session_manager.get_entries() if isinstance(entry, CompactionEntry)
        ]
        assert result.usage == create_usage(10)
        assert len(compaction_entries) == 1
        assert compaction_entries[0].usage == create_usage(10)
    finally:
        harness.cleanup()


async def test_auto_compacts_with_a_custom_stream_fn_when_registry_auth_is_absent(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        seed_compactable_session(harness)
        call_count = use_summary_stream_fn(harness, "auto summary from custom stream")

        await asyncio.wait_for(harness.session._run_auto_compaction("threshold", False), timeout=5)

        compaction_entries = [
            entry for entry in harness.session_manager.get_entries() if isinstance(entry, CompactionEntry)
        ]
        compaction_end = harness.events_of_type("compaction_end")[-1]
        assert len(compaction_entries) == 1
        assert compaction_end.result is not None
        assert compaction_end.result.estimated_tokens_after > 0
        assert call_count[0] == 1
    finally:
        harness.cleanup()


async def test_compacts_and_resumes_after_a_length_stop_below_the_desired_output_limit(tmp_path: Path) -> None:
    from pi_ai.providers.faux import FauxModelDefinition

    harness = await create_harness(
        tmp_path,
        models=[FauxModelDefinition(id="faux-1", context_window=1000, max_tokens=100)],
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[extension_summary_factory("overflow compacted")],
    )
    try:
        harness.set_responses(
            [
                faux_assistant_message("partial response", stop_reason="length"),
                faux_assistant_message("completed response"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("x" * 5000), timeout=10)

        assert harness.faux.state.call_count == 2
        compaction_end = harness.events_of_type("compaction_end")[-1]
        assert compaction_end.reason == "overflow"
        assert compaction_end.aborted is False
        assert compaction_end.will_retry is True
        assert harness.session.get_last_assistant_text() == "completed response"
    finally:
        harness.cleanup()


async def test_does_not_compact_when_a_length_stop_reaches_the_desired_output_limit(tmp_path: Path) -> None:
    from pi_ai.providers.faux import FauxModelDefinition

    harness = await create_harness(
        tmp_path,
        models=[FauxModelDefinition(id="faux-1", context_window=1_000_000, max_tokens=100)],
    )
    try:
        harness.set_responses([faux_assistant_message("x" * 400, stop_reason="length")])

        await asyncio.wait_for(harness.session.prompt("hello"), timeout=10)

        assert harness.faux.state.call_count == 1
        assert harness.events_of_type("compaction_start") == []
    finally:
        harness.cleanup()


async def test_stops_after_one_compact_and_retry_when_a_second_response_is_also_truncated(tmp_path: Path) -> None:
    from pi_ai.providers.faux import FauxModelDefinition

    harness = await create_harness(
        tmp_path,
        models=[FauxModelDefinition(id="faux-1", context_window=1_000_000, max_tokens=100)],
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[extension_summary_factory("overflow compacted")],
    )
    try:
        harness.set_responses(
            [
                lambda *_: faux_assistant_message("x" * 64, stop_reason="length", timestamp=now_ms() + 10_000),
                lambda *_: faux_assistant_message("y" * 64, stop_reason="length", timestamp=now_ms() + 10_000),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("x" * 5000), timeout=10)

        assert harness.faux.state.call_count == 2
        overflow_starts = [event for event in harness.events_of_type("compaction_start") if event.reason == "overflow"]
        assert len(overflow_starts) == 1
        assert harness.events_of_type("compaction_end")[-1].error_message == (
            "Context overflow recovery failed after one compact-and-retry attempt. "
            "Try reducing context or switching to a larger-context model."
        )
    finally:
        harness.cleanup()


async def test_cancels_in_progress_manual_compaction_when_abort_compaction_is_called(tmp_path: Path) -> None:
    def factory(pi) -> None:
        async def on_before_compact(event, ctx) -> SessionBeforeCompactResult:
            await event.signal.wait()
            return SessionBeforeCompactResult(cancel=True)

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[factory],
    )
    try:
        harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
        await asyncio.wait_for(harness.session.prompt("one"), timeout=5)
        await asyncio.wait_for(harness.session.prompt("two"), timeout=5)

        compact_task = asyncio.ensure_future(harness.session.compact())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        harness.session.abort_compaction()

        with pytest.raises(RuntimeError, match="Compaction cancelled"):
            await asyncio.wait_for(compact_task, timeout=5)
    finally:
        harness.cleanup()


async def test_resumes_after_threshold_compaction_with_only_agent_level_queued_messages(tmp_path: Path) -> None:
    from pi_agent.harness.messages import CustomMessage

    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[extension_summary_factory("auto compacted")],
    )
    try:
        harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
        await asyncio.wait_for(harness.session.prompt("first"), timeout=5)
        await asyncio.wait_for(harness.session.prompt("second"), timeout=5)

        harness.session.agent.follow_up(
            CustomMessage(
                custom_type="test",
                content=[TextContent(text="queued custom")],
                display=False,
                timestamp=now_ms(),
            )
        )

        assert await asyncio.wait_for(harness.session._run_auto_compaction("threshold", False), timeout=5) is True
    finally:
        harness.cleanup()


async def test_does_not_retry_overflow_recovery_more_than_once(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        overflow_message = create_assistant(
            harness, stop_reason="error", error_message="prompt is too long", timestamp=now_ms()
        )
        calls: list[tuple[str, bool]] = []

        async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
            calls.append((reason, will_retry))
            return False

        harness.session._run_auto_compaction = fake_run_auto_compaction

        compaction_errors: list[str] = []

        def on_event(event) -> None:
            if event.type == "compaction_end" and event.error_message:
                compaction_errors.append(event.error_message)

        harness.session.subscribe(on_event)

        await asyncio.wait_for(harness.session._check_compaction(overflow_message), timeout=5)
        second = create_assistant(
            harness, stop_reason="error", error_message="prompt is too long", timestamp=now_ms() + 1
        )
        await asyncio.wait_for(harness.session._check_compaction(second), timeout=5)

        assert len(calls) == 1
        assert (
            "Context overflow recovery failed after one compact-and-retry attempt. "
            "Try reducing context or switching to a larger-context model." in compaction_errors
        )
    finally:
        harness.cleanup()


async def test_compacts_successful_overflow_responses_without_retrying(tmp_path: Path) -> None:
    from pi_ai.providers.faux import FauxModelDefinition

    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        models=[FauxModelDefinition(id="faux-1", context_window=1, max_tokens=100)],
        extension_factories=[extension_summary_factory("successful overflow compacted")],
    )
    try:
        harness.set_responses([faux_assistant_message("completed answer")])

        assert await asyncio.wait_for(harness.session.prompt("hello"), timeout=10) is None

        compaction_end = harness.events_of_type("compaction_end")[-1]
        assert compaction_end.reason == "overflow"
        assert compaction_end.aborted is False
        assert compaction_end.will_retry is False
        assert harness.faux.state.call_count == 1
    finally:
        harness.cleanup()


async def test_ignores_stale_pre_compaction_assistant_usage_on_pre_prompt_checks(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        stale_timestamp = now_ms() - 10_000
        stale_assistant = create_assistant(harness, stop_reason="stop", total_tokens=610_000, timestamp=stale_timestamp)

        harness.session_manager.append_message(
            UserMessage(content=[TextContent(text="before compaction")], timestamp=stale_timestamp - 1000)
        )
        harness.session_manager.append_message(stale_assistant)
        first_kept_entry_id = harness.session_manager.get_entries()[0].id
        harness.session_manager.append_compaction(
            "summary", first_kept_entry_id, stale_assistant.usage.total_tokens, None, False
        )
        harness.session_manager.append_message(
            UserMessage(content=[TextContent(text="after compaction")], timestamp=now_ms())
        )

        calls: list[tuple[str, bool]] = []

        async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
            calls.append((reason, will_retry))
            return False

        harness.session._run_auto_compaction = fake_run_auto_compaction

        await asyncio.wait_for(harness.session._check_compaction(stale_assistant, False), timeout=5)

        assert calls == []
    finally:
        harness.cleanup()


async def test_triggers_threshold_compaction_for_errors_using_last_successful_usage(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        successful_assistant = create_assistant(harness, stop_reason="stop", total_tokens=190_000, timestamp=now_ms())
        error_assistant = create_assistant(
            harness, stop_reason="error", error_message="529 overloaded", timestamp=now_ms() + 1000
        )
        harness.session.agent.state.messages = [
            UserMessage(content=[TextContent(text="hello")], timestamp=now_ms() - 1000),
            successful_assistant,
            UserMessage(content=[TextContent(text="retry")], timestamp=now_ms() + 500),
            error_assistant,
        ]

        calls: list[tuple[str, bool]] = []

        async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
            calls.append((reason, will_retry))
            return False

        harness.session._run_auto_compaction = fake_run_auto_compaction

        await asyncio.wait_for(harness.session._check_compaction(error_assistant), timeout=5)

        assert calls == [("threshold", False)]
    finally:
        harness.cleanup()


async def test_does_not_trigger_threshold_compaction_for_errors_without_prior_usage(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        error_assistant = create_assistant(
            harness, stop_reason="error", error_message="529 overloaded", timestamp=now_ms()
        )
        harness.session.agent.state.messages = [
            UserMessage(content=[TextContent(text="hello")], timestamp=now_ms() - 1000),
            error_assistant,
        ]

        calls: list[tuple[str, bool]] = []

        async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
            calls.append((reason, will_retry))
            return False

        harness.session._run_auto_compaction = fake_run_auto_compaction

        await asyncio.wait_for(harness.session._check_compaction(error_assistant), timeout=5)

        assert calls == []
    finally:
        harness.cleanup()


async def test_does_not_trigger_threshold_compaction_with_only_kept_pre_compaction_usage(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        pre_compaction_timestamp = now_ms() - 10_000
        kept_assistant = create_assistant(
            harness, stop_reason="stop", total_tokens=190_000, timestamp=pre_compaction_timestamp
        )

        harness.session_manager.append_message(
            UserMessage(content=[TextContent(text="before compaction")], timestamp=pre_compaction_timestamp - 1000)
        )
        harness.session_manager.append_message(kept_assistant)
        first_kept_entry_id = harness.session_manager.get_entries()[0].id
        harness.session_manager.append_compaction(
            "summary", first_kept_entry_id, kept_assistant.usage.total_tokens, None, False
        )

        error_assistant = create_assistant(
            harness, stop_reason="error", error_message="529 overloaded", timestamp=now_ms()
        )
        harness.session.agent.state.messages = [
            UserMessage(content=[TextContent(text="kept user")], timestamp=pre_compaction_timestamp - 1000),
            kept_assistant,
            UserMessage(content=[TextContent(text="new prompt")], timestamp=now_ms() - 500),
            error_assistant,
        ]

        calls: list[tuple[str, bool]] = []

        async def fake_run_auto_compaction(reason: str, will_retry: bool) -> bool:
            calls.append((reason, will_retry))
            return False

        harness.session._run_auto_compaction = fake_run_auto_compaction

        await asyncio.wait_for(harness.session._check_compaction(error_assistant), timeout=5)

        assert calls == []
    finally:
        harness.cleanup()


async def test_does_not_trigger_threshold_compaction_below_threshold_or_when_disabled(tmp_path: Path) -> None:
    from pi_ai.providers.faux import FauxModelDefinition

    below_threshold_harness = await create_harness(
        tmp_path / "below",
        settings={"compaction": {"enabled": True, "reserveTokens": 1000}},
        models=[FauxModelDefinition(id="faux-1", context_window=200_000)],
    )
    disabled_harness = await create_harness(tmp_path / "disabled", settings={"compaction": {"enabled": False}})
    try:
        below_calls: list[tuple[str, bool]] = []
        disabled_calls: list[tuple[str, bool]] = []

        async def fake_below(reason: str, will_retry: bool) -> bool:
            below_calls.append((reason, will_retry))
            return False

        async def fake_disabled(reason: str, will_retry: bool) -> bool:
            disabled_calls.append((reason, will_retry))
            return False

        below_threshold_harness.session._run_auto_compaction = fake_below
        disabled_harness.session._run_auto_compaction = fake_disabled

        await asyncio.wait_for(
            below_threshold_harness.session._check_compaction(
                create_assistant(below_threshold_harness, stop_reason="stop", total_tokens=1_000, timestamp=now_ms())
            ),
            timeout=5,
        )
        await asyncio.wait_for(
            disabled_harness.session._check_compaction(
                create_assistant(disabled_harness, stop_reason="stop", total_tokens=1_000_000, timestamp=now_ms())
            ),
            timeout=5,
        )

        assert below_calls == []
        assert disabled_calls == []
    finally:
        below_threshold_harness.cleanup()
        disabled_harness.cleanup()
