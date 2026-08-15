"""Tests for `pi_coding_agent.core.compaction`, ported from
`packages/coding-agent/test/compaction.test.ts`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from pi_ai.types import AssistantMessage, Cost, Model, TextContent, Usage, UserMessage

from pi_coding_agent.core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    GenerateBranchSummaryOptions,
    calculate_context_tokens,
    collect_entries_for_branch_summary,
    compact,
    estimate_context_tokens,
    find_cut_point,
    generate_branch_summary,
    get_last_assistant_usage,
    prepare_branch_entries,
    prepare_compaction,
    should_compact,
)
from pi_coding_agent.core.session_manager import (
    CompactionEntry,
    CustomMessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionManager,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
    _entry_from_raw,
    build_session_context,
    migrate_session_entries,
    parse_session_entries,
)

_LARGE_SESSION_FIXTURE = Path(__file__).parent / "fixtures" / "large-session.jsonl"


def _load_large_session_entries() -> list[SessionEntry]:
    """Port of `loadLargeSessionEntries`.

    TypeScript's `parseSessionEntries` returns the parsed JSON directly as
    `SessionEntry[]` (types are erased at runtime), so the port has one extra step:
    raw dicts must be converted to typed entries with `_entry_from_raw` after the
    v1 migration has stamped `id`/`parentId` on them.
    """
    raw = parse_session_entries(_LARGE_SESSION_FIXTURE.read_text(encoding="utf-8"))
    migrate_session_entries(raw)
    entries: list[SessionEntry] = []
    for item in raw:
        if item.get("type") == "session":
            continue
        entry = _entry_from_raw(item)
        if entry is not None:
            entries.append(entry)
    return entries


_entry_counter = 0
_last_id: str | None = None


def _reset_entry_counter() -> None:
    global _entry_counter, _last_id
    _entry_counter = 0
    _last_id = None


def _next_id() -> str:
    global _entry_counter
    entry_id = f"test-id-{_entry_counter}"
    _entry_counter += 1
    return entry_id


def _mock_usage(input_tokens: int, output: int, cache_read: int = 0, cache_write: int = 0) -> Usage:
    return Usage(
        input=input_tokens,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output + cache_read + cache_write,
        cost=Cost(),
    )


def _user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def _assistant_message(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(text=text)],
        usage=usage or _mock_usage(100, 50),
        stop_reason="stop",
        timestamp=1,
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )


def _message_entry(message) -> SessionMessageEntry:
    global _last_id
    entry_id = _next_id()
    entry = SessionMessageEntry(id=entry_id, parent_id=_last_id, timestamp="2025-01-01T00:00:00Z", message=message)
    _last_id = entry_id
    return entry


def _compaction_entry(summary: str, first_kept_entry_id: str) -> CompactionEntry:
    global _last_id
    entry_id = _next_id()
    entry = CompactionEntry(
        id=entry_id,
        parent_id=_last_id,
        timestamp="2025-01-01T00:00:00Z",
        summary=summary,
        first_kept_entry_id=first_kept_entry_id,
        tokens_before=10000,
    )
    _last_id = entry_id
    return entry


def _model_change_entry(provider: str, model_id: str) -> ModelChangeEntry:
    global _last_id
    entry_id = _next_id()
    entry = ModelChangeEntry(
        id=entry_id, parent_id=_last_id, timestamp="2025-01-01T00:00:00Z", provider=provider, model_id=model_id
    )
    _last_id = entry_id
    return entry


def _thinking_level_entry(thinking_level: str) -> ThinkingLevelChangeEntry:
    global _last_id
    entry_id = _next_id()
    entry = ThinkingLevelChangeEntry(
        id=entry_id, parent_id=_last_id, timestamp="2025-01-01T00:00:00Z", thinking_level=thinking_level
    )
    _last_id = entry_id
    return entry


def _custom_message_entry(content: str) -> CustomMessageEntry:
    global _last_id
    entry_id = _next_id()
    entry = CustomMessageEntry(
        id=entry_id,
        parent_id=_last_id,
        timestamp="2025-01-01T00:00:00Z",
        custom_type="test",
        content=content,
        display=True,
    )
    _last_id = entry_id
    return entry


def _extract_text(messages) -> str:
    parts = []
    for message in messages:
        role = message.role
        if role == "user":
            content = message.content
            parts.append(content if isinstance(content, str) else " ".join(b.text for b in content if b.type == "text"))
        elif role == "assistant":
            parts.append(" ".join(b.text for b in message.content if b.type == "text"))
        elif role in ("branchSummary", "compactionSummary"):
            parts.append(message.summary)
        elif role in ("custom", "toolResult"):
            content = message.content
            parts.append(content if isinstance(content, str) else " ".join(b.text for b in content if b.type == "text"))
        elif role == "bashExecution":
            parts.append(f"{message.command}\n{message.output}")
        else:
            parts.append("")
    return "\n".join(parts)


class TestTokenCalculation:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_calculates_total_context_tokens_from_usage(self) -> None:
        usage = _mock_usage(1000, 500, 200, 100)
        assert calculate_context_tokens(usage) == 1800

    def test_handles_zero_values(self) -> None:
        assert calculate_context_tokens(_mock_usage(0, 0, 0, 0)) == 0


class TestGetLastAssistantUsage:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_finds_last_non_aborted_assistant_usage(self) -> None:
        entries = [
            _message_entry(_user_message("Hello")),
            _message_entry(_assistant_message("Hi", _mock_usage(100, 50))),
            _message_entry(_user_message("How are you?")),
            _message_entry(_assistant_message("Good", _mock_usage(200, 100))),
        ]
        usage = get_last_assistant_usage(entries)
        assert usage is not None
        assert usage.input == 200

    def test_skips_aborted_messages(self) -> None:
        aborted_msg = replace(_assistant_message("Aborted", _mock_usage(300, 150)), stop_reason="aborted")
        entries = [
            _message_entry(_user_message("Hello")),
            _message_entry(_assistant_message("Hi", _mock_usage(100, 50))),
            _message_entry(_user_message("How are you?")),
            _message_entry(aborted_msg),
        ]
        usage = get_last_assistant_usage(entries)
        assert usage is not None
        assert usage.input == 100

    def test_skips_all_zero_assistant_usage(self) -> None:
        entries = [
            _message_entry(_user_message("Hello")),
            _message_entry(_assistant_message("Hi", _mock_usage(100, 50))),
            _message_entry(_user_message("continue")),
            _message_entry(_assistant_message("Partial", _mock_usage(0, 0))),
        ]
        usage = get_last_assistant_usage(entries)
        assert usage is not None
        assert usage.input == 100

    def test_returns_none_if_no_assistant_messages(self) -> None:
        entries = [_message_entry(_user_message("Hello"))]
        assert get_last_assistant_usage(entries) is None


class TestEstimateContextTokens:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_uses_last_nonzero_assistant_usage_as_anchor(self) -> None:
        messages = [
            _user_message("Hello"),
            _assistant_message("Hi", _mock_usage(100, 50)),
            _user_message("continue"),
            _assistant_message("Partial thinking", _mock_usage(0, 0)),
        ]
        estimate = estimate_context_tokens(messages)
        assert estimate.usage_tokens == 150
        assert estimate.last_usage_index == 1
        assert estimate.trailing_tokens > 0
        assert estimate.tokens == 150 + estimate.trailing_tokens


class TestShouldCompact:
    def test_true_when_context_exceeds_threshold(self) -> None:
        settings = CompactionSettings(enabled=True, reserve_tokens=10000, keep_recent_tokens=20000)
        assert should_compact(95000, 100000, settings) is True
        assert should_compact(89000, 100000, settings) is False

    def test_false_when_disabled(self) -> None:
        settings = CompactionSettings(enabled=False, reserve_tokens=10000, keep_recent_tokens=20000)
        assert should_compact(95000, 100000, settings) is False


class TestFindCutPoint:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_finds_cut_point_based_on_token_differences(self) -> None:
        entries = []
        for i in range(10):
            entries.append(_message_entry(_user_message(f"User {i}")))
            entries.append(_message_entry(_assistant_message(f"Assistant {i}", _mock_usage(0, 100, (i + 1) * 1000, 0))))

        result = find_cut_point(entries, 0, len(entries), 2500)
        cut_entry = entries[result.first_kept_entry_index]
        assert cut_entry.type == "message"
        assert cut_entry.message.role in ("user", "assistant")

    def test_returns_start_index_if_no_valid_cut_points(self) -> None:
        entries = [_message_entry(_assistant_message("a"))]
        result = find_cut_point(entries, 0, len(entries), 1000)
        assert result.first_kept_entry_index == 0

    def test_keeps_everything_if_all_fit_within_budget(self) -> None:
        entries = [
            _message_entry(_user_message("1")),
            _message_entry(_assistant_message("a", _mock_usage(0, 50, 500, 0))),
            _message_entry(_user_message("2")),
            _message_entry(_assistant_message("b", _mock_usage(0, 50, 1000, 0))),
        ]
        result = find_cut_point(entries, 0, len(entries), 50000)
        assert result.first_kept_entry_index == 0

    def test_indicates_split_turn_when_cutting_at_assistant_message(self) -> None:
        entries = [
            _message_entry(_user_message("Turn 1")),
            _message_entry(_assistant_message("A1", _mock_usage(0, 100, 1000, 0))),
            _message_entry(_user_message("Turn 2")),  # index 2
            _message_entry(_assistant_message("A2-1", _mock_usage(0, 100, 5000, 0))),  # index 3
            _message_entry(_assistant_message("A2-2", _mock_usage(0, 100, 8000, 0))),  # index 4
            _message_entry(_assistant_message("A2-3", _mock_usage(0, 100, 10000, 0))),  # index 5
        ]
        result = find_cut_point(entries, 0, len(entries), 3000)
        cut_entry = entries[result.first_kept_entry_index]
        if cut_entry.message.role == "assistant":
            assert result.is_split_turn is True
            assert result.turn_start_index == 2

    def test_budgets_context_visible_custom_message_entries(self) -> None:
        entries = [
            _message_entry(_user_message("hi")),
            _message_entry(_assistant_message("hello")),
            _custom_message_entry("x" * 4000),
            _message_entry(_assistant_message("ok")),
        ]

        tiny_budget = find_cut_point(entries, 0, len(entries), 1)
        assert tiny_budget.first_kept_entry_index == 3
        assert tiny_budget.is_split_turn is True
        assert tiny_budget.turn_start_index == 2

        custom_fits_budget = find_cut_point(entries, 0, len(entries), 2)
        assert custom_fits_budget.first_kept_entry_index == 2
        assert custom_fits_budget.is_split_turn is False
        assert custom_fits_budget.turn_start_index == -1


class TestBuildSessionContextForCompaction:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_loads_all_messages_when_no_compaction(self) -> None:
        entries = [
            _message_entry(_user_message("1")),
            _message_entry(_assistant_message("a")),
            _message_entry(_user_message("2")),
            _message_entry(_assistant_message("b")),
        ]
        loaded = build_session_context(entries)
        assert len(loaded.messages) == 4
        assert loaded.thinking_level == "off"
        assert loaded.model.provider == "anthropic"
        assert loaded.model.model_id == "claude-sonnet-4-5"

    def test_handles_single_compaction(self) -> None:
        u1 = _message_entry(_user_message("1"))
        a1 = _message_entry(_assistant_message("a"))
        u2 = _message_entry(_user_message("2"))
        a2 = _message_entry(_assistant_message("b"))
        compaction = _compaction_entry("Summary of 1,a,2,b", u2.id)  # keep from u2 onwards
        u3 = _message_entry(_user_message("3"))
        a3 = _message_entry(_assistant_message("c"))

        entries = [u1, a1, u2, a2, compaction, u3, a3]

        loaded = build_session_context(entries)
        # summary + kept (u2, a2) + after (u3, a3) = 5
        assert len(loaded.messages) == 5
        assert loaded.messages[0].role == "compactionSummary"
        assert "Summary of 1,a,2,b" in loaded.messages[0].summary

    def test_handles_multiple_compactions_only_latest_matters(self) -> None:
        u1 = _message_entry(_user_message("1"))
        a1 = _message_entry(_assistant_message("a"))
        compact1 = _compaction_entry("First summary", u1.id)
        u2 = _message_entry(_user_message("2"))
        b = _message_entry(_assistant_message("b"))
        u3 = _message_entry(_user_message("3"))
        c = _message_entry(_assistant_message("c"))
        compact2 = _compaction_entry("Second summary", u3.id)  # keep from u3 onwards
        u4 = _message_entry(_user_message("4"))
        d = _message_entry(_assistant_message("d"))

        entries = [u1, a1, compact1, u2, b, u3, c, compact2, u4, d]

        loaded = build_session_context(entries)
        # summary + kept from u3 (u3, c) + after (u4, d) = 5
        assert len(loaded.messages) == 5
        assert "Second summary" in loaded.messages[0].summary

    def test_keeps_all_messages_when_first_kept_entry_id_is_first_entry(self) -> None:
        u1 = _message_entry(_user_message("1"))
        a1 = _message_entry(_assistant_message("a"))
        compact1 = _compaction_entry("First summary", u1.id)  # keep from first entry
        u2 = _message_entry(_user_message("2"))
        b = _message_entry(_assistant_message("b"))

        entries = [u1, a1, compact1, u2, b]

        loaded = build_session_context(entries)
        # summary + all messages (u1, a1, u2, b) = 5
        assert len(loaded.messages) == 5

    def test_tracks_model_and_thinking_level_changes(self) -> None:
        entries = [
            _message_entry(_user_message("1")),
            _model_change_entry("openai", "gpt-4"),
            _message_entry(_assistant_message("a")),
            _thinking_level_entry("high"),
        ]
        loaded = build_session_context(entries)
        assert loaded.model.provider == "anthropic"
        assert loaded.model.model_id == "claude-sonnet-4-5"
        assert loaded.thinking_level == "high"


class TestPrepareCompactionWithPreviousCompaction:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_skips_repeated_compaction_when_kept_messages_still_fit(self) -> None:
        u1 = _message_entry(_user_message("user msg 1 (summarized by compaction1)"))
        a1 = _message_entry(_assistant_message("assistant msg 1"))
        u2 = _message_entry(_user_message("user msg 2 - kept by compaction1"))
        a2 = _message_entry(_assistant_message("assistant msg 2"))
        u3 = _message_entry(_user_message("user msg 3 - kept by compaction1"))
        a3 = _message_entry(_assistant_message("assistant msg 3", _mock_usage(5000, 1000)))
        compaction1 = _compaction_entry("First summary", u2.id)
        u4 = _message_entry(_user_message("user msg 4 (new after compaction1)"))
        a4 = _message_entry(_assistant_message("assistant msg 4", _mock_usage(8000, 2000)))

        path_entries = [u1, a1, u2, a2, u3, a3, compaction1, u4, a4]
        preparation = prepare_compaction(path_entries, DEFAULT_COMPACTION_SETTINGS)

        assert preparation is None

    def test_resummarizes_kept_messages_when_recent_window_moves_past_them(self) -> None:
        u1 = _message_entry(_user_message("user msg 1 (summarized by compaction1)" * 4))
        a1 = _message_entry(_assistant_message("assistant msg 1" * 4))
        u2 = _message_entry(_user_message("user msg 2 - kept by compaction1 " * 12))
        a2 = _message_entry(_assistant_message("assistant msg 2 " * 12))
        u3 = _message_entry(_user_message("user msg 3 - kept by compaction1 " * 12))
        a3 = _message_entry(_assistant_message("assistant msg 3 " * 12, _mock_usage(5000, 1000)))
        compaction1 = _compaction_entry("First summary", u2.id)
        u4 = _message_entry(_user_message("user msg 4 (new after compaction1) " * 12))
        a4 = _message_entry(_assistant_message("assistant msg 4 " * 12, _mock_usage(8000, 2000)))

        settings = replace(DEFAULT_COMPACTION_SETTINGS, keep_recent_tokens=100)
        preparation = prepare_compaction([u1, a1, u2, a2, u3, a3, compaction1, u4, a4], settings)

        assert preparation is not None
        summarized_text = _extract_text(preparation.messages_to_summarize)
        assert "user msg 2 - kept by compaction1" in summarized_text
        assert "user msg 3 - kept by compaction1" in summarized_text
        assert "First summary" not in summarized_text
        assert preparation.previous_summary == "First summary"


class TestLargeSessionFixture:
    """Port of the TS `describe("Large session fixture")` block.

    The fixture is the head of the TypeScript `test/fixtures/large-session.jsonl`
    (a real v1 session with no `id`/`parentId`), trimmed to the first 162 lines so
    that the repo does not carry a 1MB blob. The trim keeps 160 message entries, so
    every TS assertion (`> 100` entries / messages) holds on exactly the same data.
    """

    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_parses_the_large_session(self) -> None:
        entries = _load_large_session_entries()
        assert len(entries) > 100

        message_count = len([e for e in entries if isinstance(e, SessionMessageEntry)])
        assert message_count > 100

    def test_finds_cut_point_in_large_session(self) -> None:
        entries = _load_large_session_entries()
        result = find_cut_point(entries, 0, len(entries), DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens)

        # Cut point should be at a message entry (user or assistant).
        cut_entry = entries[result.first_kept_entry_index]
        assert isinstance(cut_entry, SessionMessageEntry)
        assert cut_entry.message.role in ("user", "assistant")

    def test_loads_session_correctly(self) -> None:
        entries = _load_large_session_entries()
        loaded = build_session_context(entries)

        assert len(loaded.messages) > 100
        assert loaded.model is not None


class TestLLMSummarization:
    """Port of the TS `describe.skipIf(!ANTHROPIC_OAUTH_TOKEN)("LLM summarization")` block.

    The TypeScript cases issue a real Anthropic call. Here the same `compact()` code
    path runs against a scripted stream function, so the structural assertions
    (a non-empty summary threaded into the result, a truthy `firstKeptEntryId`, a
    non-zero `tokensBefore`, and a reloaded context that is strictly smaller and
    starts with the summary) are checked without a network call.
    """

    def setup_method(self) -> None:
        _reset_entry_counter()

    _SUMMARY_TEXT = "The user asked about the pi codebase and the assistant explored it. " * 3

    def _model(self) -> Model:
        return Model(
            id="claude-sonnet-4-5",
            api="anthropic-messages",
            provider="anthropic",
            context_window=200000,
            max_tokens=8192,
        )

    def _stream_fn(self):
        from pi_ai import (
            AssistantMessageEventStream,
            DoneEvent,
            StartEvent,
            TextDeltaEvent,
            TextEndEvent,
            TextStartEvent,
        )

        text = self._SUMMARY_TEXT

        def stream_fn(_model, _context, _options):
            stream = AssistantMessageEventStream()
            message = _assistant_message(text, _mock_usage(1000, 200))
            stream.push(StartEvent(partial=message))
            stream.push(TextStartEvent(content_index=0, partial=message))
            stream.push(TextDeltaEvent(content_index=0, delta=text, partial=message))
            stream.push(TextEndEvent(content_index=0, content=text, partial=message))
            stream.push(DoneEvent(reason="stop", message=message))
            stream.end()
            return stream

        return stream_fn

    def test_generates_a_compaction_result_for_the_large_session(self) -> None:
        entries = _load_large_session_entries()

        preparation = prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS)
        assert preparation is not None

        result = asyncio.run(
            asyncio.wait_for(compact(preparation, self._stream_fn(), self._model()), timeout=10),
        )

        assert len(result.summary) > 100
        assert result.first_kept_entry_id
        assert result.tokens_before > 0

    def test_produces_valid_session_after_compaction(self) -> None:
        entries = _load_large_session_entries()
        loaded = build_session_context(entries)

        preparation = prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS)
        assert preparation is not None

        result = asyncio.run(
            asyncio.wait_for(compact(preparation, self._stream_fn(), self._model()), timeout=10),
        )

        compaction_entry = CompactionEntry(
            id="compaction-test-id",
            parent_id=entries[-1].id,
            timestamp="2025-01-01T00:00:00Z",
            summary=result.summary,
            first_kept_entry_id=result.first_kept_entry_id,
            tokens_before=result.tokens_before,
        )
        reloaded = build_session_context([*entries, compaction_entry])

        assert len(reloaded.messages) < len(loaded.messages)
        assert reloaded.messages[0].role == "compactionSummary"
        assert result.summary in reloaded.messages[0].summary


class TestCollectEntriesForBranchSummary:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def _manager(self, tmp_path) -> SessionManager:
        return SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"), persist=True)

    def test_returns_empty_when_no_old_leaf(self, tmp_path) -> None:
        manager = self._manager(tmp_path)
        target_id = manager.append_message(_user_message("hi"))
        result = collect_entries_for_branch_summary(manager, None, target_id)
        assert result.entries == []
        assert result.common_ancestor_id is None

    def test_collects_entries_back_to_common_ancestor(self, tmp_path) -> None:
        manager = self._manager(tmp_path)
        root_id = manager.append_message(_user_message("root"))
        manager.branch(root_id)
        old_leaf = manager.append_message(_assistant_message("old branch a"))
        old_leaf = manager.append_message(_user_message("old branch b"))

        manager.branch(root_id)
        target_id = manager.append_message(_user_message("new branch"))

        result = collect_entries_for_branch_summary(manager, old_leaf, target_id)
        assert result.common_ancestor_id == root_id
        assert [e.id for e in result.entries] == [e.id for e in manager.get_branch(old_leaf) if e.id != root_id]


class TestPrepareBranchEntries:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_includes_all_messages_with_no_budget(self) -> None:
        entries = [
            _message_entry(_user_message("1")),
            _message_entry(_assistant_message("a")),
            _message_entry(_user_message("2")),
            _message_entry(_assistant_message("b")),
        ]
        preparation = prepare_branch_entries(entries)
        assert len(preparation.messages) == 4
        assert preparation.total_tokens > 0

    def test_skips_tool_result_messages(self) -> None:
        from pi_ai.types import ToolResultMessage

        entries = [
            _message_entry(_user_message("1")),
            _message_entry(_assistant_message("a")),
            _message_entry(ToolResultMessage(tool_call_id="x", tool_name="t", content="result")),
        ]
        preparation = prepare_branch_entries(entries)
        assert [m.role for m in preparation.messages] == ["user", "assistant"]

    def test_respects_token_budget_keeping_most_recent(self) -> None:
        entries = [
            _message_entry(_user_message("1" * 4000)),
            _message_entry(_assistant_message("2" * 4000)),
            _message_entry(_user_message("3" * 4000)),
        ]
        # Budget large enough for exactly one ~4000-char message's estimated tokens.
        single_message_tokens = prepare_branch_entries([entries[-1]]).total_tokens
        preparation = prepare_branch_entries(entries, token_budget=single_message_tokens)
        # Only the most recent message should fit (walking newest to oldest).
        assert len(preparation.messages) == 1
        assert preparation.messages[0].content == entries[-1].message.content


class TestGenerateBranchSummary:
    def setup_method(self) -> None:
        _reset_entry_counter()

    def test_returns_no_content_message_for_empty_entries(self) -> None:
        model = Model(id="test-model", api="test-api", provider="test", context_window=1000, max_tokens=100)
        options = GenerateBranchSummaryOptions(model=model, stream_fn=lambda *a, **k: None)
        result = asyncio.run(asyncio.wait_for(generate_branch_summary([], options), timeout=5))
        assert result.summary == "No content to summarize"

    def test_full_summary_generation_with_scripted_stream(self) -> None:
        from pi_ai import (
            AssistantMessageEventStream,
            DoneEvent,
            StartEvent,
            TextDeltaEvent,
            TextEndEvent,
            TextStartEvent,
        )

        model = Model(id="test-model", api="test-api", provider="test", context_window=1000, max_tokens=100)

        def stream_fn(_model, _context, _options):
            stream = AssistantMessageEventStream()
            summary_message = _assistant_message("## Goal\nDid the thing.")
            stream.push(StartEvent(partial=summary_message))
            stream.push(TextStartEvent(content_index=0, partial=summary_message))
            stream.push(TextDeltaEvent(content_index=0, delta="## Goal\nDid the thing.", partial=summary_message))
            stream.push(TextEndEvent(content_index=0, content="## Goal\nDid the thing.", partial=summary_message))
            stream.push(DoneEvent(reason="stop", message=summary_message))
            stream.end()
            return stream

        entries = [
            _message_entry(_user_message("Please do the thing")),
            _message_entry(_assistant_message("Sure, doing the thing")),
        ]
        options = GenerateBranchSummaryOptions(model=model, stream_fn=stream_fn)
        result = asyncio.run(asyncio.wait_for(generate_branch_summary(entries, options), timeout=5))

        assert result.aborted is False
        assert result.error is None
        assert result.summary is not None
        assert "Did the thing" in result.summary
        assert result.summary.startswith("The user explored a different conversation branch")
