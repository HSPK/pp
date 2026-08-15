"""Tests for `pi_coding_agent.core.session_manager`, ported from the TypeScript
suite under `packages/coding-agent/test/session-manager/`:
`packages/coding-agent/test/session-manager/build-context.test.ts`,
`packages/coding-agent/test/session-manager/custom-session-id.test.ts`,
`packages/coding-agent/test/session-manager/migration.test.ts`,
`packages/coding-agent/test/session-manager/save-entry.test.ts`,
`packages/coding-agent/test/session-manager/tree-traversal.test.ts`, and
`packages/coding-agent/test/sdk-session-manager.test.ts`.

`labels.test.ts` and `file-operations.test.ts` are ported separately in
`tests/test_session_manager_labels.py` and
`tests/test_session_manager_file_operations.py`.

All filesystem access goes through `tmp_path` — no test touches the real
user home directory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from pi_ai.types import AssistantMessage, Cost, TextContent, Usage, UserMessage

from pi_coding_agent.core.session_manager import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    ModelChangeEntry,
    SessionHeader,
    SessionManager,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
    build_context_entries,
    build_session_context,
    load_entries_from_file,
    migrate_session_entries,
)

UUID_V7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _assistant_usage() -> Usage:
    return Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=Cost())


def msg(entry_id: str, parent_id: str | None, role: str, text: str) -> SessionMessageEntry:
    if role == "user":
        message = UserMessage(content=text, timestamp=1)
    else:
        message = AssistantMessage(
            api="anthropic-messages",
            provider="anthropic",
            model="claude-test",
            content=[TextContent(text=text)],
            usage=_assistant_usage(),
            stop_reason="stop",
            timestamp=1,
        )
    return SessionMessageEntry(id=entry_id, parent_id=parent_id, timestamp="2025-01-01T00:00:00Z", message=message)


def compaction(entry_id: str, parent_id: str | None, summary: str, first_kept_entry_id: str) -> CompactionEntry:
    return CompactionEntry(
        id=entry_id,
        parent_id=parent_id,
        timestamp="2025-01-01T00:00:00Z",
        summary=summary,
        first_kept_entry_id=first_kept_entry_id,
        tokens_before=1000,
    )


def branch_summary(entry_id: str, parent_id: str | None, summary: str, from_id: str) -> BranchSummaryEntry:
    return BranchSummaryEntry(
        id=entry_id, parent_id=parent_id, timestamp="2025-01-01T00:00:00Z", summary=summary, from_id=from_id
    )


def custom(entry_id: str, parent_id: str | None, custom_type: str, data=None) -> CustomEntry:
    return CustomEntry(
        id=entry_id, parent_id=parent_id, timestamp="2025-01-01T00:00:00Z", custom_type=custom_type, data=data
    )


def thinking_level(entry_id: str, parent_id: str | None, level: str) -> ThinkingLevelChangeEntry:
    return ThinkingLevelChangeEntry(
        id=entry_id, parent_id=parent_id, timestamp="2025-01-01T00:00:00Z", thinking_level=level
    )


def model_change(entry_id: str, parent_id: str | None, provider: str, model_id: str) -> ModelChangeEntry:
    return ModelChangeEntry(
        id=entry_id, parent_id=parent_id, timestamp="2025-01-01T00:00:00Z", provider=provider, model_id=model_id
    )


# ---------------------------------------------------------------------------
# build_session_context / build_context_entries
# ---------------------------------------------------------------------------


class TestBuildSessionContext:
    def test_empty_entries_returns_empty_context(self) -> None:
        ctx = build_session_context([])
        assert ctx.messages == []
        assert ctx.thinking_level == "off"
        assert ctx.model is None

    def test_single_user_message(self) -> None:
        ctx = build_session_context([msg("1", None, "user", "hello")])
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"

    def test_simple_conversation(self) -> None:
        entries = [
            msg("1", None, "user", "hello"),
            msg("2", "1", "assistant", "hi there"),
            msg("3", "2", "user", "how are you"),
            msg("4", "3", "assistant", "great"),
        ]
        ctx = build_session_context(entries)
        assert len(ctx.messages) == 4
        assert [m.role for m in ctx.messages] == ["user", "assistant", "user", "assistant"]

    def test_tracks_thinking_level_changes(self) -> None:
        entries = [
            msg("1", None, "user", "hello"),
            thinking_level("2", "1", "high"),
            msg("3", "2", "assistant", "thinking hard"),
        ]
        ctx = build_session_context(entries)
        assert ctx.thinking_level == "high"
        assert len(ctx.messages) == 2

    def test_tracks_model_from_assistant_message(self) -> None:
        entries = [msg("1", None, "user", "hello"), msg("2", "1", "assistant", "hi")]
        ctx = build_session_context(entries)
        assert ctx.model.provider == "anthropic"
        assert ctx.model.model_id == "claude-test"

    def test_tracks_model_from_model_change_entry_overwritten_by_assistant(self) -> None:
        entries = [
            msg("1", None, "user", "hello"),
            model_change("2", "1", "openai", "gpt-4"),
            msg("3", "2", "assistant", "hi"),
        ]
        ctx = build_session_context(entries)
        assert ctx.model.provider == "anthropic"
        assert ctx.model.model_id == "claude-test"

    def test_compaction_includes_summary_before_kept_messages(self) -> None:
        entries = [
            msg("1", None, "user", "first"),
            msg("2", "1", "assistant", "response1"),
            msg("3", "2", "user", "second"),
            msg("4", "3", "assistant", "response2"),
            compaction("5", "4", "Summary of first two turns", "3"),
            msg("6", "5", "user", "third"),
            msg("7", "6", "assistant", "response3"),
        ]
        ctx = build_session_context(entries)
        assert len(ctx.messages) == 5
        assert "Summary of first two turns" in ctx.messages[0].summary
        assert ctx.messages[1].content == "second"
        assert ctx.messages[2].content[0].text == "response2"
        assert ctx.messages[3].content == "third"
        assert ctx.messages[4].content[0].text == "response3"

    def test_compaction_keeping_from_first_message(self) -> None:
        entries = [
            msg("1", None, "user", "first"),
            msg("2", "1", "assistant", "response"),
            compaction("3", "2", "Empty summary", "1"),
            msg("4", "3", "user", "second"),
        ]
        ctx = build_session_context(entries)
        assert len(ctx.messages) == 4
        assert "Empty summary" in ctx.messages[0].summary

    def test_multiple_compactions_uses_latest(self) -> None:
        entries = [
            msg("1", None, "user", "a"),
            msg("2", "1", "assistant", "b"),
            compaction("3", "2", "First summary", "1"),
            msg("4", "3", "user", "c"),
            msg("5", "4", "assistant", "d"),
            compaction("6", "5", "Second summary", "4"),
            msg("7", "6", "user", "e"),
        ]
        ctx = build_session_context(entries)
        assert len(ctx.messages) == 4
        assert "Second summary" in ctx.messages[0].summary

    def test_build_context_entries_includes_custom_entries(self) -> None:
        entries = [
            msg("1", None, "user", "first"),
            custom("2", "1", "old-state", {"hidden": True}),
            msg("3", "2", "assistant", "response1"),
            custom("4", "3", "kept-card", {"title": "Kept"}),
            msg("5", "4", "user", "second"),
            compaction("6", "5", "Summary", "4"),
            custom("7", "6", "after-card", {"title": "After"}),
            msg("8", "7", "assistant", "response2"),
        ]
        assert [e.id for e in build_context_entries(entries)] == ["6", "4", "5", "7", "8"]
        ctx = build_session_context(entries)
        assert [m.role for m in ctx.messages] == ["compactionSummary", "user", "assistant"]

    def test_keeps_settings_from_full_path_after_compaction(self) -> None:
        entries = [
            msg("1", None, "user", "first"),
            thinking_level("2", "1", "high"),
            msg("3", "2", "assistant", "response1"),
            msg("4", "3", "user", "second"),
            compaction("5", "4", "Summary", "4"),
        ]
        ctx = build_session_context(entries)
        assert ctx.thinking_level == "high"
        assert [m.role for m in ctx.messages] == ["compactionSummary", "user"]

    def test_follows_path_to_specified_leaf(self) -> None:
        entries = [
            msg("1", None, "user", "start"),
            msg("2", "1", "assistant", "response"),
            msg("3", "2", "user", "branch A"),
            msg("4", "2", "user", "branch B"),
        ]
        ctx_a = build_session_context(entries, "3")
        assert len(ctx_a.messages) == 3
        assert ctx_a.messages[2].content == "branch A"

        ctx_b = build_session_context(entries, "4")
        assert len(ctx_b.messages) == 3
        assert ctx_b.messages[2].content == "branch B"

    def test_includes_branch_summary_in_path(self) -> None:
        entries = [
            msg("1", None, "user", "start"),
            msg("2", "1", "assistant", "response"),
            msg("3", "2", "user", "abandoned path"),
            branch_summary("4", "2", "Summary of abandoned work", "3"),
            msg("5", "4", "user", "new direction"),
        ]
        ctx = build_session_context(entries, "5")
        assert len(ctx.messages) == 4
        assert "Summary of abandoned work" in ctx.messages[2].summary
        assert ctx.messages[3].content == "new direction"

    def test_complex_tree_with_multiple_branches_and_compaction(self) -> None:
        entries = [
            msg("1", None, "user", "start"),
            msg("2", "1", "assistant", "r1"),
            msg("3", "2", "user", "q2"),
            msg("4", "3", "assistant", "r2"),
            compaction("5", "4", "Compacted history", "3"),
            msg("6", "5", "user", "q3"),
            msg("7", "6", "assistant", "r3"),
            msg("8", "3", "user", "wrong path"),
            msg("9", "8", "assistant", "wrong response"),
            branch_summary("10", "3", "Tried wrong approach", "9"),
            msg("11", "10", "user", "better approach"),
        ]
        ctx_main = build_session_context(entries, "7")
        assert len(ctx_main.messages) == 5
        assert "Compacted history" in ctx_main.messages[0].summary
        assert ctx_main.messages[1].content == "q2"
        assert ctx_main.messages[2].content[0].text == "r2"
        assert ctx_main.messages[3].content == "q3"
        assert ctx_main.messages[4].content[0].text == "r3"

        ctx_branch = build_session_context(entries, "11")
        assert len(ctx_branch.messages) == 5
        assert ctx_branch.messages[0].content == "start"
        assert ctx_branch.messages[1].content[0].text == "r1"
        assert ctx_branch.messages[2].content == "q2"
        assert "Tried wrong approach" in ctx_branch.messages[3].summary
        assert ctx_branch.messages[4].content == "better approach"

    def test_uses_last_entry_when_leaf_id_not_found(self) -> None:
        entries = [msg("1", None, "user", "hello"), msg("2", "1", "assistant", "hi")]
        ctx = build_session_context(entries, "nonexistent")
        assert len(ctx.messages) == 2

    def test_handles_orphaned_entries_gracefully(self) -> None:
        entries = [msg("1", None, "user", "hello"), msg("2", "missing", "assistant", "orphan")]
        ctx = build_session_context(entries, "2")
        assert len(ctx.messages) == 1


# ---------------------------------------------------------------------------
# migrate_session_entries
# ---------------------------------------------------------------------------


class TestMigrateSessionEntries:
    def test_adds_id_parent_id_to_v1_entries(self) -> None:
        entries = [
            {"type": "session", "id": "sess-1", "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
            {
                "type": "message",
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {"role": "user", "content": "hi", "timestamp": 1},
            },
            {
                "type": "message",
                "timestamp": "2025-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "api": "test",
                    "provider": "test",
                    "model": "test",
                    "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    "stopReason": "stop",
                    "timestamp": 2,
                },
            },
        ]
        migrate_session_entries(entries)
        assert entries[0]["version"] == 3
        assert len(entries[1]["id"]) == 8
        assert entries[1]["parentId"] is None
        assert len(entries[2]["id"]) == 8
        assert entries[2]["parentId"] == entries[1]["id"]

    def test_idempotent_skip_already_migrated(self) -> None:
        entries = [
            {"type": "session", "id": "sess-1", "version": 2, "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
            {
                "type": "message",
                "id": "abc12345",
                "parentId": None,
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {"role": "user", "content": "hi", "timestamp": 1},
            },
            {
                "type": "message",
                "id": "def67890",
                "parentId": "abc12345",
                "timestamp": "2025-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "api": "test",
                    "provider": "test",
                    "model": "test",
                    "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    "stopReason": "stop",
                    "timestamp": 2,
                },
            },
        ]
        migrate_session_entries(entries)
        assert entries[1]["id"] == "abc12345"
        assert entries[2]["id"] == "def67890"
        assert entries[2]["parentId"] == "abc12345"


# ---------------------------------------------------------------------------
# load_entries_from_file
# ---------------------------------------------------------------------------


class TestLoadEntriesFromFile:
    def test_returns_empty_for_nonexistent_file(self, tmp_path: Path) -> None:
        assert load_entries_from_file(tmp_path / "nonexistent.jsonl") == []

    def test_returns_empty_for_empty_file(self, tmp_path: Path) -> None:
        file = tmp_path / "empty.jsonl"
        file.write_text("")
        assert load_entries_from_file(file) == []

    def test_returns_empty_for_file_without_valid_header(self, tmp_path: Path) -> None:
        file = tmp_path / "no-header.jsonl"
        file.write_text('{"type":"message","id":"1"}\n')
        assert load_entries_from_file(file) == []

    def test_returns_empty_for_malformed_json(self, tmp_path: Path) -> None:
        file = tmp_path / "malformed.jsonl"
        file.write_text("not json\n")
        assert load_entries_from_file(file) == []

    def test_loads_valid_session_file(self, tmp_path: Path) -> None:
        file = tmp_path / "valid.jsonl"
        file.write_text(
            '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
            '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z",'
            '"message":{"role":"user","content":"hi","timestamp":1}}\n'
        )
        entries = load_entries_from_file(file)
        assert len(entries) == 2
        assert entries[0].type == "session"
        assert entries[1].type == "message"

    def test_skips_malformed_lines_but_keeps_valid_ones(self, tmp_path: Path) -> None:
        file = tmp_path / "mixed.jsonl"
        file.write_text(
            '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
            "not valid json\n"
            '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z",'
            '"message":{"role":"user","content":"hi","timestamp":1}}\n'
        )
        entries = load_entries_from_file(file)
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# SessionManager.appendCustomEntry / labels / tree
# ---------------------------------------------------------------------------


class TestSaveCustomEntry:
    def test_saves_custom_entries_and_includes_in_tree_traversal(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(UserMessage(content="hello", timestamp=1))
        custom_id = session.append_custom_entry("my_data", {"foo": "bar"})
        msg2_id = session.append_message(
            AssistantMessage(
                api="anthropic-messages",
                provider="anthropic",
                model="test",
                content=[TextContent(text="hi")],
                usage=_assistant_usage(),
                stop_reason="stop",
                timestamp=2,
            )
        )

        entries = session.get_entries()
        assert len(entries) == 3

        custom_entry = next(e for e in entries if e.type == "custom")
        assert custom_entry.custom_type == "my_data"
        assert custom_entry.data == {"foo": "bar"}
        assert custom_entry.id == custom_id
        assert custom_entry.parent_id == msg_id

        path = session.get_branch()
        assert len(path) == 3
        assert path[0].id == msg_id
        assert path[1].id == custom_id
        assert path[2].id == msg2_id

        ctx = session.build_session_context()
        assert len(ctx.messages) == 2


class TestLabels:
    def test_sets_and_gets_labels(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(UserMessage(content="hello", timestamp=1))
        assert session.get_label(msg_id) is None

        label_id = session.append_label_change(msg_id, "checkpoint")
        assert session.get_label(msg_id) == "checkpoint"

        entries = session.get_entries()
        label_entry = next(e for e in entries if e.type == "label")
        assert label_entry.id == label_id
        assert label_entry.target_id == msg_id
        assert label_entry.label == "checkpoint"

    def test_clears_labels_with_none(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(UserMessage(content="hello", timestamp=1))
        session.append_label_change(msg_id, "checkpoint")
        assert session.get_label(msg_id) == "checkpoint"
        session.append_label_change(msg_id, None)
        assert session.get_label(msg_id) is None

    def test_last_label_wins(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(UserMessage(content="hello", timestamp=1))
        session.append_label_change(msg_id, "first")
        session.append_label_change(msg_id, "second")
        last_label_id = session.append_label_change(msg_id, "third")
        assert session.get_label(msg_id) == "third"

        entries = session.get_entries()
        last_label_entry = next(e for e in entries if e.id == last_label_id)
        tree = session.get_tree()
        msg_node = next(n for n in tree if n.entry.id == msg_id)
        assert msg_node.label_timestamp == last_label_entry.timestamp


# ---------------------------------------------------------------------------
# SessionManager.newSession with custom id
# ---------------------------------------------------------------------------


class TestNewSessionCustomId:
    def test_uses_provided_id(self) -> None:
        session = SessionManager.in_memory()
        from pi_coding_agent.core.session_manager import NewSessionOptions

        session.new_session(NewSessionOptions(id="my-custom-id"))
        assert session.get_session_id() == "my-custom-id"

    def test_uses_provided_id_when_creating_in_memory_session(self) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        session = SessionManager.in_memory(os.getcwd(), NewSessionOptions(id="memory-session-id"))
        assert session.get_session_id() == "memory-session-id"
        assert session.get_header().id == "memory-session-id"
        assert session.get_session_file() is None

    def test_allows_alphanumeric_ids_with_interior_punctuation(self) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        session = SessionManager.in_memory()
        session.new_session(NewSessionOptions(id="abc-123_def.456"))
        assert session.get_session_id() == "abc-123_def.456"

    def test_rejects_invalid_custom_session_ids(self) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        invalid_ids = ["", "-abc", "abc-", "_abc", "abc_", ".abc", "abc.", "abc/def", "abc\\def", "abc def"]
        for invalid_id in invalid_ids:
            session = SessionManager.in_memory()
            with pytest.raises(ValueError, match="Session id must be non-empty, contain only alphanumeric characters"):
                session.new_session(NewSessionOptions(id=invalid_id))

    def test_generates_uuidv7_id_when_no_id_provided(self) -> None:
        session = SessionManager.in_memory()
        session.new_session()
        session_id = session.get_session_id()
        assert session_id
        assert UUID_V7_RE.match(session_id)

    def test_generates_uuidv7_id_when_options_provided_without_id(self) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        session = SessionManager.in_memory()
        session.new_session(NewSessionOptions(parent_session="parent.jsonl"))
        session_id = session.get_session_id()
        assert session_id
        assert UUID_V7_RE.match(session_id)

    def test_includes_custom_id_in_header(self) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        session = SessionManager.in_memory()
        session.new_session(NewSessionOptions(id="header-test-id"))
        header = session.get_header()
        assert header is not None
        assert header.id == "header-test-id"

    def test_generates_uuidv7_id_without_explicit_id(self) -> None:
        session = SessionManager.in_memory()
        assert UUID_V7_RE.match(session.get_session_id())
        assert session.get_header().id == session.get_session_id()

    def test_uses_provided_id_when_creating_persisted_session(self, tmp_path: Path) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        session = SessionManager.create(str(tmp_path), str(tmp_path), NewSessionOptions(id="created-session-id"))
        assert session.get_session_id() == "created-session-id"
        assert session.get_header().id == "created-session-id"
        session_file = session.get_session_file()
        assert "created-session-id" in session_file
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z_created-session-id\.jsonl$", Path(session_file).name
        )
        assert not Path(session_file).exists()

    def test_generates_uuidv7_id_when_creating_branched_session(self) -> None:
        session = SessionManager.in_memory()
        first_id = session.append_message(UserMessage(content=[TextContent(text="hello")], timestamp=1))
        session.create_branched_session(first_id)
        assert UUID_V7_RE.match(session.get_session_id())
        assert session.get_header().id == session.get_session_id()

    def test_generates_uuidv7_id_when_forking(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.jsonl"
        source_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session",
                            "version": 3,
                            "id": "legacy-session-id",
                            "timestamp": "2025-01-01T00:00:00Z",
                            "cwd": str(tmp_path),
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message",
                            "id": "entry-1",
                            "parentId": None,
                            "timestamp": "2025-01-01T00:00:00Z",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "hello"}],
                                "api": "openai-responses",
                                "provider": "openai",
                                "model": "gpt-5.4",
                                "usage": {
                                    "input": 0,
                                    "output": 0,
                                    "cacheRead": 0,
                                    "cacheWrite": 0,
                                    "totalTokens": 0,
                                    "cost": {
                                        "input": 0,
                                        "output": 0,
                                        "cacheRead": 0,
                                        "cacheWrite": 0,
                                        "total": 0,
                                    },
                                },
                                "stopReason": "stop",
                                "timestamp": 1,
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        forked = SessionManager.fork_from(str(source_path), str(tmp_path), str(tmp_path))
        header = forked.get_header()
        assert header is not None
        assert UUID_V7_RE.match(header.id)
        assert header.parent_session == str(source_path)

    def test_uses_provided_id_when_forking(self, tmp_path: Path) -> None:
        from pi_coding_agent.core.session_manager import NewSessionOptions

        source_path = tmp_path / "source.jsonl"
        source_path.write_text(
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": "source-session-id",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "cwd": str(tmp_path),
                }
            )
            + "\n"
        )
        forked = SessionManager.fork_from(
            str(source_path), str(tmp_path), str(tmp_path), NewSessionOptions(id="forked-session-id")
        )
        header = forked.get_header()
        assert header is not None
        assert header.id == "forked-session-id"
        assert header.parent_session == str(source_path)
        session_file = forked.get_session_file()
        assert "forked-session-id" in session_file
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z_forked-session-id\.jsonl$", Path(session_file).name
        )


# ---------------------------------------------------------------------------
# Persistence roundtrip / append + reload
# ---------------------------------------------------------------------------


class TestPersistenceRoundtrip:
    def test_appended_entries_persist_and_reload(self, tmp_path: Path) -> None:
        session = SessionManager.create(str(tmp_path), str(tmp_path))
        session.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=1))
        session.append_message(
            AssistantMessage(
                api="a", provider="p", model="m", content=[TextContent(text="hello")], usage=Usage(), stop_reason="stop"
            )
        )
        session_file = session.get_session_file()
        assert session_file is not None
        assert Path(session_file).exists()

        entries = load_entries_from_file(session_file)
        assert len(entries) == 3
        assert isinstance(entries[0], SessionHeader)
        assert entries[1].message.role == "user"
        assert entries[2].message.role == "assistant"

    def test_branch_and_reset_leaf(self) -> None:
        session = SessionManager.in_memory()
        first_id = session.append_message(UserMessage(content="hello", timestamp=1))
        session.append_message(UserMessage(content="second", timestamp=2))
        assert len(session.get_branch()) == 2

        session.branch(first_id)
        assert session.get_leaf_id() == first_id
        assert len(session.get_branch()) == 1

        session.reset_leaf()
        assert session.get_leaf_id() is None
        assert session.get_branch() == []


# ---------------------------------------------------------------------------
# Append + tree traversal, ported from tree-traversal.test.ts
# ---------------------------------------------------------------------------


def user_msg(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def assistant_msg(text: str) -> AssistantMessage:
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="test",
        content=[TextContent(text=text)],
        usage=_assistant_usage(),
        stop_reason="stop",
        timestamp=1,
    )


class TestAppendOperations:
    def test_append_message_creates_parent_id_chain(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("first"))
        id2 = session.append_message(assistant_msg("second"))
        id3 = session.append_message(user_msg("third"))

        entries = session.get_entries()
        assert len(entries) == 3
        assert entries[0].id == id1
        assert entries[0].parent_id is None
        assert entries[0].type == "message"
        assert entries[1].id == id2
        assert entries[1].parent_id == id1
        assert entries[2].id == id3
        assert entries[2].parent_id == id2

    def test_append_thinking_level_change_integrates_into_tree(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(user_msg("hello"))
        thinking_id = session.append_thinking_level_change("high")
        session.append_message(assistant_msg("response"))

        entries = session.get_entries()
        assert len(entries) == 3
        thinking_entry = next(e for e in entries if e.type == "thinking_level_change")
        assert thinking_entry.id == thinking_id
        assert thinking_entry.parent_id == msg_id
        assert entries[2].parent_id == thinking_id

    def test_append_model_change_integrates_into_tree(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(user_msg("hello"))
        model_id = session.append_model_change("openai", "gpt-4")
        session.append_message(assistant_msg("response"))

        entries = session.get_entries()
        model_entry = next(e for e in entries if e.type == "model_change")
        assert model_entry.id == model_id
        assert model_entry.parent_id == msg_id
        assert model_entry.provider == "openai"
        assert model_entry.model_id == "gpt-4"
        assert entries[2].parent_id == model_id

    def test_append_compaction_integrates_into_tree(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        usage = Usage(input=10, output=20, cache_read=30, cache_write=40, total_tokens=100)
        compaction_id = session.append_compaction("summary", id1, 1000, None, False, usage)
        session.append_message(user_msg("3"))

        entries = session.get_entries()
        compaction_entry = next(e for e in entries if e.type == "compaction")
        assert compaction_entry.id == compaction_id
        assert compaction_entry.parent_id == id2
        assert compaction_entry.summary == "summary"
        assert compaction_entry.first_kept_entry_id == id1
        assert compaction_entry.tokens_before == 1000
        assert compaction_entry.usage == usage
        assert entries[3].parent_id == compaction_id

    def test_append_custom_entry_integrates_into_tree(self) -> None:
        session = SessionManager.in_memory()
        msg_id = session.append_message(user_msg("hello"))
        custom_id = session.append_custom_entry("my_data", {"key": "value"})
        session.append_message(assistant_msg("response"))

        entries = session.get_entries()
        custom_entry = next(e for e in entries if e.type == "custom")
        assert custom_entry.id == custom_id
        assert custom_entry.parent_id == msg_id
        assert custom_entry.custom_type == "my_data"
        assert custom_entry.data == {"key": "value"}
        assert entries[2].parent_id == custom_id

    def test_leaf_pointer_advances_after_each_append(self) -> None:
        session = SessionManager.in_memory()
        assert session.get_leaf_id() is None

        id1 = session.append_message(user_msg("1"))
        assert session.get_leaf_id() == id1

        id2 = session.append_message(assistant_msg("2"))
        assert session.get_leaf_id() == id2

        id3 = session.append_thinking_level_change("high")
        assert session.get_leaf_id() == id3


class TestGetPath:
    def test_empty_session_returns_empty(self) -> None:
        assert SessionManager.in_memory().get_branch() == []

    def test_single_entry_path(self) -> None:
        session = SessionManager.in_memory()
        entry_id = session.append_message(user_msg("hello"))
        path = session.get_branch()
        assert len(path) == 1
        assert path[0].id == entry_id

    def test_full_path_root_to_leaf(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        id3 = session.append_thinking_level_change("high")
        id4 = session.append_message(user_msg("3"))
        path = session.get_branch()
        assert [e.id for e in path] == [id1, id2, id3, id4]

    def test_path_from_specified_entry(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        session.append_message(user_msg("3"))
        session.append_message(assistant_msg("4"))
        path = session.get_branch(id2)
        assert [e.id for e in path] == [id1, id2]


class TestGetTree:
    def test_empty_session_returns_empty(self) -> None:
        assert SessionManager.in_memory().get_tree() == []

    def test_linear_session_single_root(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        id3 = session.append_message(user_msg("3"))

        tree = session.get_tree()
        assert len(tree) == 1
        root = tree[0]
        assert root.entry.id == id1
        assert len(root.children) == 1
        assert root.children[0].entry.id == id2
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].entry.id == id3
        assert root.children[0].children[0].children == []

    def test_tree_with_branches_after_branch(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        id3 = session.append_message(user_msg("3"))

        session.branch(id2)
        id4 = session.append_message(user_msg("4-branch"))

        tree = session.get_tree()
        assert len(tree) == 1
        root = tree[0]
        assert root.entry.id == id1
        assert len(root.children) == 1
        node2 = root.children[0]
        assert node2.entry.id == id2
        assert len(node2.children) == 2
        assert sorted(c.entry.id for c in node2.children) == sorted([id3, id4])

    def test_multiple_branches_at_same_point(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("root"))
        id2 = session.append_message(assistant_msg("response"))

        session.branch(id2)
        id_a = session.append_message(user_msg("branch-A"))
        session.branch(id2)
        id_b = session.append_message(user_msg("branch-B"))
        session.branch(id2)
        id_c = session.append_message(user_msg("branch-C"))

        tree = session.get_tree()
        node2 = tree[0].children[0]
        assert node2.entry.id == id2
        assert len(node2.children) == 3
        assert sorted(c.entry.id for c in node2.children) == sorted([id_a, id_b, id_c])

    def test_deep_branching(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        id3 = session.append_message(user_msg("3"))
        session.append_message(assistant_msg("4"))

        session.branch(id2)
        id5 = session.append_message(user_msg("5"))
        session.append_message(assistant_msg("6"))

        session.branch(id5)
        session.append_message(user_msg("7"))

        tree = session.get_tree()
        node2 = tree[0].children[0]
        assert len(node2.children) == 2
        node5 = next(c for c in node2.children if c.entry.id == id5)
        assert len(node5.children) == 2
        node3 = next(c for c in node2.children if c.entry.id == id3)
        assert len(node3.children) == 1


class TestBranch:
    def test_moves_leaf_pointer(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        session.append_message(assistant_msg("2"))
        id3 = session.append_message(user_msg("3"))
        assert session.get_leaf_id() == id3

        session.branch(id1)
        assert session.get_leaf_id() == id1

    def test_throws_for_nonexistent_entry(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("hello"))
        with pytest.raises(ValueError, match="Entry nonexistent not found"):
            session.branch("nonexistent")

    def test_new_appends_become_children_of_branch_point(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        session.append_message(assistant_msg("2"))

        session.branch(id1)
        id3 = session.append_message(user_msg("branched"))

        entries = session.get_entries()
        branched_entry = next(e for e in entries if e.id == id3)
        assert branched_entry.parent_id == id1


class TestBranchWithSummary:
    def test_inserts_branch_summary_and_advances_leaf(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        session.append_message(assistant_msg("2"))
        session.append_message(user_msg("3"))

        usage = Usage(input=10, output=20, cache_read=30, cache_write=40, total_tokens=100)
        summary_id = session.branch_with_summary(id1, "Summary of abandoned work", None, False, usage)
        assert session.get_leaf_id() == summary_id

        entries = session.get_entries()
        summary_entry = next(e for e in entries if e.type == "branch_summary")
        assert summary_entry.parent_id == id1
        assert summary_entry.summary == "Summary of abandoned work"
        assert summary_entry.usage == usage

    def test_throws_for_nonexistent_entry(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("hello"))
        with pytest.raises(ValueError, match="Entry nonexistent not found"):
            session.branch_with_summary("nonexistent", "summary")


class TestGetLeafEntry:
    def test_returns_none_for_empty_session(self) -> None:
        assert SessionManager.in_memory().get_leaf_entry() is None

    def test_returns_current_leaf_entry(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        leaf = session.get_leaf_entry()
        assert leaf is not None
        assert leaf.id == id2


class TestGetEntry:
    def test_returns_none_for_nonexistent_id(self) -> None:
        assert SessionManager.in_memory().get_entry("nonexistent") is None

    def test_returns_entry_by_id(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("first"))
        id2 = session.append_message(assistant_msg("second"))

        entry1 = session.get_entry(id1)
        assert entry1 is not None
        assert entry1.type == "message"
        assert entry1.message.content == "first"

        entry2 = session.get_entry(id2)
        assert entry2 is not None
        assert entry2.message.content[0].text == "second"


class TestBuildSessionContextWithBranches:
    def test_returns_messages_from_current_branch_only(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("msg1"))
        id2 = session.append_message(assistant_msg("msg2"))
        session.append_message(user_msg("msg3"))

        session.branch(id2)
        session.append_message(assistant_msg("msg4-branch"))

        ctx = session.build_session_context()
        assert len(ctx.messages) == 3
        assert ctx.messages[0].content == "msg1"
        assert ctx.messages[1].content[0].text == "msg2"
        assert ctx.messages[2].content[0].text == "msg4-branch"


class TestCreateBranchedSession:
    def test_throws_for_nonexistent_entry(self) -> None:
        session = SessionManager.in_memory()
        session.append_message(user_msg("hello"))
        with pytest.raises(ValueError, match="Entry nonexistent not found"):
            session.create_branched_session("nonexistent")

    def test_creates_new_in_memory_session_with_path_to_leaf(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        id3 = session.append_message(user_msg("3"))
        session.append_message(assistant_msg("4"))

        session.branch(id3)
        session.append_message(user_msg("5"))

        result = session.create_branched_session(id2)
        assert result is None

        entries = session.get_entries()
        assert len(entries) == 2
        assert entries[0].id == id1
        assert entries[1].id == id2

    def test_extracts_correct_path_from_branched_tree(self) -> None:
        session = SessionManager.in_memory()
        id1 = session.append_message(user_msg("1"))
        id2 = session.append_message(assistant_msg("2"))
        session.append_message(user_msg("3"))

        session.branch(id2)
        id4 = session.append_message(user_msg("4"))
        id5 = session.append_message(assistant_msg("5"))

        session.create_branched_session(id5)

        entries = session.get_entries()
        assert [e.id for e in entries] == [id1, id2, id4, id5]

    def test_no_duplicate_entries_when_forking_from_first_user_message(self, tmp_path: Path) -> None:
        session = SessionManager.create(str(tmp_path), str(tmp_path))
        id1 = session.append_message(user_msg("first question"))
        session.append_message(assistant_msg("first answer"))
        session.append_message(user_msg("second question"))
        session.append_message(assistant_msg("second answer"))

        new_file = session.create_branched_session(id1)
        assert new_file is not None
        assert not Path(new_file).exists()

        session.append_custom_entry("preset-state", {"name": "plan"})
        session.append_message(assistant_msg("new answer"))

        assert Path(new_file).exists()
        records = [json.loads(line) for line in Path(new_file).read_text().splitlines() if line.strip()]
        assert len([r for r in records if r["type"] == "session"]) == 1
        entry_ids = [r["id"] for r in records if r["type"] != "session"]
        assert len(set(entry_ids)) == len(entry_ids)

    def test_preserves_usage_across_file_backed_reload(self, tmp_path: Path) -> None:
        from pi_ai.types import ToolResultMessage

        session = SessionManager.create(str(tmp_path), str(tmp_path))
        root_id = session.append_message(user_msg("question"))
        session.append_message(assistant_msg("answer"))
        # TS's `usage` literal carries a `cost` object too; keeping it here means
        # the reload assertions also pin cost round-tripping, not just tokens.
        usage = Usage(
            input=10,
            output=20,
            cache_read=30,
            cache_write=40,
            total_tokens=100,
            cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
        )
        session.append_message(
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="nested-model",
                content=[TextContent(text="result")],
                is_error=False,
                usage=usage,
                timestamp=1,
            )
        )
        session.append_compaction("summary", root_id, 100, None, False, usage)
        session.branch_with_summary(root_id, "branch summary", None, False, usage)

        file = session.get_session_file()
        assert file is not None
        reopened = SessionManager.open(file, str(tmp_path))
        entries = reopened.get_entries()

        compaction_entry = next(e for e in entries if e.type == "compaction")
        assert compaction_entry.usage == usage
        branch_summary_entry = next(e for e in entries if e.type == "branch_summary")
        assert branch_summary_entry.usage == usage
        tool_result_entry = next(e for e in entries if e.type == "message" and e.message.role == "toolResult")
        assert tool_result_entry.message.usage == usage

    def test_writes_file_immediately_when_forking_with_assistant_messages(self, tmp_path: Path) -> None:
        session = SessionManager.create(str(tmp_path), str(tmp_path))
        session.append_message(user_msg("first question"))
        id2 = session.append_message(assistant_msg("first answer"))
        session.append_message(user_msg("second question"))
        session.append_message(assistant_msg("second answer"))

        new_file = session.create_branched_session(id2)
        assert new_file is not None
        assert Path(new_file).exists()
        records = [json.loads(line) for line in Path(new_file).read_text().splitlines() if line.strip()]
        assert len([r for r in records if r["type"] == "session"]) == 1
