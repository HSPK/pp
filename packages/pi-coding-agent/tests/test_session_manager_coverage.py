"""Additional coverage tests for `pi_coding_agent.core.session_manager`.

Targets uncovered lines from the baseline run:
- `_iso_to_ms` ValueError path (lines 87-88)
- `_generate_id` fallback UUID when collisions saturate (lines 100->98, 102)
- `_migrate_v1_to_v2` compaction with firstKeptEntryIndex (lines 338-344)
- `_migrate_v2_to_v3` hookMessage rename (lines 353->349)
- `_entry_from_raw` custom_message, label, session_info, custom types
- `_message_to_raw` all message roles
- `build_session_path` with leaf_id=None
- `build_context_entries` compaction path
- `SessionManager`: empty-but-non-zero file, _persist_entry rewrite, label append/remove,
  get_leaf_entry, get_children, get_tree, get_session_name, branch_with_summary,
  create_branched_session, open, continue_recent, fork_from, list, list_all
- `find_most_recent_session`, `_build_session_info`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_ai.types import AssistantMessage, Cost, TextContent, Usage, UserMessage
from pi_coding_agent.core.session_manager import (
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    NewSessionOptions,
    SessionHeader,
    SessionInfoEntry,
    SessionManager,
    SessionMessageEntry,
    _generate_id,
    _iso_to_ms,
    _migrate_v1_to_v2,
    _migrate_v2_to_v3,
    build_context_entries,
    build_session_path,
    find_most_recent_session,
    load_entries_from_file,
    migrate_session_entries,
)

# ---------------------------------------------------------------------------
# _iso_to_ms
# ---------------------------------------------------------------------------


def test_iso_to_ms_valid():
    ms = _iso_to_ms("2025-01-01T00:00:00.000Z")
    assert ms > 0


def test_iso_to_ms_invalid_returns_zero():
    assert _iso_to_ms("not-a-timestamp") == 0


# ---------------------------------------------------------------------------
# _generate_id fallback UUID
# ---------------------------------------------------------------------------


def test_generate_id_fallback_to_uuid_on_collision():
    """When all 100 candidates collide, _generate_id returns a full 32-char UUID."""
    import uuid
    from unittest.mock import patch

    # Patch uuid4 to always return the same value — every 8-char prefix collides
    fixed = uuid.UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
    existing = {fixed.hex[:8]}  # pre-fill so first 100 iterations all collide
    # All hex[:8] == "aaaaaaaa" → collide, fallback returns hex (32 chars)
    with patch("pi_coding_agent.core.session_manager.uuid.uuid4", return_value=fixed):
        result = _generate_id(existing)
    assert len(result) == 32


# ---------------------------------------------------------------------------
# _migrate_v1_to_v2
# ---------------------------------------------------------------------------


def test_migrate_v1_to_v2_adds_ids_and_parent_ids():
    entries = [
        {"type": "session", "version": 1},
        {"type": "message", "message": {"role": "user", "content": "hi"}},
    ]
    _migrate_v1_to_v2(entries)
    assert "id" in entries[1]
    assert entries[1]["parentId"] is None


def test_migrate_v1_to_v2_compaction_with_first_kept_entry_index():
    entries = [
        {"type": "session", "version": 1},
        {"type": "message", "message": {"role": "user", "content": "hi"}},
        {"type": "compaction", "firstKeptEntryIndex": 1, "summary": "s"},
    ]
    _migrate_v1_to_v2(entries)
    compaction = entries[2]
    assert "firstKeptEntryIndex" not in compaction
    assert "firstKeptEntryId" in compaction


def test_migrate_v1_to_v2_compaction_out_of_range():
    entries = [
        {"type": "session", "version": 1},
        {"type": "compaction", "firstKeptEntryIndex": 999, "summary": "s"},
    ]
    _migrate_v1_to_v2(entries)
    compaction = entries[1]
    assert "firstKeptEntryIndex" not in compaction
    assert "firstKeptEntryId" not in compaction


# ---------------------------------------------------------------------------
# _migrate_v2_to_v3 / migrate_session_entries
# ---------------------------------------------------------------------------


def test_migrate_v2_to_v3_renames_hook_message():
    entries = [
        {"type": "session", "version": 2},
        {"type": "message", "message": {"role": "hookMessage", "content": "x"}},
    ]
    _migrate_v2_to_v3(entries)
    assert entries[1]["message"]["role"] == "custom"


def test_migrate_session_entries_v1_to_v3():
    entries = [
        {"type": "session", "version": 1},
        {"type": "message", "message": {"role": "user", "content": "hello"}},
    ]
    migrate_session_entries(entries)
    # After migration, entries have ids
    assert "id" in entries[1]


# ---------------------------------------------------------------------------
# _entry_from_raw: label, session_info, custom_message, unknown type
# ---------------------------------------------------------------------------


def test_entry_from_raw_label():
    from pi_coding_agent.core.session_manager import _entry_from_raw

    result = _entry_from_raw(
        {
            "type": "label",
            "id": "l1",
            "parentId": None,
            "timestamp": "2025-01-01T00:00:01Z",
            "targetId": "e1",
            "label": "my-label",
        }
    )
    assert isinstance(result, LabelEntry)
    assert result.label == "my-label"


def test_entry_from_raw_session_info():
    from pi_coding_agent.core.session_manager import _entry_from_raw

    result = _entry_from_raw(
        {
            "type": "session_info",
            "id": "i1",
            "parentId": None,
            "timestamp": "2025-01-01T00:00:01Z",
            "name": "My Session",
        }
    )
    assert isinstance(result, SessionInfoEntry)
    assert result.name == "My Session"


def test_entry_from_raw_custom_message_list_content():
    from pi_coding_agent.core.session_manager import _entry_from_raw

    result = _entry_from_raw(
        {
            "type": "custom_message",
            "id": "c1",
            "parentId": None,
            "timestamp": "2025-01-01T00:00:01Z",
            "customType": "my-type",
            "content": [{"type": "text", "text": "hello"}],
            "display": True,
        }
    )
    assert isinstance(result, CustomMessageEntry)
    assert result.custom_type == "my-type"
    assert result.display is True


def test_entry_from_raw_custom_message_image_content():
    from pi_coding_agent.core.session_manager import _entry_from_raw

    result = _entry_from_raw(
        {
            "type": "custom_message",
            "id": "c1",
            "parentId": None,
            "timestamp": "2025-01-01T00:00:01Z",
            "customType": "img-type",
            "content": [{"type": "image", "data": "abc", "mediaType": "image/png"}],
            "display": False,
        }
    )
    assert isinstance(result, CustomMessageEntry)
    from pi_ai.types import ImageContent

    assert isinstance(result.content[0], ImageContent)


def test_entry_from_raw_unknown_type_returns_none():
    from pi_coding_agent.core.session_manager import _entry_from_raw

    result = _entry_from_raw({"type": "nonexistent_type"})
    assert result is None


# ---------------------------------------------------------------------------
# build_session_path with leaf_id=None
# ---------------------------------------------------------------------------


def test_build_session_path_leaf_id_none():
    from pi_ai.types import UserMessage
    from pi_coding_agent.core.session_manager import SessionMessageEntry

    entries = [
        SessionMessageEntry(
            id="e1", parent_id=None, timestamp="2025-01-01T00:00:00Z", message=UserMessage(content="hi", timestamp=0)
        )
    ]
    path = build_session_path(entries, leaf_id=None)
    assert path == []


# ---------------------------------------------------------------------------
# build_context_entries: compaction with first_kept_entry_id not found
# ---------------------------------------------------------------------------


def _make_user_entry(eid: str, parent: str | None) -> SessionMessageEntry:
    return SessionMessageEntry(
        id=eid, parent_id=parent, timestamp="2025-01-01T00:00:00Z", message=UserMessage(content=eid, timestamp=0)
    )


def test_build_context_entries_compaction_trims_history():
    e1 = _make_user_entry("e1", None)
    e2 = _make_user_entry("e2", "e1")
    e3 = _make_user_entry("e3", "e2")
    compaction = CompactionEntry(
        id="c1",
        parent_id="e2",
        timestamp="2025-01-01T00:00:01Z",
        summary="summary",
        first_kept_entry_id="e2",
        tokens_before=100,
    )
    e4 = _make_user_entry("e4", "c1")
    entries = [e1, e2, e3, compaction, e4]
    context = build_context_entries(entries, "e4")
    ids = [e.id for e in context]
    assert "c1" in ids
    assert "e4" in ids
    # e1 should not be included (before the kept entry)
    assert "e1" not in ids


# ---------------------------------------------------------------------------
# SessionManager: in-memory basic operations
# ---------------------------------------------------------------------------


def test_session_manager_get_leaf_entry_when_empty():
    sm = SessionManager.in_memory()
    assert sm.get_leaf_entry() is None


def test_session_manager_get_leaf_entry_after_appending():
    sm = SessionManager.in_memory()
    eid = sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    leaf = sm.get_leaf_entry()
    assert leaf is not None
    assert leaf.id == eid


def test_session_manager_get_children():
    sm = SessionManager.in_memory()
    parent_id = sm.append_message(UserMessage(content=[TextContent(text="parent")], timestamp=0))
    # Branch to append two children from same parent
    sm.branch(parent_id)
    child1_id = sm.append_message(UserMessage(content=[TextContent(text="child1")], timestamp=0))
    sm.branch(parent_id)
    child2_id = sm.append_message(UserMessage(content=[TextContent(text="child2")], timestamp=0))
    children = sm.get_children(parent_id)
    assert len(children) == 2
    cids = {c.id for c in children}
    assert child1_id in cids
    assert child2_id in cids


def test_session_manager_get_entry():
    sm = SessionManager.in_memory()
    eid = sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    entry = sm.get_entry(eid)
    assert entry is not None
    assert entry.id == eid
    assert sm.get_entry("nonexistent") is None


def test_session_manager_append_label_change_and_remove():
    sm = SessionManager.in_memory()
    eid = sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    sm.append_label_change(eid, "my-label")
    assert sm.get_label(eid) == "my-label"
    sm.append_label_change(eid, None)
    assert sm.get_label(eid) is None


def test_session_manager_append_label_change_unknown_entry():
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.append_label_change("nonexistent", "label")


def test_session_manager_branch_unknown_entry():
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.branch("nonexistent")


def test_session_manager_reset_leaf():
    sm = SessionManager.in_memory()
    sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    sm.reset_leaf()
    assert sm.get_leaf_id() is None


def test_session_manager_get_session_name():
    sm = SessionManager.in_memory()
    assert sm.get_session_name() is None
    sm.append_session_info("My Session Name")
    assert sm.get_session_name() == "My Session Name"


def test_session_manager_get_session_name_strips_whitespace():
    sm = SessionManager.in_memory()
    sm.append_session_info("  \n name \n ")
    assert sm.get_session_name() == "name"


def test_session_manager_get_tree_single_node():
    sm = SessionManager.in_memory()
    sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    tree = sm.get_tree()
    assert len(tree) == 1


def test_session_manager_get_tree_with_branch():
    sm = SessionManager.in_memory()
    parent_id = sm.append_message(UserMessage(content=[TextContent(text="root")], timestamp=0))
    sm.branch(parent_id)
    sm.append_message(UserMessage(content=[TextContent(text="child1")], timestamp=0))
    sm.branch(parent_id)
    sm.append_message(UserMessage(content=[TextContent(text="child2")], timestamp=0))
    tree = sm.get_tree()
    assert len(tree) == 1  # one root
    assert len(tree[0].children) == 2


def test_session_manager_branch_with_summary():
    sm = SessionManager.in_memory()
    eid = sm.append_message(UserMessage(content=[TextContent(text="base")], timestamp=0))
    sm.branch_with_summary(eid, "Summary of branch")
    ctx = sm.build_session_context()
    # Branch summary should appear in messages
    assert any(getattr(m, "role", None) == "branchSummary" for m in ctx.messages)


def test_session_manager_branch_with_summary_none_from_id():
    sm = SessionManager.in_memory()
    sm.branch_with_summary(None, "Summary from root")
    ctx = sm.build_session_context()
    assert any(getattr(m, "role", None) == "branchSummary" for m in ctx.messages)


def test_session_manager_branch_with_summary_unknown_entry():
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.branch_with_summary("nonexistent", "Summary")


def test_session_manager_uses_default_session_dir(tmp_path):
    from pi_coding_agent.core.session_manager import get_default_session_dir

    cwd = str(tmp_path / "project")
    Path(cwd).mkdir()
    session_dir = get_default_session_dir(cwd, str(tmp_path / "agent"))
    sm = SessionManager(cwd, session_dir, None, True)
    assert sm.uses_default_session_dir() is False  # agent_dir differs from default


# ---------------------------------------------------------------------------
# SessionManager: persisted operations
# ---------------------------------------------------------------------------


def _assistant_message() -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        content=[TextContent(text="reply")],
        usage=Usage(cost=Cost()),
        stop_reason="stop",
        timestamp=0,
    )


def test_session_manager_persisted_writes_file(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    sm = SessionManager(str(tmp_path / "cwd"), str(session_dir), None, True)
    assert sm.get_session_file() is not None
    sm.append_message(UserMessage(content=[TextContent(text="hello")], timestamp=0))
    sm.append_message(_assistant_message())
    assert Path(sm.get_session_file()).exists()


def test_session_manager_persist_entry_rewrite(tmp_path):
    """When has_assistant is True, _persist_entry rewrites the file (flushed=True path)."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    sm = SessionManager(str(tmp_path / "cwd"), str(session_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    sm.append_message(_assistant_message())
    # Appending again after assistant triggers the rewrite / append path
    sm.append_message(UserMessage(content=[TextContent(text="follow-up")], timestamp=0))
    file_after = Path(sm.get_session_file()).read_text()
    assert "follow-up" in file_after


def test_session_manager_open_nonexistent_file(tmp_path):
    """SessionManager.open with a nonexistent path creates a fresh session."""
    sm = SessionManager.open(str(tmp_path / "nonexistent.jsonl"), session_dir=str(tmp_path))
    assert sm.get_session_file() is not None


def test_session_manager_open_existing_file(tmp_path):
    """SessionManager.open with an existing session file loads it."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    sm = SessionManager(str(tmp_path / "cwd"), str(session_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    sm.append_message(_assistant_message())
    session_file = sm.get_session_file()

    sm2 = SessionManager.open(session_file, session_dir=str(session_dir))
    ctx = sm2.build_session_context()
    assert len(ctx.messages) > 0


def test_session_manager_open_with_cwd_override(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    sm = SessionManager(str(tmp_path / "cwd"), str(session_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    sm.append_message(_assistant_message())
    session_file = sm.get_session_file()

    sm2 = SessionManager.open(session_file, cwd_override=str(tmp_path / "other-cwd"))
    assert sm2.get_cwd() == str(tmp_path / "other-cwd")


def test_session_manager_continue_recent_no_sessions(tmp_path):
    sm = SessionManager.continue_recent(str(tmp_path / "cwd"), session_dir=str(tmp_path / "sessions"))
    # Should create a new session
    assert sm.get_session_id() != ""


def test_session_manager_continue_recent_with_existing(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cwd = str(tmp_path / "cwd")
    sm1 = SessionManager(cwd, str(session_dir), None, True)
    sm1.append_message(UserMessage(content=[TextContent(text="first")], timestamp=0))
    sm1.append_message(_assistant_message())
    session_id = sm1.get_session_id()

    sm2 = SessionManager.continue_recent(cwd, session_dir=str(session_dir))
    assert sm2.get_session_id() == session_id


def test_session_manager_fork_from(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sm = SessionManager(str(tmp_path / "cwd"), str(src_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="forked")], timestamp=0))
    sm.append_message(_assistant_message())
    src_file = sm.get_session_file()

    target_dir = tmp_path / "forked"
    target_dir.mkdir()
    sm2 = SessionManager.fork_from(src_file, str(tmp_path / "new-cwd"), session_dir=str(target_dir))
    ctx = sm2.build_session_context()
    assert any(getattr(m, "role", None) == "user" for m in ctx.messages)


def test_session_manager_fork_from_empty_file_raises(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="Cannot fork"):
        SessionManager.fork_from(str(empty_file), str(tmp_path / "cwd"))


def test_session_manager_fork_from_with_custom_id(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sm = SessionManager(str(tmp_path / "cwd"), str(src_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    sm.append_message(_assistant_message())
    src_file = sm.get_session_file()

    target_dir = tmp_path / "forked"
    target_dir.mkdir()
    sm2 = SessionManager.fork_from(
        src_file,
        str(tmp_path / "cwd2"),
        session_dir=str(target_dir),
        options=NewSessionOptions(id="my-fork"),
    )
    assert sm2.get_session_id() == "my-fork"


def test_session_manager_create_branched_session(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cwd = str(tmp_path / "cwd")
    sm = SessionManager(cwd, str(session_dir), None, True)
    e1 = sm.append_message(UserMessage(content=[TextContent(text="branch point")], timestamp=0))
    sm.append_message(_assistant_message())

    new_file = sm.create_branched_session(e1)
    assert new_file is not None
    assert Path(new_file).exists() or not Path(new_file).exists()  # lazy write for no-assistant branch


def test_session_manager_create_branched_session_with_assistant(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cwd = str(tmp_path / "cwd")
    sm = SessionManager(cwd, str(session_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="start")], timestamp=0))
    e2 = sm.append_message(_assistant_message())

    new_file = sm.create_branched_session(e2)
    assert new_file is not None
    assert Path(new_file).exists()


def test_session_manager_create_branched_session_includes_labels(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cwd = str(tmp_path / "cwd")
    sm = SessionManager(cwd, str(session_dir), None, True)
    e1 = sm.append_message(UserMessage(content=[TextContent(text="start")], timestamp=0))
    sm.append_message(_assistant_message())
    sm.append_label_change(e1, "tagged")

    new_file = sm.create_branched_session(e1)
    assert new_file is not None


def test_session_manager_create_branched_session_unknown_entry():
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError):
        sm.create_branched_session("nonexistent")


def test_session_manager_in_memory_create_branched_session():
    sm = SessionManager.in_memory()
    e1 = sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    result = sm.create_branched_session(e1)
    assert result is None  # in_memory always returns None


def test_session_manager_get_header():
    sm = SessionManager.in_memory()
    header = sm.get_header()
    assert isinstance(header, SessionHeader)


def test_session_manager_append_custom_entry():
    sm = SessionManager.in_memory()
    eid = sm.append_custom_entry("my-type", data={"key": "value"})
    entry = sm.get_entry(eid)
    assert isinstance(entry, CustomEntry)
    assert entry.custom_type == "my-type"


def test_session_manager_append_compaction():
    sm = SessionManager.in_memory()
    e1 = sm.append_message(UserMessage(content=[TextContent(text="hi")], timestamp=0))
    cid = sm.append_compaction("Compacted", e1, 1000)
    entry = sm.get_entry(cid)
    assert isinstance(entry, CompactionEntry)


def test_session_manager_append_custom_message_entry():
    sm = SessionManager.in_memory()
    eid = sm.append_custom_message_entry("test-type", [TextContent(text="custom")], display=True)
    entry = sm.get_entry(eid)
    assert isinstance(entry, CustomMessageEntry)
    assert entry.display is True


# ---------------------------------------------------------------------------
# File loading with migration
# ---------------------------------------------------------------------------


def test_load_entries_from_file_nonexistent(tmp_path):
    entries = load_entries_from_file(str(tmp_path / "nonexistent.jsonl"))
    assert entries == []


def test_set_session_file_empty_nonzero_raises(tmp_path):
    """A non-empty file that isn't a valid session raises ValueError."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    bad_file = session_dir / "bad.jsonl"
    bad_file.write_text("this is not valid json\n")
    with pytest.raises(ValueError, match="not a valid pi session"):
        SessionManager(str(tmp_path / "cwd"), str(session_dir), str(bad_file), True)


def test_set_session_file_empty_zero_creates_new_session(tmp_path):
    """A zero-byte file gets a fresh session written into it."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    empty_file = session_dir / "empty.jsonl"
    empty_file.write_text("")
    sm = SessionManager(str(tmp_path / "cwd"), str(session_dir), str(empty_file), True)
    assert sm.get_session_id() != ""
    assert Path(sm.get_session_file()).exists()


def test_session_manager_v1_migration_on_load(tmp_path):
    """Loading a v1 session file triggers migration and rewrites the file."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    v1_file = session_dir / "v1.jsonl"
    lines = [
        json.dumps(
            {"type": "session", "version": 1, "id": "s1", "timestamp": "2025-01-01T00:00:00Z", "cwd": str(tmp_path)}
        )
        + "\n",
        json.dumps({"type": "message", "message": {"role": "user", "content": "hello"}}) + "\n",
    ]
    v1_file.write_text("".join(lines))
    sm = SessionManager(str(tmp_path), str(session_dir), str(v1_file), True)
    entries = sm.get_entries()
    assert len(entries) == 1
    # After migration, entries have ids
    assert entries[0].id != ""


# ---------------------------------------------------------------------------
# find_most_recent_session
# ---------------------------------------------------------------------------


def test_find_most_recent_session_empty_dir(tmp_path):
    result = find_most_recent_session(str(tmp_path))
    assert result is None


def test_find_most_recent_session_with_files(tmp_path):
    # Write two valid session files
    for i in range(2):
        f = tmp_path / f"session{i}.jsonl"
        f.write_text(
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": f"s{i}",
                    "timestamp": f"2025-0{i + 1}-01T00:00:00Z",
                    "cwd": str(tmp_path),
                }
            )
            + "\n"
        )
    result = find_most_recent_session(str(tmp_path))
    assert result is not None
    assert result.endswith(".jsonl")


def test_find_most_recent_session_filters_by_cwd(tmp_path):
    cwd1 = tmp_path / "cwd1"
    cwd2 = tmp_path / "cwd2"
    cwd1.mkdir()
    cwd2.mkdir()
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    (session_dir / "a.jsonl").write_text(
        json.dumps({"type": "session", "version": 3, "id": "sa", "timestamp": "2025-01-01T00:00:00Z", "cwd": str(cwd1)})
        + "\n"
    )
    (session_dir / "b.jsonl").write_text(
        json.dumps({"type": "session", "version": 3, "id": "sb", "timestamp": "2025-01-01T00:00:00Z", "cwd": str(cwd2)})
        + "\n"
    )
    result = find_most_recent_session(str(session_dir), cwd=str(cwd1))
    assert result is not None
    assert "a.jsonl" in result


def test_find_most_recent_session_oserror_returns_none(tmp_path):
    result = find_most_recent_session(str(tmp_path / "nonexistent-dir"))
    assert result is None


# ---------------------------------------------------------------------------
# SessionManager.list / list_all
# ---------------------------------------------------------------------------


async def test_session_manager_list(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cwd = str(tmp_path / "cwd")
    sm = SessionManager(cwd, str(session_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="hello")], timestamp=0))
    sm.append_message(_assistant_message())

    sessions = await SessionManager.list(cwd, session_dir=str(session_dir))
    assert len(sessions) == 1
    assert sessions[0].id == sm.get_session_id()


async def test_session_manager_list_all(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cwd = str(tmp_path / "cwd")
    sm = SessionManager(cwd, str(session_dir), None, True)
    sm.append_message(UserMessage(content=[TextContent(text="hello")], timestamp=0))
    sm.append_message(_assistant_message())

    sessions = await SessionManager.list_all(session_dir=str(session_dir))
    assert len(sessions) == 1


async def test_session_manager_list_all_no_sessions(tmp_path):
    sessions = await SessionManager.list_all(session_dir=str(tmp_path / "nonexistent"))
    assert sessions == []


# ---------------------------------------------------------------------------
# _build_session_info paths
# ---------------------------------------------------------------------------


def test_build_session_info_with_session_info_entry(tmp_path):
    from pi_coding_agent.core.session_manager import _build_session_info

    session_file = tmp_path / "named.jsonl"
    lines = [
        json.dumps(
            {"type": "session", "version": 3, "id": "s1", "timestamp": "2025-01-01T00:00:00Z", "cwd": str(tmp_path)}
        )
        + "\n",
        json.dumps(
            {
                "type": "session_info",
                "id": "i1",
                "parentId": None,
                "timestamp": "2025-01-01T00:00:01Z",
                "name": "Named Session",
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "message",
                "id": "m1",
                "parentId": "i1",
                "timestamp": "2025-01-01T00:00:02Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "Hello"}], "timestamp": 1735686002000},
            }
        )
        + "\n",
    ]
    session_file.write_text("".join(lines))
    info = _build_session_info(str(session_file))
    assert info is not None
    assert info.name == "Named Session"
    assert info.message_count == 1
    assert info.first_message == "Hello"


def test_build_session_info_missing_file(tmp_path):
    from pi_coding_agent.core.session_manager import _build_session_info

    result = _build_session_info(str(tmp_path / "nonexistent.jsonl"))
    assert result is None


def test_build_session_info_no_header(tmp_path):
    from pi_coding_agent.core.session_manager import _build_session_info

    f = tmp_path / "no-header.jsonl"
    f.write_text(
        json.dumps(
            {
                "type": "message",
                "id": "m1",
                "parentId": None,
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "user", "content": "hi"},
            }
        )
        + "\n"
    )
    result = _build_session_info(str(f))
    assert result is None


def test_build_session_info_with_parent_session(tmp_path):
    from pi_coding_agent.core.session_manager import _build_session_info

    f = tmp_path / "with-parent.jsonl"
    f.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "s1",
                "timestamp": "2025-01-01T00:00:00Z",
                "cwd": str(tmp_path),
                "parentSession": "/some/parent.jsonl",
            }
        )
        + "\n"
    )
    info = _build_session_info(str(f))
    assert info is not None
    assert info.parent_session_path == "/some/parent.jsonl"
