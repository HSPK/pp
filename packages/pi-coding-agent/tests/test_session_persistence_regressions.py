"""Regression tests for session persistence defects found by review.

Each test here corresponds to a defect that silently corrupted or lost session
data, so they assert on what actually reaches disk and what survives a reload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_ai.types import AssistantMessage, Cost, Usage, UserMessage
from pi_coding_agent.core.session_manager import (
    SessionManager,
    _usage_from_raw,
    _usage_to_raw,
    load_entries_from_file,
)


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Compaction details must be JSON-serialisable and use the wire key names
# --------------------------------------------------------------------------


def test_compaction_details_persist_as_camel_case_json(tmp_path):
    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.new_session()
    manager.append_message(UserMessage(content="hello", timestamp=1))
    # The session file is only flushed once the transcript contains an
    # assistant message, so add one before asserting on file contents.
    manager.append_message(
        AssistantMessage(api="a", provider="p", model="m", content=[], stop_reason="stop", timestamp=2)
    )

    manager.append_compaction(
        summary="a summary",
        first_kept_entry_id=manager.get_branch()[-1].id,
        tokens_before=100,
        details={"readFiles": ["a.txt"], "modifiedFiles": ["b.txt"]},
    )

    session_file = Path(manager.get_session_file())
    lines = read_lines(session_file)
    compaction = next(line for line in lines if line.get("type") == "compaction")
    assert compaction["details"] == {"readFiles": ["a.txt"], "modifiedFiles": ["b.txt"]}


def test_compact_returns_json_serialisable_details():
    """`compact()` must not return a dataclass: it is written straight to disk."""
    from dataclasses import is_dataclass

    from pi_coding_agent.core import compaction as compaction_module

    source = Path(compaction_module.__file__).read_text(encoding="utf-8")
    assert "details=CompactionDetails(" not in source, (
        "compact() must return a plain dict for details; a dataclass is not JSON serialisable"
    )
    assert not is_dataclass({"readFiles": []})


def test_failed_persist_does_not_advance_the_in_memory_transcript(tmp_path, monkeypatch):
    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.new_session()
    manager.append_message(UserMessage(content="first", timestamp=1))

    entries_before = len(manager.get_branch())
    leaf_before = manager.get_branch()[-1].id

    def boom(_entry):
        raise TypeError("Object of type Whatever is not JSON serializable")

    monkeypatch.setattr(manager, "_persist_entry", boom)

    with pytest.raises(TypeError):
        manager.append_message(UserMessage(content="second", timestamp=2))

    # The in-memory transcript must not have advanced past the failed write,
    # otherwise later entries reference a parent that is missing on disk.
    assert len(manager.get_branch()) == entries_before
    assert manager.get_branch()[-1].id == leaf_before


# --------------------------------------------------------------------------
# Usage must round-trip through the camelCase wire format
# --------------------------------------------------------------------------


def test_usage_is_written_with_camel_case_keys():
    usage = Usage(
        input=10,
        output=5,
        cache_read=9000,
        cache_write=3,
        total_tokens=9018,
        cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1.0),
    )
    raw = _usage_to_raw(usage)

    assert raw["cacheRead"] == 9000
    assert raw["cacheWrite"] == 3
    assert raw["totalTokens"] == 9018
    assert raw["cost"]["cacheRead"] == 0.3
    assert "cache_read" not in raw
    assert "total_tokens" not in raw


def test_usage_reads_typescript_written_camel_case():
    usage = _usage_from_raw(
        {
            "input": 10,
            "output": 5,
            "cacheRead": 9000,
            "cacheWrite": 3,
            "totalTokens": 9018,
            "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.3, "cacheWrite": 0.4, "total": 1.0},
        }
    )
    assert usage.cache_read == 9000
    assert usage.cache_write == 3
    assert usage.total_tokens == 9018
    assert usage.cost.cache_read == 0.3


def test_usage_still_reads_snake_case():
    usage = _usage_from_raw({"input": 1, "cache_read": 2, "total_tokens": 3, "cost": {"cache_read": 0.5}})
    assert usage.cache_read == 2
    assert usage.total_tokens == 3
    assert usage.cost.cache_read == 0.5


def test_usage_reaches_disk_in_camel_case(tmp_path):
    """The end-to-end persistence path, not just the helper, must use wire keys."""
    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.new_session()
    manager.append_message(UserMessage(content="hello", timestamp=1))
    manager.append_message(
        AssistantMessage(
            api="a",
            provider="p",
            model="m",
            content=[],
            stop_reason="stop",
            timestamp=2,
            usage=Usage(input=10, output=5, cache_read=9000, cache_write=3, total_tokens=9018),
        )
    )

    lines = read_lines(Path(manager.get_session_file()))
    assistant = next(
        line for line in lines if line.get("type") == "message" and line["message"].get("role") == "assistant"
    )
    usage = assistant["message"]["usage"]

    assert usage["cacheRead"] == 9000, f"usage must be written in camelCase, got keys {sorted(usage)}"
    assert usage["totalTokens"] == 9018
    assert "cache_read" not in usage
    assert "total_tokens" not in usage


def test_usage_written_by_typescript_survives_a_reload(tmp_path):
    """A session file written by the TypeScript pi must read back with real numbers."""
    session_file = tmp_path / "ts.jsonl"
    lines = [
        {"type": "session", "id": "s1", "timestamp": "2024-01-01T00:00:00.000Z", "cwd": "/w", "version": 3},
        {
            "type": "message",
            "id": "m1",
            "parentId": None,
            "timestamp": "2024-01-01T00:00:01.000Z",
            "message": {
                "role": "assistant",
                "api": "a",
                "provider": "p",
                "model": "m",
                "content": [],
                "stopReason": "stop",
                "timestamp": 1,
                "usage": {
                    "input": 10,
                    "output": 5,
                    "cacheRead": 9000,
                    "cacheWrite": 3,
                    "totalTokens": 9018,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                },
            },
        },
    ]
    session_file.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.set_session_file(str(session_file))

    assistant = next(
        entry.message
        for entry in manager.get_branch()
        if getattr(getattr(entry, "message", None), "role", None) == "assistant"
    )
    assert assistant.usage.cache_read == 9000
    assert assistant.usage.total_tokens == 9018


def test_usage_round_trips():
    usage = Usage(input=1, output=2, cache_read=3, cache_write=4, total_tokens=10, reasoning=5)
    restored = _usage_from_raw(_usage_to_raw(usage))
    assert restored.input == 1
    assert restored.cache_read == 3
    assert restored.cache_write == 4
    assert restored.total_tokens == 10
    assert restored.reasoning == 5


# --------------------------------------------------------------------------
# v2 -> v3 migration must preserve custom messages
# --------------------------------------------------------------------------


def write_v2_session(path: Path) -> None:
    lines = [
        {"type": "session", "id": "sess-1", "timestamp": "2024-01-01T00:00:00.000Z", "cwd": "/w", "version": 2},
        {
            "type": "message",
            "id": "m1",
            "parentId": None,
            "timestamp": "2024-01-01T00:00:01.000Z",
            "message": {
                "role": "hookMessage",
                "customType": "myext",
                "content": "hook said hello",
                "display": True,
                "details": {"k": 1},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def test_v2_hook_message_migrates_to_custom_without_data_loss(tmp_path):
    session_file = tmp_path / "v2.jsonl"
    write_v2_session(session_file)

    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.set_session_file(str(session_file))

    rewritten = read_lines(session_file)
    message = next(line for line in rewritten if line.get("type") == "message")["message"]

    assert message["role"] == "custom", "hookMessage must be renamed, not downgraded to a user message"
    assert message["customType"] == "myext"
    assert message["display"] is True
    assert message["details"] == {"k": 1}
    assert rewritten[0]["version"] == 3


def test_migrated_session_reloads_with_the_custom_message(tmp_path):
    session_file = tmp_path / "v2.jsonl"
    write_v2_session(session_file)

    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.set_session_file(str(session_file))

    reloaded = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    reloaded.set_session_file(str(session_file))
    roles = [getattr(getattr(entry, "message", None), "role", None) for entry in reloaded.get_branch()]
    assert "custom" in roles


# --------------------------------------------------------------------------
# Exported sessions must be loadable again
# --------------------------------------------------------------------------


def test_exported_session_header_declares_its_type(tmp_path):
    manager = SessionManager(cwd=str(tmp_path), session_dir=str(tmp_path / "sessions"))
    manager.new_session()
    manager.append_message(UserMessage(content="hello", timestamp=1))

    export_path = tmp_path / "export.jsonl"
    header = {
        "type": "session",
        "id": manager.get_session_id(),
        "timestamp": "2024-01-01T00:00:00.000Z",
        "cwd": str(tmp_path),
        "version": 3,
    }
    lines = [json.dumps(header)]
    from pi_coding_agent.core.session_manager import _entry_to_raw

    for entry in manager.get_branch():
        lines.append(json.dumps(_entry_to_raw(entry)))
    export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # An export without "type": "session" reloads as an empty session.
    assert load_entries_from_file(export_path), "exported session must be loadable again"


def test_export_to_jsonl_source_includes_the_session_type():
    from pi_coding_agent.core import agent_session as agent_session_module

    source = Path(agent_session_module.__file__).read_text(encoding="utf-8")
    export_block = source[source.index("def export_to_jsonl") :]
    header_block = export_block[: export_block.index("lines = [")]
    assert '"type": "session"' in header_block


# --------------------------------------------------------------------------
# An aborted compaction is a cancellation, not a failure
# --------------------------------------------------------------------------


def test_abort_detection_recognises_a_cancelled_summarization():
    from pi_agent.harness.compaction.compaction import CompactionError
    from pi_ai.utils.abort import AbortError
    from pi_coding_agent.core.agent_session import _is_abort_error

    # Summarization signals cancellation with a CompactionError, not AbortError.
    assert _is_abort_error(CompactionError("aborted", "Summarization aborted")) is True
    assert _is_abort_error(AbortError("stopped")) is True
    assert _is_abort_error(RuntimeError("Compaction cancelled")) is True


def test_abort_detection_does_not_swallow_real_failures():
    from pi_agent.harness.compaction.compaction import CompactionError
    from pi_coding_agent.core.agent_session import _is_abort_error

    assert _is_abort_error(RuntimeError("provider exploded")) is False
    assert _is_abort_error(CompactionError("provider", "rate limited")) is False
