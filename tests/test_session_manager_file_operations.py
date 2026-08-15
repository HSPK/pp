"""Python port of `packages/coding-agent/test/session-manager/file-operations.test.ts`."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from pi_ai.types import AssistantMessage, Cost, TextContent, Usage, UserMessage, now_ms

from pi_coding_agent.core.session_manager import (
    SessionManager,
    find_most_recent_session,
    load_entries_from_file,
)

HEADER_SCAN_LIMIT_BYTES = 1024 * 1024

_VALID_HEADER = '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
_VALID_MESSAGE = (
    '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z",'
    '"message":{"role":"user","content":"hi","timestamp":1}}\n'
)


def _order_by_mtime(*files: Path) -> None:
    """Give ``files`` strictly increasing mtimes.

    The TypeScript test inserts a 10ms `setTimeout` between writes so the two
    files get different mtimes. Stamping the mtimes directly is deterministic
    and does not sleep.
    """
    base = 1_700_000_000.0
    for index, file in enumerate(files):
        os.utime(file, (base + index, base + index))


def _write_session_header(file: Path, cwd: str, session_id: str, prefix: str = "") -> None:
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "cwd": cwd,
    }
    file.write_text(f"{prefix}{json.dumps(header)}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# load_entries_from_file
# ---------------------------------------------------------------------------


def test_returns_empty_list_for_non_existent_file(tmp_path: Path):
    assert load_entries_from_file(tmp_path / "nonexistent.jsonl") == []


def test_returns_empty_list_for_empty_file(tmp_path: Path):
    file = tmp_path / "empty.jsonl"
    file.write_text("", encoding="utf-8")
    assert load_entries_from_file(file) == []


def test_returns_empty_list_for_file_without_valid_session_header(tmp_path: Path):
    file = tmp_path / "no-header.jsonl"
    file.write_text('{"type":"message","id":"1"}\n', encoding="utf-8")
    assert load_entries_from_file(file) == []


def test_returns_empty_list_for_malformed_json(tmp_path: Path):
    file = tmp_path / "malformed.jsonl"
    file.write_text("not json\n", encoding="utf-8")
    assert load_entries_from_file(file) == []


def test_loads_valid_session_file(tmp_path: Path):
    file = tmp_path / "valid.jsonl"
    file.write_text(_VALID_HEADER + _VALID_MESSAGE, encoding="utf-8")

    entries = load_entries_from_file(file)

    assert len(entries) == 2
    assert entries[0].type == "session"
    assert entries[1].type == "message"


def test_skips_malformed_lines_but_keeps_valid_ones(tmp_path: Path):
    file = tmp_path / "mixed.jsonl"
    file.write_text(_VALID_HEADER + "not valid json\n" + _VALID_MESSAGE, encoding="utf-8")

    assert len(load_entries_from_file(file)) == 2


@pytest.mark.parametrize(
    ("prefix", "session_id"),
    [
        pytest.param("\n  \n", "leading-blank", id="leading blank lines"),
        pytest.param("not json\n{broken json\n", "leading-malformed", id="leading malformed lines"),
        pytest.param("", "a" * 8192, id="a multi-buffer header"),
    ],
)
def test_reads_cwd_from_a_session_with_awkward_leading_content(tmp_path: Path, prefix: str, session_id: str):
    file = tmp_path / "header.jsonl"
    stored_cwd = str(tmp_path / "stored-project")
    _write_session_header(file, stored_cwd, session_id, prefix)

    session_manager = SessionManager.open(str(file), str(tmp_path))

    assert session_manager.get_session_id() == session_id
    assert session_manager.get_cwd() == stored_cwd


def test_opens_compatible_sessions_beyond_the_discovery_scan_limit(tmp_path: Path):
    stored_cwd = str(tmp_path / "stored-project")
    override_cwd = str(tmp_path / "override-project")
    cases = [
        ("large-header", "a" * (HEADER_SCAN_LIMIT_BYTES + 1), ""),
        ("large-prefix", "large-prefix", "x" * (HEADER_SCAN_LIMIT_BYTES + 1) + "\n"),
    ]

    for name, session_id, prefix in cases:
        file = tmp_path / f"{name}.jsonl"
        _write_session_header(file, stored_cwd, session_id, prefix)
        for cwd_override in (None, override_cwd):
            session_manager = SessionManager.open(str(file), str(tmp_path), cwd_override)
            assert session_manager.get_session_id() == session_id
            assert session_manager.get_cwd() == (cwd_override or stored_cwd)


# The TypeScript "opens session files larger than Node's max string length"
# case writes a >512MB sparse file to cross `buffer.constants.MAX_STRING_LENGTH`.
# That limit is a V8 string-length cap with no Python equivalent -- `str` has no
# such bound -- and the case exists only to prove the chunked reader does not
# hit it. There is nothing to assert here.


# ---------------------------------------------------------------------------
# find_most_recent_session
# ---------------------------------------------------------------------------


def test_find_most_recent_returns_none_for_empty_directory(tmp_path: Path):
    assert find_most_recent_session(str(tmp_path)) is None


def test_find_most_recent_returns_none_for_non_existent_directory(tmp_path: Path):
    assert find_most_recent_session(str(tmp_path / "nonexistent")) is None


def test_find_most_recent_ignores_non_jsonl_files(tmp_path: Path):
    (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "file.json").write_text("{}", encoding="utf-8")

    assert find_most_recent_session(str(tmp_path)) is None


def test_find_most_recent_ignores_jsonl_files_without_valid_session_header(tmp_path: Path):
    (tmp_path / "invalid.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")

    assert find_most_recent_session(str(tmp_path)) is None


def test_find_most_recent_returns_single_valid_session_file(tmp_path: Path):
    file = tmp_path / "session.jsonl"
    file.write_text(_VALID_HEADER, encoding="utf-8")

    assert find_most_recent_session(str(tmp_path)) == str(file)


def test_find_most_recent_returns_most_recently_modified_session(tmp_path: Path):
    file1 = tmp_path / "older.jsonl"
    file2 = tmp_path / "newer.jsonl"

    file1.write_text('{"type":"session","id":"old","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n', "utf-8")
    file2.write_text('{"type":"session","id":"new","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n', "utf-8")
    _order_by_mtime(file1, file2)

    assert find_most_recent_session(str(tmp_path)) == str(file2)


def test_find_most_recent_skips_invalid_files_and_returns_valid_one(tmp_path: Path):
    invalid = tmp_path / "invalid.jsonl"
    valid = tmp_path / "valid.jsonl"

    invalid.write_text('{"type":"not-session"}\n', encoding="utf-8")
    valid.write_text(_VALID_HEADER, encoding="utf-8")
    _order_by_mtime(invalid, valid)

    assert find_most_recent_session(str(tmp_path)) == str(valid)


def test_find_most_recent_skips_oversized_corrupt_files_and_returns_a_valid_session(tmp_path: Path):
    (tmp_path / "oversized.jsonl").write_text("x" * (HEADER_SCAN_LIMIT_BYTES + 1), encoding="utf-8")
    valid = tmp_path / "valid.jsonl"
    valid.write_text(_VALID_HEADER, encoding="utf-8")

    assert find_most_recent_session(str(tmp_path)) == str(valid)


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits are not enforced for root")
def test_find_most_recent_skips_unreadable_file_and_returns_a_valid_session(tmp_path: Path):
    """Not a literal TS case, but the same contract the TS suite pins via
    `readSessionHeaderForDiscovery` (session-manager.ts): each candidate file is probed
    independently and any read error (permission denied, corrupt content, oversized data)
    must only exclude that one file, never abort discovery of the others. The existing
    "skips invalid/oversized" cases above happen to succeed even without per-file isolation
    because JSON/size failures are handled internally without raising; permission-denied is
    the case that actually exercises the isolation, since `Path.read_text()` raises `OSError`
    for it. This caught a real defect: `find_most_recent_session` used one `try/except OSError`
    around the whole directory loop, so one unreadable file discarded every candidate already
    found and returned `None` instead of the valid session."""
    unreadable = tmp_path / "unreadable.jsonl"
    valid = tmp_path / "valid.jsonl"
    unreadable.write_text(_VALID_HEADER, encoding="utf-8")
    valid.write_text(_VALID_HEADER, encoding="utf-8")
    os.chmod(unreadable, 0o000)
    try:
        assert find_most_recent_session(str(tmp_path)) == str(valid)
    finally:
        os.chmod(unreadable, 0o644)


def test_find_most_recent_filters_by_cwd(tmp_path: Path):
    project_a = str(tmp_path / "project-a")
    project_b = str(tmp_path / "project-b")
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"

    file_a.write_text(
        json.dumps({"type": "session", "id": "a", "timestamp": "2025-01-01T00:00:00Z", "cwd": project_a}) + "\n",
        encoding="utf-8",
    )
    file_b.write_text(
        json.dumps({"type": "session", "id": "b", "timestamp": "2025-01-01T00:00:00Z", "cwd": project_b}) + "\n",
        encoding="utf-8",
    )
    _order_by_mtime(file_a, file_b)

    assert find_most_recent_session(str(tmp_path), project_a) == str(file_a)
    assert find_most_recent_session(str(tmp_path), project_b) == str(file_b)


# ---------------------------------------------------------------------------
# Custom flat session directory
# ---------------------------------------------------------------------------


def _create_persisted_session(cwd: str, session_dir: str, label: str) -> str:
    session = SessionManager.create(cwd, session_dir)
    session.append_message(UserMessage(content=label, timestamp=now_ms()))
    session.append_message(
        AssistantMessage(
            api="anthropic-messages",
            provider="anthropic",
            model="test",
            content=[TextContent(text=f"reply to {label}")],
            usage=Usage(
                input=1,
                output=1,
                cache_read=0,
                cache_write=0,
                total_tokens=2,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            stop_reason="stop",
            timestamp=now_ms(),
        )
    )
    session_file = session.get_session_file()
    assert session_file is not None, "Expected persisted session file"
    return session_file


async def test_scopes_current_folder_apis_by_cwd_while_listing_all_flat_sessions(tmp_path: Path):
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    session_a = _create_persisted_session(str(project_a), str(tmp_path), "from A")
    session_b = _create_persisted_session(str(project_b), str(tmp_path), "from B")
    _order_by_mtime(Path(session_a), Path(session_b))

    current_a = await SessionManager.list(str(project_a), str(tmp_path))
    assert [session.path for session in current_a] == [session_a]

    all_sessions = await SessionManager.list_all(str(tmp_path))
    assert {session.path for session in all_sessions} == {session_a, session_b}

    continued_a = SessionManager.continue_recent(str(project_a), str(tmp_path))
    assert continued_a.get_session_file() == session_a


# ---------------------------------------------------------------------------
# SessionManager.open with corrupted files
# ---------------------------------------------------------------------------


def test_truncates_and_rewrites_empty_file_with_valid_header(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    manager = SessionManager.open(str(empty_file), str(tmp_path))

    assert manager.get_session_id()
    header = manager.get_header()
    assert header is not None
    assert header.type == "session"

    lines = [line for line in empty_file.read_text(encoding="utf-8").strip().split("\n") if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "session"
    assert json.loads(lines[0])["id"] == manager.get_session_id()


def test_raises_and_preserves_non_empty_file_without_valid_header(tmp_path: Path):
    no_header_file = tmp_path / "no-header.jsonl"
    original_content = (
        '{"type":"message","id":"abc","parentId":"orphaned","timestamp":"2025-01-01T00:00:00Z",'
        '"message":{"role":"assistant","content":"test"}}\n'
    )
    no_header_file.write_text(original_content, encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"Session file is not a valid pi session: {no_header_file}")):
        SessionManager.open(str(no_header_file), str(tmp_path))

    assert no_header_file.read_text(encoding="utf-8") == original_content


def test_raises_and_preserves_non_session_jsonl_files(tmp_path: Path):
    non_session_file = tmp_path / "not-a-session.log"
    original_content = '{"type":"event","data":"not a session"}\n'
    non_session_file.write_text(original_content, encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"Session file is not a valid pi session: {non_session_file}")):
        SessionManager.open(str(non_session_file), str(tmp_path))

    assert non_session_file.read_text(encoding="utf-8") == original_content


def test_preserves_explicit_session_file_path_when_recovering_from_corrupted_file(tmp_path: Path):
    explicit_path = tmp_path / "my-session.jsonl"
    explicit_path.write_text("", encoding="utf-8")

    manager = SessionManager.open(str(explicit_path), str(tmp_path))

    assert manager.get_session_file() == str(explicit_path)


def test_subsequent_loads_of_initialized_empty_file_work_correctly(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    first = SessionManager.open(str(empty_file), str(tmp_path))
    session_id = first.get_session_id()

    second = SessionManager.open(str(empty_file), str(tmp_path))
    assert second.get_session_id() == session_id
    header = second.get_header()
    assert header is not None
    assert header.type == "session"
