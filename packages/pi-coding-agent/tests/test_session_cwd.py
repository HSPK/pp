"""Tests for `pi_coding_agent.core.session_cwd`, ported from
`packages/coding-agent/test/session-cwd.test.ts`.

All filesystem access goes through `tmp_path`; the real user home directory is
never touched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pi_coding_agent.core.agent_session_runtime import create_agent_session_runtime
from pi_coding_agent.core.session_cwd import (
    MissingSessionCwdError,
    SessionCwdIssue,
    assert_session_cwd_exists,
    format_missing_session_cwd_error,
    format_missing_session_cwd_prompt,
    get_missing_session_cwd_issue,
)
from pi_coding_agent.core.session_manager import SessionManager

TIMEOUT = 5.0


class FakeSource:
    """A minimal `SessionCwdSource`: the protocol is only two getters."""

    def __init__(self, cwd: str, session_file: str | None) -> None:
        self._cwd = cwd
        self._session_file = session_file

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_file(self) -> str | None:
        return self._session_file


def write_session_file(path: Path, cwd: str) -> None:
    header = {
        "type": "session",
        "version": 3,
        "id": "session-id",
        "timestamp": "2025-01-01T00:00:00.000Z",
        "cwd": cwd,
    }
    path.write_text(json.dumps(header) + "\n")


def test_detects_missing_session_cwd_from_persisted_sessions(tmp_path: Path):
    fallback_cwd = tmp_path / "fallback"
    fallback_cwd.mkdir()
    missing_cwd = fallback_cwd / "does-not-exist"
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    session_file = session_dir / "session.jsonl"
    write_session_file(session_file, str(missing_cwd))

    session_manager = SessionManager.open(str(session_file))
    issue = get_missing_session_cwd_issue(session_manager, str(fallback_cwd))

    assert issue == SessionCwdIssue(
        sessionCwd=str(missing_cwd),
        fallbackCwd=str(fallback_cwd),
        sessionFile=session_manager.get_session_file(),
    )


def test_supports_overriding_the_effective_cwd_when_opening_a_session(tmp_path: Path):
    fallback_cwd = tmp_path / "fallback"
    fallback_cwd.mkdir()
    missing_cwd = fallback_cwd / "does-not-exist"
    session_file = tmp_path / "session.jsonl"
    write_session_file(session_file, str(missing_cwd))

    session_manager = SessionManager.open(str(session_file), None, str(fallback_cwd))

    assert session_manager.get_cwd() == str(fallback_cwd)
    assert get_missing_session_cwd_issue(session_manager, str(fallback_cwd)) is None


def test_existing_session_cwd_produces_no_issue(tmp_path: Path):
    existing = tmp_path / "workspace"
    existing.mkdir()
    source = FakeSource(str(existing), str(tmp_path / "session.jsonl"))

    assert get_missing_session_cwd_issue(source, str(tmp_path)) is None
    assert_session_cwd_exists(source, str(tmp_path))


def test_no_session_file_means_no_issue(tmp_path: Path):
    source = FakeSource(str(tmp_path / "missing-dir"), None)

    assert get_missing_session_cwd_issue(source, str(tmp_path)) is None
    assert_session_cwd_exists(source, str(tmp_path))


def test_empty_session_cwd_means_no_issue(tmp_path: Path):
    source = FakeSource("", str(tmp_path / "session.jsonl"))

    assert get_missing_session_cwd_issue(source, str(tmp_path)) is None


def test_a_session_cwd_that_is_a_file_counts_as_existing(tmp_path: Path):
    # `Path.exists()` mirrors `existsSync`: any existing path, not just a directory.
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x")
    source = FakeSource(str(file_path), str(tmp_path / "session.jsonl"))

    assert get_missing_session_cwd_issue(source, str(tmp_path)) is None


def test_assert_session_cwd_exists_raises_with_the_formatted_message(tmp_path: Path):
    missing = tmp_path / "gone"
    session_file = tmp_path / "session.jsonl"
    source = FakeSource(str(missing), str(session_file))

    with pytest.raises(MissingSessionCwdError) as excinfo:
        assert_session_cwd_exists(source, str(tmp_path))

    error = excinfo.value
    assert error.issue == SessionCwdIssue(
        sessionCwd=str(missing), fallbackCwd=str(tmp_path), sessionFile=str(session_file)
    )
    assert str(error) == format_missing_session_cwd_error(error.issue)
    assert str(missing) in str(error)


def test_format_missing_session_cwd_error_includes_the_session_file():
    issue = SessionCwdIssue(sessionCwd="/old/cwd", fallbackCwd="/new/cwd", sessionFile="/sessions/s.jsonl")

    assert format_missing_session_cwd_error(issue) == (
        "Stored session working directory does not exist: /old/cwd\n"
        "Session file: /sessions/s.jsonl\n"
        "Current working directory: /new/cwd"
    )


def test_format_missing_session_cwd_error_without_a_session_file():
    issue = SessionCwdIssue(sessionCwd="/old/cwd", fallbackCwd="/new/cwd")

    assert format_missing_session_cwd_error(issue) == (
        "Stored session working directory does not exist: /old/cwd\nCurrent working directory: /new/cwd"
    )


def test_format_missing_session_cwd_prompt():
    issue = SessionCwdIssue(sessionCwd="/old/cwd", fallbackCwd="/new/cwd", sessionFile="/sessions/s.jsonl")

    assert format_missing_session_cwd_prompt(issue) == (
        "cwd from session file does not exist\n/old/cwd\n\ncontinue in current cwd\n/new/cwd"
    )


async def test_runtime_creation_fails_before_the_factory_runs_when_the_stored_cwd_is_missing(tmp_path: Path):
    fallback_cwd = tmp_path / "fallback"
    fallback_cwd.mkdir()
    missing_cwd = fallback_cwd / "does-not-exist"
    session_file = tmp_path / "session.jsonl"
    write_session_file(session_file, str(missing_cwd))

    session_manager = SessionManager.open(str(session_file))
    create_runtime_called = False

    async def create_runtime(**_kwargs):
        nonlocal create_runtime_called
        create_runtime_called = True
        raise AssertionError("should not be called")

    with pytest.raises(MissingSessionCwdError):
        await asyncio.wait_for(
            create_agent_session_runtime(
                create_runtime,
                cwd=str(fallback_cwd),
                agent_dir=str(fallback_cwd),
                session_manager=session_manager,
            ),
            timeout=TIMEOUT,
        )

    assert create_runtime_called is False
