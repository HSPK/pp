"""Python port of `packages/coding-agent/test/format-resume-command.test.ts`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_coding_agent.core.config import APP_NAME
from pi_coding_agent.modes.interactive.interactive_mode import format_resume_command


class _FakeSessionManager:
    def __init__(
        self,
        *,
        persisted: bool = True,
        session_file: str | None = None,
        session_id: str = "0197f6e4-4cf9-7f44-a2d8-f8f7f49ee9d3",
        session_dir: str = "/tmp/pi-sessions",
        uses_default_session_dir: bool = True,
    ) -> None:
        self._persisted = persisted
        self._session_file = session_file
        self._session_id = session_id
        self._session_dir = session_dir
        self._uses_default_session_dir = uses_default_session_dir

    def is_persisted(self) -> bool:
        return self._persisted

    def get_session_file(self) -> str | None:
        return self._session_file

    def get_session_id(self) -> str:
        return self._session_id

    def get_session_dir(self) -> str:
        return self._session_dir

    def uses_default_session_dir(self) -> bool:
        return self._uses_default_session_dir


@pytest.fixture
def set_stdout_isatty(monkeypatch: pytest.MonkeyPatch) -> Any:
    def apply(value: bool) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: value, raising=False)

    return apply


@pytest.fixture
def session_file(tmp_path: Path) -> str:
    file = tmp_path / "session.jsonl"
    file.write_text("\n", encoding="utf-8")
    return str(file)


def test_returns_session_resume_command_for_default_session_dirs(set_stdout_isatty: Any, session_file: str) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(session_file=session_file, session_id="test-session")

    assert format_resume_command(manager) == f"{APP_NAME} --session test-session"  # type: ignore[arg-type]


def test_includes_unquoted_safe_session_dirs_for_non_default_dirs(set_stdout_isatty: Any, session_file: str) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(
        session_file=session_file,
        session_id="test-session",
        session_dir="/tmp/custom-pi-sessions",
        uses_default_session_dir=False,
    )

    assert (
        format_resume_command(manager)  # type: ignore[arg-type]
        == f"{APP_NAME} --session-dir /tmp/custom-pi-sessions --session test-session"
    )


def test_quotes_session_dirs_containing_spaces(set_stdout_isatty: Any, session_file: str) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(
        session_file=session_file,
        session_id="test-session",
        session_dir="/tmp/custom pi sessions",
        uses_default_session_dir=False,
    )

    assert (
        format_resume_command(manager)  # type: ignore[arg-type]
        == f"{APP_NAME} --session-dir '/tmp/custom pi sessions' --session test-session"
    )


def test_quotes_session_dirs_containing_single_quotes(set_stdout_isatty: Any, session_file: str) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(
        session_file=session_file,
        session_id="test-session",
        session_dir="/tmp/custom pi's sessions",
        uses_default_session_dir=False,
    )

    assert (
        format_resume_command(manager)  # type: ignore[arg-type]
        == f"{APP_NAME} --session-dir '/tmp/custom pi'\\''s sessions' --session test-session"
    )


def test_returns_none_when_stdout_is_not_a_tty(set_stdout_isatty: Any, session_file: str) -> None:
    set_stdout_isatty(False)
    manager = _FakeSessionManager(session_file=session_file)

    assert format_resume_command(manager) is None  # type: ignore[arg-type]


def test_returns_none_for_in_memory_sessions(set_stdout_isatty: Any, session_file: str) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(persisted=False, session_file=session_file)

    assert format_resume_command(manager) is None  # type: ignore[arg-type]


def test_returns_none_when_session_file_is_missing(set_stdout_isatty: Any, tmp_path: Path) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(session_file=str(tmp_path / "pi-missing-session.jsonl"))

    assert format_resume_command(manager) is None  # type: ignore[arg-type]


def test_returns_none_when_session_file_is_not_set(set_stdout_isatty: Any) -> None:
    set_stdout_isatty(True)
    manager = _FakeSessionManager(session_file=None)

    assert format_resume_command(manager) is None  # type: ignore[arg-type]
