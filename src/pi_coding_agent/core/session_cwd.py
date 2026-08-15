"""Detect and report a session whose stored working directory no longer exists.

Python port of `packages/coding-agent/src/core/session-cwd.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SessionCwdSource(Protocol):
    def get_cwd(self) -> str: ...
    def get_session_file(self) -> str | None: ...


@dataclass
class SessionCwdIssue:
    sessionCwd: str
    fallbackCwd: str
    sessionFile: str | None = None


def get_missing_session_cwd_issue(session_manager: SessionCwdSource, fallback_cwd: str) -> SessionCwdIssue | None:
    session_file = session_manager.get_session_file()
    if not session_file:
        return None

    session_cwd = session_manager.get_cwd()
    if not session_cwd or Path(session_cwd).exists():
        return None

    return SessionCwdIssue(sessionCwd=session_cwd, fallbackCwd=fallback_cwd, sessionFile=session_file)


def format_missing_session_cwd_error(issue: SessionCwdIssue) -> str:
    session_file = f"\nSession file: {issue.sessionFile}" if issue.sessionFile else ""
    return (
        f"Stored session working directory does not exist: {issue.sessionCwd}{session_file}\n"
        f"Current working directory: {issue.fallbackCwd}"
    )


def format_missing_session_cwd_prompt(issue: SessionCwdIssue) -> str:
    return f"cwd from session file does not exist\n{issue.sessionCwd}\n\ncontinue in current cwd\n{issue.fallbackCwd}"


class MissingSessionCwdError(Exception):
    def __init__(self, issue: SessionCwdIssue) -> None:
        super().__init__(format_missing_session_cwd_error(issue))
        self.issue = issue


def assert_session_cwd_exists(session_manager: SessionCwdSource, fallback_cwd: str) -> None:
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue:
        raise MissingSessionCwdError(issue)
