"""Turn ``--session``/``--fork``/``--continue``/``--resume`` into a `SessionManager`.

Ported from the session-resolution helpers in
``packages/coding-agent/src/main.ts``: ``resolveSessionPath``,
``findLocalSessionByExactId``, ``validateForkFlags``, ``validateSessionIdFlags``
and ``createSessionManager``.

A session argument is either a path (contains a separator or ends in
``.jsonl``) or a session-id prefix. Ids are matched exactly first, then by
prefix, in the current project first and then across all projects. A match in
another project can only be *forked* into the current one, never opened in
place, because a session records the cwd its file paths are relative to.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

from pi_coding_agent.cli.args import Args
from pi_coding_agent.core.session_manager import (
    NewSessionOptions,
    SessionManager,
    assert_valid_session_id,
)
from pi_coding_agent.utils.paths import resolve_path

ResolvedSessionType = Literal["path", "local", "global", "not_found"]


class SessionSelectionError(Exception):
    """A user-facing problem selecting a session; `main` prints it and exits 1."""


class SessionSelectionAborted(Exception):
    """The user declined a prompt; `main` exits 0."""


@dataclass
class ResolvedSession:
    type: ResolvedSessionType
    path: str | None = None
    cwd: str | None = None
    arg: str | None = None


def _looks_like_path(session_arg: str) -> bool:
    return "/" in session_arg or "\\" in session_arg or session_arg.endswith(".jsonl")


def _match_by_id(sessions: list, session_arg: str):
    for session in sessions:
        if session.id == session_arg:
            return session
    for session in sessions:
        if session.id.startswith(session_arg):
            return session
    return None


async def find_local_session_by_exact_id(
    session_id: str, cwd: str, session_dir: str | None = None
) -> ResolvedSession | None:
    local_sessions = await SessionManager.list(cwd, session_dir)
    for session in local_sessions:
        if session.id == session_id:
            return ResolvedSession(type="local", path=session.path)
    return None


async def resolve_session_path(session_arg: str, cwd: str, session_dir: str | None = None) -> ResolvedSession:
    if _looks_like_path(session_arg):
        return ResolvedSession(type="path", path=resolve_path(session_arg, cwd))

    local_match = _match_by_id(await SessionManager.list(cwd, session_dir), session_arg)
    if local_match is not None:
        return ResolvedSession(type="local", path=local_match.path)

    global_match = _match_by_id(await SessionManager.list_all(session_dir), session_arg)
    if global_match is not None:
        return ResolvedSession(type="global", path=global_match.path, cwd=global_match.cwd)

    return ResolvedSession(type="not_found", arg=session_arg)


def validate_fork_flags(parsed: Args) -> None:
    if not parsed.fork:
        return
    conflicting = [
        flag
        for flag, active in (
            ("--session", parsed.session),
            ("--continue", parsed.continue_),
            ("--resume", parsed.resume),
            ("--no-session", parsed.no_session),
        )
        if active
    ]
    if conflicting:
        raise SessionSelectionError(f"Error: --fork cannot be combined with {', '.join(conflicting)}")


def validate_session_id_flags(parsed: Args) -> None:
    if parsed.session_id is None:
        return
    conflicting = [
        flag
        for flag, active in (
            ("--session", parsed.session),
            ("--continue", parsed.continue_),
            ("--resume", parsed.resume),
        )
        if active
    ]
    if conflicting:
        raise SessionSelectionError(f"Error: --session-id cannot be combined with {', '.join(conflicting)}")
    try:
        assert_valid_session_id(parsed.session_id)
    except Exception as error:
        raise SessionSelectionError(f"Error: {error}") from error


def _open_or_raise(path: str, session_dir: str | None) -> SessionManager:
    try:
        return SessionManager.open(path, session_dir)
    except Exception as error:
        raise SessionSelectionError(f"Error: {error}") from error


def _fork_or_raise(source_path: str, cwd: str, session_dir: str | None, session_id: str | None) -> SessionManager:
    try:
        options = NewSessionOptions(id=session_id) if session_id else None
        return SessionManager.fork_from(source_path, cwd, session_dir, options)
    except Exception as error:
        raise SessionSelectionError(f"Error: {error}") from error


def _prompt_confirm(message: str) -> bool:
    try:
        answer = input(f"{message} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


async def create_session_manager(
    parsed: Args,
    cwd: str,
    session_dir: str | None = None,
    *,
    select_session=None,
    confirm=None,
) -> SessionManager:
    """Port of TS ``createSessionManager``.

    ``select_session`` is the interactive session picker used by ``--resume``;
    when it is not supplied, ``--resume`` falls back to the most recent session
    (the same target ``--continue`` would pick) rather than failing.
    """
    validate_fork_flags(parsed)
    validate_session_id_flags(parsed)

    new_session_options = NewSessionOptions(id=parsed.session_id) if parsed.session_id else None

    if parsed.no_session or parsed.help or parsed.list_models is not None:
        return SessionManager.in_memory(cwd, new_session_options)

    if parsed.fork:
        if parsed.session_id:
            existing = await find_local_session_by_exact_id(parsed.session_id, cwd, session_dir)
            if existing is not None:
                raise SessionSelectionError(f"Session already exists with id '{parsed.session_id}'")
        resolved = await resolve_session_path(parsed.fork, cwd, session_dir)
        if resolved.type == "not_found":
            raise SessionSelectionError(f"No session found matching '{resolved.arg}'")
        assert resolved.path is not None
        return _fork_or_raise(resolved.path, cwd, session_dir, parsed.session_id)

    if parsed.session:
        resolved = await resolve_session_path(parsed.session, cwd, session_dir)
        if resolved.type == "not_found":
            raise SessionSelectionError(f"No session found matching '{resolved.arg}'")
        assert resolved.path is not None
        if resolved.type == "global":
            print(f"Session found in different project: {resolved.cwd}")
            if not (confirm or _prompt_confirm)("Fork this session into current directory?"):
                raise SessionSelectionAborted("Aborted.")
            return _fork_or_raise(resolved.path, cwd, session_dir, None)
        return _open_or_raise(resolved.path, session_dir)

    if parsed.resume:
        if select_session is None:
            return SessionManager.continue_recent(cwd, session_dir)
        selected_path = await select_session(cwd, session_dir)
        if not selected_path:
            raise SessionSelectionAborted("No session selected")
        return _open_or_raise(selected_path, session_dir)

    if parsed.continue_:
        return SessionManager.continue_recent(cwd, session_dir)

    if parsed.session_id:
        existing = await find_local_session_by_exact_id(parsed.session_id, cwd, session_dir)
        if existing is not None:
            assert existing.path is not None
            return _open_or_raise(existing.path, session_dir)
        print(
            f"Warning: No project session found with id '{parsed.session_id}'; creating a new session with that id.",
            file=sys.stderr,
        )

    return SessionManager.create(cwd, session_dir, new_session_options)


__all__ = [
    "ResolvedSession",
    "SessionSelectionAborted",
    "SessionSelectionError",
    "create_session_manager",
    "find_local_session_by_exact_id",
    "resolve_session_path",
    "validate_fork_flags",
    "validate_session_id_flags",
]
