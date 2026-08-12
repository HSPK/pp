"""Python port of `packages/coding-agent/test/suite/regressions/7497-session-discovery-symlink.test.ts`.

Regression #7497: `SessionManager.listAll()` walks `<agentDir>/sessions/` one
level deep. Each child there is a per-cwd bucket directory, and a bucket may be
a *symlink* to a directory somewhere else. The bug was that symlinked buckets
were skipped, so sessions stored behind them disappeared from `/resume`.

Three cases are pinned: a working directory link (whose sessions must be found
*and* reported under the alias path, not the resolved target), a broken
directory link (must be ignored without hiding the sibling buckets), and a link
to a plain file (likewise ignored).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from pi_coding_agent.core.config import ENV_AGENT_DIR
from pi_coding_agent.core.session_manager import SessionManager


@pytest.fixture
def session_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    temp_dir = Path(tempfile.mkdtemp(prefix="pi-session-discovery-"))
    agent_dir = temp_dir / "agent"
    sessions_dir = agent_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))
    try:
        yield temp_dir, sessions_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_session(temp_dir: Path, directory: Path, session_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": "2026-08-03T00:00:00.000Z",
        "cwd": str(temp_dir / "project"),
    }
    (directory / f"{session_id}.jsonl").write_text(json.dumps(header) + "\n", encoding="utf-8")


def test_discovers_a_session_through_a_directory_link_and_preserves_the_alias_path(
    session_root: tuple[Path, Path],
) -> None:
    temp_dir, sessions_dir = session_root
    target_dir = temp_dir / "linked-sessions"
    _write_session(temp_dir, target_dir, "linked")
    alias_dir = sessions_dir / "--linked--"
    os.symlink(target_dir, alias_dir, target_is_directory=True)

    sessions = asyncio.run(SessionManager.list_all())

    assert [session.id for session in sessions] == ["linked"]
    assert sessions[0].path == str(alias_dir / "linked.jsonl")


def test_ignores_a_broken_directory_link_without_hiding_valid_sessions(
    session_root: tuple[Path, Path],
) -> None:
    temp_dir, sessions_dir = session_root
    _write_session(temp_dir, sessions_dir / "--regular--", "regular")
    target_dir = temp_dir / "removed-sessions"
    target_dir.mkdir()
    os.symlink(target_dir, sessions_dir / "--broken--", target_is_directory=True)
    shutil.rmtree(target_dir)

    sessions = asyncio.run(SessionManager.list_all())

    assert [session.id for session in sessions] == ["regular"]


def test_ignores_links_to_files(session_root: tuple[Path, Path]) -> None:
    temp_dir, sessions_dir = session_root
    _write_session(temp_dir, sessions_dir / "--regular--", "regular")
    target_file = temp_dir / "not-a-directory"
    target_file.write_text("", encoding="utf-8")
    os.symlink(target_file, sessions_dir / "--file--")

    sessions = asyncio.run(SessionManager.list_all())

    assert [session.id for session in sessions] == ["regular"]
