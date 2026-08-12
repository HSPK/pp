"""Python port of `packages/coding-agent/test/session-id-readonly.test.ts`."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from _cli_process import CliResult, run_cli


def _has_session_with_id(root: Path, session_id: str) -> bool:
    if not root.exists():
        return False
    for dir_path, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            try:
                with open(Path(dir_path) / name, encoding="utf-8") as handle:
                    header = json.loads(handle.readline())
            except (OSError, ValueError):
                continue
            if header.get("type") == "session" and header.get("id") == session_id:
                return True
    return False


def _write_session(session_dir: Path, cwd: Path, session_id: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cwd": str(cwd),
    }
    (session_dir / f"{session_id}.jsonl").write_text(f"{json.dumps(header)}\n")


class _Dirs:
    def __init__(self, root: Path) -> None:
        self.agent_dir = root / "agent"
        self.project_dir = root / "project"
        self.session_dir = root / "sessions"
        self.agent_dir.mkdir(parents=True)
        self.project_dir.mkdir(parents=True)


def _run(
    tmp_path: Path,
    args: list[str] | Callable[[_Dirs], list[str]],
    setup: Callable[[_Dirs], None] | None = None,
) -> tuple[CliResult, _Dirs]:
    dirs = _Dirs(tmp_path)
    if setup is not None:
        setup(dirs)
    resolved = args(dirs) if callable(args) else args
    return run_cli(resolved, str(dirs.project_dir), str(dirs.agent_dir)), dirs


def test_does_not_reserve_a_session_for_help(tmp_path: Path) -> None:
    result, dirs = _run(tmp_path, ["--session-id", "read-only-help", "--help"])

    assert result.code == 0
    assert _has_session_with_id(dirs.agent_dir / "sessions", "read-only-help") is False


def test_allows_no_session_with_session_id(tmp_path: Path) -> None:
    result, dirs = _run(tmp_path, ["--no-session", "--session-id", "ephemeral-id", "--help"])

    assert result.code == 0
    assert _has_session_with_id(dirs.agent_dir / "sessions", "ephemeral-id") is False


def test_does_not_reserve_a_session_for_list_models(tmp_path: Path) -> None:
    result, dirs = _run(tmp_path, ["--session-id", "read-only-models", "--list-models"])

    assert result.code == 0
    assert _has_session_with_id(dirs.agent_dir / "sessions", "read-only-models") is False


def test_warns_when_a_missing_session_id_creates_a_new_session(tmp_path: Path) -> None:
    result, _dirs = _run(
        tmp_path,
        lambda dirs: [
            "--session-dir",
            str(dirs.session_dir),
            "--session-id",
            "missing-session-id",
            "--model",
            "missing-model",
            "-p",
            "hi",
        ],
    )

    assert result.code == 1
    assert (
        "Warning: No project session found with id 'missing-session-id'; creating a new session with that id."
        in result.stderr
    )


def test_does_not_warn_when_session_id_opens_an_existing_session(tmp_path: Path) -> None:
    result, _dirs = _run(
        tmp_path,
        lambda dirs: [
            "--session-dir",
            str(dirs.session_dir),
            "--session-id",
            "existing-session-id",
            "--model",
            "missing-model",
            "-p",
            "hi",
        ],
        lambda dirs: _write_session(dirs.session_dir, dirs.project_dir, "existing-session-id"),
    )

    assert result.code == 1
    assert "No project session found with id 'existing-session-id'" not in result.stderr


def test_rejects_an_existing_fork_target_session_id(tmp_path: Path) -> None:
    def setup(dirs: _Dirs) -> None:
        _write_session(dirs.session_dir, dirs.project_dir, "source-id")
        _write_session(dirs.session_dir, dirs.project_dir, "existing-id")

    result, _dirs = _run(
        tmp_path,
        lambda dirs: [
            "--session-dir",
            str(dirs.session_dir),
            "--fork",
            "source-id",
            "--session-id",
            "existing-id",
            "-p",
            "hi",
        ],
        setup,
    )

    assert result.code == 1
    assert "Session already exists with id 'existing-id'" in result.stderr


def test_rejects_ids_invalid_under_session_manager_rules_without_stack_traces(tmp_path: Path) -> None:
    for index, session_id in enumerate(["-bad", "bad id"]):
        root = tmp_path / f"case-{index}"
        root.mkdir()
        result, _dirs = _run(root, ["--session-id", session_id, "-p", "hi"])

        assert result.code == 1
        assert "Session id must be non-empty" in result.stderr
        assert "SessionManager.create" not in result.stderr
        assert "Traceback" not in result.stderr
