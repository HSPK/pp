"""Python port of `packages/coding-agent/test/startup-session-name.test.ts`."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from _cli_process import CliResult, run_cli


class _Dirs:
    def __init__(self, root: Path) -> None:
        self.agent_dir = root / "agent"
        self.project_dir = root / "project"
        self.session_file = root / "session.jsonl"
        self.agent_dir.mkdir(parents=True)
        self.project_dir.mkdir(parents=True)


def _create_session_file(dirs: _Dirs) -> None:
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    header = {
        "type": "session",
        "version": 3,
        "id": "existing-session",
        "timestamp": timestamp,
        "cwd": str(dirs.project_dir),
    }
    message = {
        "type": "message",
        "id": "assistant-1",
        "parentId": None,
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "timestamp": int(time.time() * 1000),
        },
    }
    dirs.session_file.write_text(f"{json.dumps(header)}\n{json.dumps(message)}\n")


def _read_session_info_names(session_file: Path) -> list[str]:
    entries = [json.loads(line) for line in session_file.read_text().strip().split("\n")]
    return [entry.get("name", "") for entry in entries if entry.get("type") == "session_info"]


def _setup(tmp_path: Path) -> _Dirs:
    dirs = _Dirs(tmp_path)
    _create_session_file(dirs)
    return dirs


def _run(dirs: _Dirs, args: list[str]) -> CliResult:
    return run_cli(args, str(dirs.project_dir), str(dirs.agent_dir))


def test_sets_name_on_the_selected_session_before_model_validation(tmp_path: Path) -> None:
    dirs = _setup(tmp_path)

    result = _run(
        dirs,
        [
            "--session",
            str(dirs.session_file),
            "--name",
            "  CLI Named Session  ",
            "--model",
            "missing-model",
            "-p",
            "hi",
        ],
    )

    assert result.code == 1
    assert _read_session_info_names(dirs.session_file) == ["CLI Named Session"]


def test_rejects_empty_name_values_without_appending_session_metadata(tmp_path: Path) -> None:
    dirs = _setup(tmp_path)

    result = _run(
        dirs,
        ["--session", str(dirs.session_file), "--name", "   ", "--model", "missing-model", "-p", "hi"],
    )

    assert result.code == 1
    assert "--name requires a non-empty value" in result.stderr
    assert _read_session_info_names(dirs.session_file) == []
