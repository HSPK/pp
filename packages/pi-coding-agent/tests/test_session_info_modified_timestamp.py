"""Python port of `packages/coding-agent/test/session-info-modified-timestamp.test.ts`."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pi_ai.types import AssistantMessage, Cost, TextContent, Usage
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.modes.interactive.theme.theme import init_theme


def _usage() -> Usage:
    return Usage(
        input=1,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=2,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _assistant(text: str, timestamp: int) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-completions",
        provider="openai",
        model="test",
        usage=_usage(),
        stop_reason="stop",
        timestamp=timestamp,
    )


def _create_session_file(path: Path) -> None:
    header = {
        "type": "session",
        "id": "test-session",
        "version": 3,
        "timestamp": "1970-01-01T00:00:00.000Z",
        "cwd": "/tmp",
    }
    path.write_text(f"{json.dumps(header)}\n", encoding="utf-8")

    # SessionManager only persists once it has seen at least one assistant message.
    manager = SessionManager.open(str(path))
    manager.append_message(_assistant("hi", int(time.time() * 1000)))


async def test_modified_uses_last_message_timestamp_instead_of_file_mtime(tmp_path: Path) -> None:
    init_theme("dark")
    file_path = tmp_path / "pi-session-modified.jsonl"
    _create_session_file(file_path)

    # The TypeScript test sleeps 10ms so the file mtime can differ from the message
    # timestamp on coarse filesystems. Stamping the mtime is deterministic and does not sleep.
    os.utime(file_path, (0, 0))
    before_mtime_ms = file_path.stat().st_mtime * 1000

    manager = SessionManager.open(str(file_path))
    message_time = int(time.time() * 1000)
    manager.append_message(_assistant("later", message_time))

    sessions = await SessionManager.list("/tmp", str(tmp_path))
    session = next((s for s in sessions if s.path == str(file_path)), None)
    assert session is not None
    assert int(session.modified.timestamp() * 1000) == message_time
    assert int(session.modified.timestamp() * 1000) != int(before_mtime_ms)
