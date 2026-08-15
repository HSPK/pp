"""Strict JSONL framing for the stdio RPC mode.

Python port of `packages/coding-agent/src/modes/rpc/jsonl.ts`.

Framing is LF-only on purpose. Payload strings may legitimately contain other
Unicode separators such as U+2028 and U+2029, and a reader that also splits on
those would corrupt records; the TypeScript avoids Node's `readline` for exactly
this reason, and this port avoids `str.splitlines()` for the same reason.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import Callable
from typing import Any


def serialize_json_line(value: Any) -> str:
    """Serialize one strict JSONL record, terminated by a single LF."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


class JsonlLineReader:
    """Incremental LF-only JSONL reader.

    Feed it arbitrary ``bytes`` or ``str`` chunks with :meth:`feed`; each
    completed line is handed to ``on_line`` with a trailing CR stripped, so
    CRLF-framed input is accepted too. Call :meth:`close` at end of stream to
    flush a final unterminated line.
    """

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._on_line = on_line
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._closed = False

    def feed(self, chunk: bytes | str) -> None:
        if self._closed:
            raise RuntimeError("JsonlLineReader is closed")
        self._buffer += chunk if isinstance(chunk, str) else self._decoder.decode(chunk)

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                return
            self._emit(self._buffer[:newline_index])
            self._buffer = self._buffer[newline_index + 1 :]

    def close(self) -> None:
        """Flush any trailing partial line. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        self._on_line(line[:-1] if line.endswith("\r") else line)


def iter_json_lines(chunks: list[bytes | str]) -> list[str]:
    """Decode a sequence of chunks into complete lines. Convenience for tests."""
    lines: list[str] = []
    reader = JsonlLineReader(lines.append)
    for chunk in chunks:
        reader.feed(chunk)
    reader.close()
    return lines
