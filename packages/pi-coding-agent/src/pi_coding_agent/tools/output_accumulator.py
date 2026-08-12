"""Bounded-memory streaming output accumulator for the bash tool.

Python port of `packages/coding-agent/src/core/tools/output-accumulator.ts`.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass

from pi_coding_agent.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    truncate_tail,
)


def _default_temp_file_path(prefix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{prefix}-{uuid.uuid4().hex}.log")


@dataclass
class OutputSnapshot:
    content: str
    truncation: TruncationResult
    full_output_path: str | None = None


class OutputAccumulator:
    """Incrementally tracks streaming output with bounded memory.

    Appends decoded chunks with a streaming UTF-8 decoder, keeps only a
    decoded tail for display snapshots, and opens a temp file when the full
    output needs to be preserved.
    """

    def __init__(
        self,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        temp_file_prefix: str = "pi-output",
    ) -> None:
        self._max_lines = max_lines
        self._max_bytes = max_bytes
        self._max_rolling_bytes = max(max_bytes * 2, 1)
        self._temp_file_prefix = temp_file_prefix
        self._decoder = _StreamingUtf8Decoder()

        self._raw_chunks: list[bytes] = []
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_raw_bytes = 0
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False

        self._temp_file_path: str | None = None
        self._temp_file = None

    def append(self, data: bytes) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")

        self._total_raw_bytes += len(data)
        self._append_decoded_text(self._decoder.decode(data))

        if self._temp_file is not None or self._should_use_temp_file():
            self._ensure_temp_file()
            if self._temp_file is not None:
                self._temp_file.write(data)
        elif data:
            self._raw_chunks.append(data)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._append_decoded_text(self._decoder.decode(b"", final=True))
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, persist_if_truncated: bool = False) -> OutputSnapshot:
        tail_truncation = truncate_tail(self._get_snapshot_text(), max_lines=self._max_lines, max_bytes=self._max_bytes)
        truncated = self._total_lines > self._max_lines or self._total_decoded_bytes > self._max_bytes
        truncated_by = None
        if truncated:
            truncated_by = tail_truncation.truncated_by or (
                "bytes" if self._total_decoded_bytes > self._max_bytes else "lines"
            )

        truncation = TruncationResult(
            content=tail_truncation.content,
            truncated=truncated,
            truncated_by=truncated_by,
            total_lines=self._total_lines,
            total_bytes=self._total_decoded_bytes,
            output_lines=tail_truncation.output_lines,
            output_bytes=tail_truncation.output_bytes,
            last_line_partial=tail_truncation.last_line_partial,
            first_line_exceeds_limit=tail_truncation.first_line_exceeds_limit,
            max_lines=self._max_lines,
            max_bytes=self._max_bytes,
        )

        if persist_if_truncated and truncation.truncated:
            self._ensure_temp_file()

        return OutputSnapshot(content=truncation.content, truncation=truncation, full_output_path=self._temp_file_path)

    def close_temp_file(self) -> None:
        if self._temp_file is not None:
            self._temp_file.close()
            self._temp_file = None

    def get_last_line_bytes(self) -> int:
        return self._current_line_bytes

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return

        text_bytes = len(text.encode("utf-8"))
        self._total_decoded_bytes += text_bytes
        self._tail_text += text
        self._tail_bytes += text_bytes
        if self._tail_bytes > self._max_rolling_bytes * 2:
            self._trim_tail()

        newlines = text.count("\n")
        if newlines == 0:
            self._current_line_bytes += text_bytes
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            last_newline = text.rindex("\n")
            tail = text[last_newline + 1 :]
            self._current_line_bytes = len(tail.encode("utf-8"))
            self._has_open_line = len(tail) > 0
        self._total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        buf = self._tail_text.encode("utf-8")
        if len(buf) <= self._max_rolling_bytes:
            self._tail_bytes = len(buf)
            return

        start = len(buf) - self._max_rolling_bytes
        while start < len(buf) and (buf[start] & 0xC0) == 0x80:
            start += 1

        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary if start == 0 else buf[start - 1] == 0x0A
        )
        self._tail_text = buf[start:].decode("utf-8", errors="replace")
        self._tail_bytes = len(self._tail_text.encode("utf-8"))

    def _get_snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text

        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_raw_bytes > self._max_bytes
            or self._total_decoded_bytes > self._max_bytes
            or self._total_lines > self._max_lines
        )

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path is not None:
            return
        self._temp_file_path = _default_temp_file_path(self._temp_file_prefix)
        self._temp_file = open(self._temp_file_path, "wb")  # noqa: SIM115 - kept open across appends
        for chunk in self._raw_chunks:
            self._temp_file.write(chunk)
        self._raw_chunks = []


class _StreamingUtf8Decoder:
    """Incrementally decodes UTF-8 bytes, buffering incomplete trailing sequences.

    Mirrors JavaScript's `TextDecoder` in streaming mode: a multi-byte
    character split across two `append()` calls decodes correctly instead of
    producing replacement characters at the boundary.
    """

    def __init__(self) -> None:
        self._buffer = b""

    def decode(self, data: bytes, final: bool = False) -> str:
        combined = self._buffer + data
        if final:
            self._buffer = b""
            return combined.decode("utf-8", errors="replace")

        # Find the longest valid-UTF-8 prefix, holding back an incomplete
        # trailing multi-byte sequence for the next chunk.
        valid_end = len(combined)
        for back in range(0, min(4, len(combined))):
            idx = len(combined) - 1 - back
            byte = combined[idx]
            if byte & 0xC0 == 0x80:  # continuation byte, keep scanning backwards
                continue
            if byte & 0x80 == 0:
                seq_len = 1
            elif byte & 0xE0 == 0xC0:
                seq_len = 2
            elif byte & 0xF0 == 0xE0:
                seq_len = 3
            elif byte & 0xF8 == 0xF0:
                seq_len = 4
            else:
                seq_len = 1  # invalid leading byte; let errors="replace" handle it
            if idx + seq_len > len(combined):
                valid_end = idx
            break

        self._buffer = combined[valid_end:]
        return combined[:valid_end].decode("utf-8", errors="replace")
