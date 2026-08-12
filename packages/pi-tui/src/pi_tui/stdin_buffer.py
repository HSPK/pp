"""Incremental stdin parsing into complete escape sequences.

Python port of `packages/tui/src/stdin-buffer.ts`.

`StdinBuffer` buffers input and emits complete sequences. This is necessary
because stdin data events can arrive in partial chunks, especially for escape
sequences like mouse events. Without buffering, partial sequences can be
misinterpreted as regular keypresses.

For example, the mouse SGR sequence `\\x1b[<35;20;5m` might arrive as:
- Chunk 1: `\\x1b`
- Chunk 2: `[<35`
- Chunk 3: `;20;5m`

The buffer accumulates these until a complete sequence is detected. Feed
input via `process()`.

Based on code from OpenTUI (https://github.com/anomalyco/opentui), MIT
License, Copyright (c) 2025 opentui.

The TypeScript version extends `EventEmitter` with `"data"`/`"paste"`
events; here `StdinBuffer.on("data"/"paste", callback)` registers plain
callback lists instead of depending on a full event-emitter port. The
incomplete-sequence flush timeout uses `asyncio.call_later` (requiring a
running event loop) rather than `threading.Timer`, because the flush
callback re-enters the same single-threaded input/render pipeline as every
other stdin event, and mixing threads there would not be safe.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Literal

ESC = "\x1b"
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"

_SequenceStatus = Literal["complete", "incomplete", "not-escape"]

_MOUSE_SGR_RE = re.compile(r"^<\d+;\d+;\d+[Mm]$")
_KITTY_PRINTABLE_RE = re.compile(r"^\x1b\[(\d+)(?::\d*)?(?::\d+)?u$")


def _is_complete_sequence(data: str) -> _SequenceStatus:
    if not data.startswith(ESC):
        return "not-escape"

    if len(data) == 1:
        return "incomplete"

    after_esc = data[1:]

    if after_esc.startswith("["):
        if after_esc.startswith("[M"):
            return "complete" if len(data) >= 6 else "incomplete"
        return _is_complete_csi_sequence(data)

    if after_esc.startswith("]"):
        return _is_complete_osc_sequence(data)

    if after_esc.startswith("P"):
        return _is_complete_dcs_sequence(data)

    if after_esc.startswith("_"):
        return _is_complete_apc_sequence(data)

    if after_esc.startswith("O"):
        return "complete" if len(after_esc) >= 2 else "incomplete"

    if len(after_esc) == 1:
        return "complete"

    return "complete"


def _is_complete_csi_sequence(data: str) -> Literal["complete", "incomplete"]:
    if not data.startswith(f"{ESC}["):
        return "complete"

    if len(data) < 3:
        return "incomplete"

    payload = data[2:]

    last_char = payload[-1]
    last_char_code = ord(last_char)

    if 0x40 <= last_char_code <= 0x7E:
        if payload.startswith("<"):
            if _MOUSE_SGR_RE.match(payload):
                return "complete"
            if last_char in ("M", "m"):
                parts = payload[1:-1].split(";")
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    return "complete"
            return "incomplete"

        return "complete"

    return "incomplete"


def _is_complete_osc_sequence(data: str) -> Literal["complete", "incomplete"]:
    if not data.startswith(f"{ESC}]"):
        return "complete"
    if data.endswith(f"{ESC}\\") or data.endswith("\x07"):
        return "complete"
    return "incomplete"


def _is_complete_dcs_sequence(data: str) -> Literal["complete", "incomplete"]:
    if not data.startswith(f"{ESC}P"):
        return "complete"
    if data.endswith(f"{ESC}\\"):
        return "complete"
    return "incomplete"


def _is_complete_apc_sequence(data: str) -> Literal["complete", "incomplete"]:
    if not data.startswith(f"{ESC}_"):
        return "complete"
    if data.endswith(f"{ESC}\\"):
        return "complete"
    return "incomplete"


def _parse_unmodified_kitty_printable_codepoint(sequence: str) -> int | None:
    match = _KITTY_PRINTABLE_RE.match(sequence)
    if not match:
        return None
    codepoint = int(match.group(1))
    return codepoint if codepoint >= 32 else None


def _extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    sequences: list[str] = []
    pos = 0

    while pos < len(buffer):
        remaining = buffer[pos:]

        if remaining.startswith(ESC):
            seq_end = 1
            while seq_end <= len(remaining):
                candidate = remaining[:seq_end]
                status = _is_complete_sequence(candidate)

                if status == "complete":
                    # WezTerm with enable_kitty_keyboard sends the Escape key press as
                    # a raw '\x1b' byte and the release as a full Kitty CSI-u sequence.
                    # These arrive concatenated as '\x1b\x1b[27;...u'. Without this
                    # check, '\x1b\x1b' would be treated as a complete meta-key
                    # sequence (ESC + single char), leaving '[27;...u' to be typed as
                    # plain text. If the character immediately following '\x1b\x1b'
                    # would begin a new escape sequence, emit only the first ESC and
                    # restart from the second.
                    if candidate == "\x1b\x1b":
                        next_char = remaining[seq_end] if seq_end < len(remaining) else ""
                        if next_char in ("[", "]", "O", "P", "_"):
                            sequences.append(ESC)
                            pos += 1
                            break
                    sequences.append(candidate)
                    pos += seq_end
                    break
                elif status == "incomplete":
                    seq_end += 1
                else:
                    sequences.append(candidate)
                    pos += seq_end
                    break

            if seq_end > len(remaining):
                return sequences, remaining
        else:
            sequences.append(remaining[0])
            pos += 1

    return sequences, ""


class StdinBuffer:
    """Buffers stdin input and emits complete sequences via the `"data"` event.

    Handles partial escape sequences that arrive across multiple chunks.
    """

    def __init__(self, timeout: float = 0.01) -> None:
        self._buffer = ""
        self._timeout_handle: asyncio.TimerHandle | None = None
        self._timeout_s = timeout
        self._paste_mode = False
        self._paste_buffer = ""
        self._pending_kitty_printable_codepoint: int | None = None
        self._data_listeners: list[Callable[[str], None]] = []
        self._paste_listeners: list[Callable[[str], None]] = []

    def on(self, event: Literal["data", "paste"], listener: Callable[[str], None]) -> None:
        if event == "data":
            self._data_listeners.append(listener)
        elif event == "paste":
            self._paste_listeners.append(listener)
        else:
            raise ValueError(f"Unknown StdinBuffer event: {event!r}")

    def _emit(self, event: Literal["data", "paste"], value: str) -> None:
        listeners = self._data_listeners if event == "data" else self._paste_listeners
        for listener in list(listeners):
            listener(value)

    def process(self, data: str | bytes) -> None:
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        # Handle high-byte conversion (for compatibility with parseKeypress).
        # If buffer has single byte > 127, convert to ESC + (byte - 128).
        if isinstance(data, (bytes, bytearray)):
            if len(data) == 1 and data[0] > 127:
                byte = data[0] - 128
                text = f"{ESC}{chr(byte)}"
            else:
                text = data.decode("utf-8", errors="replace")
        else:
            text = data

        if len(text) == 0 and len(self._buffer) == 0:
            self._emit_data_sequence("")
            return

        self._buffer += text

        if self._paste_mode:
            self._paste_buffer += self._buffer
            self._buffer = ""

            end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_index != -1:
                pasted_content = self._paste_buffer[:end_index]
                remaining = self._paste_buffer[end_index + len(BRACKETED_PASTE_END) :]

                self._paste_mode = False
                self._paste_buffer = ""
                self._pending_kitty_printable_codepoint = None

                self._emit("paste", pasted_content)

                if remaining:
                    self.process(remaining)
            return

        start_index = self._buffer.find(BRACKETED_PASTE_START)
        if start_index != -1:
            if start_index > 0:
                before_paste = self._buffer[:start_index]
                sequences, _ = _extract_complete_sequences(before_paste)
                for sequence in sequences:
                    self._emit_data_sequence(sequence)

            self._pending_kitty_printable_codepoint = None
            self._buffer = self._buffer[start_index + len(BRACKETED_PASTE_START) :]
            self._paste_mode = True
            self._paste_buffer = self._buffer
            self._buffer = ""

            end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_index != -1:
                pasted_content = self._paste_buffer[:end_index]
                remaining = self._paste_buffer[end_index + len(BRACKETED_PASTE_END) :]

                self._paste_mode = False
                self._paste_buffer = ""
                self._pending_kitty_printable_codepoint = None

                self._emit("paste", pasted_content)

                if remaining:
                    self.process(remaining)
            return

        sequences, remainder = _extract_complete_sequences(self._buffer)
        self._buffer = remainder

        for sequence in sequences:
            self._emit_data_sequence(sequence)

        if self._buffer:
            loop = asyncio.get_running_loop()

            def _on_timeout() -> None:
                self._timeout_handle = None
                for sequence in self.flush():
                    self._emit_data_sequence(sequence)

            self._timeout_handle = loop.call_later(self._timeout_s, _on_timeout)

    def _emit_data_sequence(self, sequence: str) -> None:
        raw_codepoint = ord(sequence) if len(sequence) == 1 else None
        if raw_codepoint is not None and raw_codepoint == self._pending_kitty_printable_codepoint:
            self._pending_kitty_printable_codepoint = None
            return

        self._pending_kitty_printable_codepoint = _parse_unmodified_kitty_printable_codepoint(sequence)
        self._emit("data", sequence)

    def flush(self) -> list[str]:
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        if len(self._buffer) == 0:
            return []

        sequences = [self._buffer]
        self._buffer = ""
        self._pending_kitty_printable_codepoint = None
        return sequences

    def clear(self) -> None:
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None
        self._buffer = ""
        self._paste_mode = False
        self._paste_buffer = ""
        self._pending_kitty_printable_codepoint = None

    def get_buffer(self) -> str:
        return self._buffer

    def destroy(self) -> None:
        self.clear()
