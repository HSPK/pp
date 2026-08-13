"""Bash command execution with streaming support and cancellation.

Port of `packages/coding-agent/src/core/bash-executor.ts`. Provides the
execution primitive `AgentSession.execute_bash()` uses: a rolling in-memory
output buffer (bounded to twice the truncation byte budget), an overflow temp
file once output exceeds that budget, and final-output truncation via
`tools.truncate.truncate_tail`.

Reuses the already-ported local-exec implementation in
`pi_coding_agent.tools.bash` (`_exec_local`) instead of re-implementing
subprocess/signal handling. `BashOperations` stays pluggable so callers (e.g.
a remote/SSH execution backend) can substitute their own `exec` -- matching
the TypeScript module's documented purpose ("used for remote execution").
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Protocol

from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.config import get_bin_dir
from pi_coding_agent.tools.bash import _exec_local
from pi_coding_agent.tools.output_accumulator import _StreamingUtf8Decoder
from pi_coding_agent.tools.truncate import DEFAULT_MAX_BYTES, truncate_tail
from pi_coding_agent.utils.ansi import strip_ansi
from pi_coding_agent.utils.shell import get_shell_env, sanitize_binary_output


@dataclass
class BashResult:
    """Port of TS `BashResult`."""

    output: str
    """Combined stdout + stderr output (sanitized, possibly truncated)."""
    exit_code: int | None
    """Process exit code (`None` if killed/cancelled)."""
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None
    """Path to a temp file containing the full output, if it exceeded the truncation threshold."""


class BashOperations(Protocol):
    """Pluggable command execution backend. Port of TS `BashOperations`."""

    async def exec(
        self,
        command: str,
        cwd: str,
        on_data: Callable[[bytes], None],
        signal: AbortSignal | None,
        timeout: float | None,
        env: dict[str, str] | None,
    ) -> int | None: ...


@dataclass
class _LocalBashOperations:
    shell_path: str | None = None

    async def exec(
        self,
        command: str,
        cwd: str,
        on_data: Callable[[bytes], None],
        signal: AbortSignal | None,
        timeout: float | None,
        env: dict[str, str] | None,
    ) -> int | None:
        return await _exec_local(
            command,
            cwd,
            on_data,
            signal,
            timeout,
            env if env is not None else get_shell_env(get_bin_dir()),
            self.shell_path,
        )


def create_local_bash_operations(shell_path: str | None = None) -> BashOperations:
    """Bash operations backed by pi's built-in local shell execution.

    ``shell_path`` is the user's `shellPath` setting; it is resolved (and
    validated) by `get_shell_config` on every exec, as in TypeScript.
    """
    return _LocalBashOperations(shell_path=shell_path)


async def execute_bash_with_operations(
    command: str,
    cwd: str,
    operations: BashOperations,
    on_chunk: Callable[[str], None] | None = None,
    signal: AbortSignal | None = None,
) -> BashResult:
    """Execute `command` via `operations`, streaming sanitized output chunks.

    Keeps a rolling in-memory buffer bounded to `2 * DEFAULT_MAX_BYTES`
    (matching the TS `maxOutputBytes`); once total output exceeds
    `DEFAULT_MAX_BYTES` it also starts writing to a temp file so the full,
    untruncated output is recoverable after truncation.
    """
    output_chunks: list[str] = []
    output_bytes = 0
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    # A multi-byte character can straddle a pipe-read boundary, so decoding must
    # be stateful; decoding each chunk independently turns the split character
    # into replacement characters in both the streamed deltas and the session
    # transcript. Mirrors the TypeScript's `TextDecoder(..., {stream: true})`.
    decoder = _StreamingUtf8Decoder()

    state: dict[str, str | None] = {"temp_file_path": None}
    temp_file_holder: list[IO[str]] = []
    total_bytes_holder = [0]

    def ensure_temp_file() -> None:
        if state["temp_file_path"]:
            return
        fd, path = tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
        state["temp_file_path"] = path
        # Stays open across subsequent `on_data` calls; closed explicitly below.
        handle = open(fd, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        for chunk in output_chunks:
            handle.write(chunk)
        temp_file_holder.append(handle)

    def _emit_text(text: str) -> None:
        nonlocal output_bytes
        if not text:
            return

        if total_bytes_holder[0] > DEFAULT_MAX_BYTES:
            ensure_temp_file()

        if temp_file_holder:
            temp_file_holder[0].write(text)

        output_chunks.append(text)
        output_bytes += len(text)
        while output_bytes > max_output_bytes and len(output_chunks) > 1:
            removed = output_chunks.pop(0)
            output_bytes -= len(removed)

        if on_chunk:
            on_chunk(text)

    def on_data(data: bytes) -> None:
        total_bytes_holder[0] += len(data)
        _emit_text(sanitize_binary_output(strip_ansi(decoder.decode(data))).replace("\r", ""))

    def finalize(cancelled: bool, exit_code: int | None) -> BashResult:
        # Flush any bytes the decoder is still holding, so output that ends mid
        # multi-byte character is not silently dropped.
        trailing = decoder.decode(b"", final=True)
        if trailing:
            _emit_text(sanitize_binary_output(strip_ansi(trailing)).replace("\r", ""))

        full_output = "".join(output_chunks)
        truncation_result = truncate_tail(full_output)
        if truncation_result.truncated:
            ensure_temp_file()
        if temp_file_holder:
            temp_file_holder[0].close()
        return BashResult(
            output=truncation_result.content if truncation_result.truncated else full_output,
            exit_code=None if cancelled else exit_code,
            cancelled=cancelled,
            truncated=truncation_result.truncated,
            full_output_path=state["temp_file_path"],
        )

    try:
        exit_code = await operations.exec(command, cwd, on_data, signal, None, None)
        return finalize(cancelled=bool(signal and signal.aborted), exit_code=exit_code)
    except Exception:
        if signal and signal.aborted:
            return finalize(cancelled=True, exit_code=None)
        if temp_file_holder:
            temp_file_holder[0].close()
        raise
