"""Serialized, backpressure-aware raw stdout writes.

Ported from ``packages/coding-agent/src/core/output-guard.ts``.

Print/JSON modes write the protocol stream to stdout while extensions and
libraries may also print. ``take_over_stdout`` redirects ordinary ``stdout``
writes to stderr so only the protocol reaches stdout, and ``write_raw_stdout``
serializes protocol writes through one queue so interleaved chunks cannot
corrupt the stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import sys
import time
from dataclasses import dataclass
from typing import IO, Any

RAW_STDOUT_RETRY_DELAY_S = 0.01
_RETRYABLE_ERRNOS = frozenset({errno.ENOBUFS, errno.EAGAIN, errno.EWOULDBLOCK})


@dataclass
class _StdoutTakeoverState:
    raw_stdout: IO[str]
    raw_stderr: IO[str]
    original_stdout: IO[str]


_takeover_state: _StdoutTakeoverState | None = None
_write_tail: asyncio.Future[None] | None = None


class _StdoutToStderr:
    """Stand-in for TypeScript's monkeypatched ``process.stdout.write``."""

    def __init__(self, target: IO[str]) -> None:
        self._target = target

    def write(self, text: str) -> int:
        return self._target.write(text)

    def flush(self) -> None:
        self._target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _raw_stdout() -> IO[str]:
    if _takeover_state is not None:
        return _takeover_state.raw_stdout
    return sys.stdout


def take_over_stdout() -> None:
    """Route ordinary ``print``/``sys.stdout`` writes to stderr."""
    global _takeover_state
    if _takeover_state is not None:
        return
    _takeover_state = _StdoutTakeoverState(raw_stdout=sys.stdout, raw_stderr=sys.stderr, original_stdout=sys.stdout)
    sys.stdout = _StdoutToStderr(sys.stderr)  # type: ignore[assignment]


def restore_stdout() -> None:
    global _takeover_state
    if _takeover_state is None:
        return
    sys.stdout = _takeover_state.original_stdout
    _takeover_state = None


def is_stdout_taken_over() -> bool:
    return _takeover_state is not None


def _write_chunk_blocking(text: str) -> None:
    """Write one chunk, retrying the transient errnos a full pipe can raise."""
    stream = _raw_stdout()
    while True:
        try:
            stream.write(text)
            stream.flush()
            return
        except OSError as error:
            if error.errno not in _RETRYABLE_ERRNOS:
                raise
            # A synchronous sleep here matches the TypeScript retry delay; the
            # async wrapper below keeps the event loop free.
            time.sleep(RAW_STDOUT_RETRY_DELAY_S)


async def _write_chunk(text: str) -> None:
    while True:
        stream = _raw_stdout()
        try:
            stream.write(text)
            stream.flush()
            return
        except OSError as error:
            if error.errno not in _RETRYABLE_ERRNOS:
                raise
            await asyncio.sleep(RAW_STDOUT_RETRY_DELAY_S)


def write_raw_stdout(text: str) -> None:
    """Queue ``text`` for stdout, preserving call order.

    Outside a running event loop the write happens inline, which is what the
    synchronous CLI paths need.
    """
    if len(text) == 0:
        return

    global _write_tail
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write_chunk_blocking(text)
        return

    previous = _write_tail

    async def run() -> None:
        if previous is not None:
            # A failed earlier write must not cancel this one.
            with contextlib.suppress(Exception):
                await asyncio.shield(previous)
        await _write_chunk(text)

    _write_tail = loop.create_task(run())


async def wait_for_raw_stdout_backpressure() -> None:
    """Wait until every queued write has drained."""
    while True:
        tail = _write_tail
        if tail is None:
            return
        try:
            await tail
        except Exception:
            return
        if tail is _write_tail:
            return


async def flush_raw_stdout() -> None:
    await wait_for_raw_stdout_backpressure()
    _raw_stdout().flush()


__all__ = [
    "flush_raw_stdout",
    "is_stdout_taken_over",
    "restore_stdout",
    "take_over_stdout",
    "wait_for_raw_stdout_backpressure",
    "write_raw_stdout",
]
