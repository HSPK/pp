"""Serialize concurrent mutations to the same file.

Python port of `packages/coding-agent/src/core/tools/file-mutation-queue.ts`.

Operations targeting the same file run one at a time (FIFO); operations
targeting different files still run concurrently. The `edit` and `write`
tools both go through this so an edit and a write racing for the same path
can never interleave their read-modify-write sequences.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

_locks: dict[str, asyncio.Lock] = {}
_waiters: dict[str, int] = {}
_registration_lock = asyncio.Lock()


def _mutation_queue_key(file_path: str) -> str:
    # Unlike Node's fs.realpath (which throws ENOENT for a missing path),
    # os.path.realpath resolves symlinks best-effort and simply returns a
    # normalized path when the target does not exist yet - the same fallback
    # the TypeScript version implements explicitly via a try/except.
    return os.path.realpath(file_path)


async def with_file_mutation_queue(file_path: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run ``fn`` with exclusive access to ``file_path``, queued behind any in-flight mutation."""
    key = _mutation_queue_key(file_path)
    async with _registration_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
        _waiters[key] = _waiters.get(key, 0) + 1

    async with lock:
        try:
            return await fn()
        finally:
            async with _registration_lock:
                _waiters[key] -= 1
                if _waiters[key] <= 0:
                    _locks.pop(key, None)
                    _waiters.pop(key, None)
