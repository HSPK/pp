"""Wait for a child process's stdio to fall idle after it exits.

Python port of the `waitForChildProcess` half of
`packages/coding-agent/src/utils/child-process.ts`. TypeScript's
`spawnProcess`/`spawnProcessSync` wrappers are not ported: they exist only to
route Windows spawns through `cross-spawn`, and this port spawns through
`asyncio.create_subprocess_exec`.

Node exposes stream `end`/`close` events, so its version is written as a single
promise around the child. Here the equivalent state is already held by the
stream-pump tasks that feed output to the caller, so the port keeps the same
policy -- after exit, wait for the pipes to fall idle, re-arming a short grace
on every chunk -- but takes those tasks as its input:

  * a descendant that keeps writing to an inherited pipe keeps us reading, so
    output written after `exit` is not truncated (earendil-works/pi#5303), and
  * a quiet descendant that merely holds the pipe open releases us after the
    grace instead of blocking until it exits.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence

EXIT_STDIO_GRACE_SECONDS = 0.1


async def wait_for_child_streams(
    pump_tasks: Sequence[asyncio.Task[None]],
    last_data_at: Callable[[], float],
    *,
    grace: float = EXIT_STDIO_GRACE_SECONDS,
) -> None:
    """Wait for the output pumps to finish, or for the pipes to fall idle.

    ``last_data_at`` returns the `loop.time()` of the most recent chunk; the
    grace deadline is measured from it, so it is re-armed by every chunk.
    """
    pending = [task for task in pump_tasks if task is not None]
    loop = asyncio.get_running_loop()
    idle_since = loop.time()
    while pending:
        remaining = max(last_data_at(), idle_since) + grace - loop.time()
        if remaining <= 0:
            break
        _done, still_pending = await asyncio.wait(pending, timeout=remaining)
        pending = [task for task in still_pending]

    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError):
            await task
