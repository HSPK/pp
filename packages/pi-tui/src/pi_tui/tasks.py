"""Background task ownership for the interactive TUI driver.

`asyncio.ensure_future` returns a task that the event loop only holds
weakly, so a fire-and-forget producer (for example the alt-screen
selection auto-scroll loop, or the terminal progress-indicator keepalive)
can be garbage collected mid-flight. Every background task started by
`pi_tui` is registered here until it finishes.

This mirrors `pi_ai.utils.tasks.spawn` but lives in `pi_tui` so this package
does not depend on `pi_ai`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_background_tasks: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Start `coro` and keep a strong reference until it completes."""
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
