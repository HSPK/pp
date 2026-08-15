"""Python port of `packages/coding-agent/test/suite/regressions/2791-fswatch-error-crash.test.ts`.

**Not portable.** The TypeScript test spawns a child Node process, installs a
custom theme with the theme file watcher enabled, digs the `FSWatcher` out of
`process._getActiveHandles()`, and emits a synthetic `'error'` event on it: an
`EventEmitter` with no `'error'` listener throws, which is exactly how the bug
killed the process.

None of that machinery exists here:

- `fs.watch`-based theme hot-reloading is not ported (see the README's list of
  omitted Node-only APIs); the Python theme loader reads theme files on
  demand, so there is no watcher to attach an error handler to.
- `process._getActiveHandles()` and Node's "unhandled `'error'` event
  terminates the process" `EventEmitter` semantics have no Python equivalent.

There is no observable behaviour left to pin, so the file records the reason
instead of a weakened test.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="fs.watch theme watcher is not ported; see module docstring")


def test_process_survives_error_event_on_theme_fswatcher() -> None:
    raise AssertionError("unreachable")
