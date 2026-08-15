"""Print a one-time-per-message deprecation warning.

Ported from ``packages/coding-agent/src/utils/deprecation.ts``.

Deprecating a config option or CLI flag without breaking callers outright
means warning about it instead. Doing that naively (printing every time the
deprecated path runs) floods the terminal when the same deprecated setting is
read on every turn; this module tracks which exact messages have already
fired and only prints each one once per process.
"""

from __future__ import annotations

import sys

_emitted_deprecation_warnings: set[str] = set()


def warn_deprecation(message: str) -> None:
    if message in _emitted_deprecation_warnings:
        return
    _emitted_deprecation_warnings.add(message)
    print(f"Deprecation warning: {message}", file=sys.stderr)


def clear_deprecation_warnings_for_tests() -> None:
    """Clear deprecation warning state. Exported for tests."""
    _emitted_deprecation_warnings.clear()
