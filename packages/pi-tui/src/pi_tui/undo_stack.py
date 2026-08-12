"""Generic undo stack with clone-on-push semantics, ported from
`packages/tui/src/undo-stack.ts`.

Stores deep copies of state snapshots (via `copy.deepcopy`, the Python
equivalent of `structuredClone`). Popped snapshots are returned directly (no
re-copying) since they are already detached.
"""

from __future__ import annotations

import copy
from typing import Generic, TypeVar

S = TypeVar("S")


class UndoStack(Generic[S]):
    def __init__(self) -> None:
        self._stack: list[S] = []

    def push(self, state: S) -> None:
        """Push a deep copy of the given state onto the stack."""
        self._stack.append(copy.deepcopy(state))

    def pop(self) -> S | None:
        """Pop and return the most recent snapshot, or `None` if empty."""
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        """Remove all snapshots."""
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)
