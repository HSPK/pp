"""Core component interfaces.

Python port of the `Component`, `Focusable`, `CURSOR_MARKER`, and `Container`
declarations from `packages/tui/src/tui.ts`. The rest of `tui.ts` (the
terminal driver, overlay stack, differential rendering loop) is out of scope
for this port phase; only the plain-data interfaces and the `Container` helper
class that other components depend on are ported here.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# Cursor position marker - APC (Application Program Command) sequence.
# This is a zero-width escape sequence that terminals ignore. Components emit
# this at the cursor position when focused; the TUI driver finds and strips
# this marker, then positions the hardware cursor there.
CURSOR_MARKER = "\x1b_pi:c\x07"


class Component(ABC):
    """All components must implement this.

    `handle_input` and `wants_key_release` are optional in the TypeScript
    interface (`handleInput?`, `wantsKeyRelease?`); subclasses that accept
    keyboard input override `handle_input` and set `wants_key_release = True`
    as needed. The base class leaves them unset so callers can use
    `getattr(component, "handle_input", None)` to detect support, matching
    the TypeScript `"handleInput" in component` check.
    """

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """Render the component to lines for the given viewport width."""

    @abstractmethod
    def invalidate(self) -> None:
        """Invalidate any cached rendering state.

        Called when theme changes or when component needs to re-render from
        scratch.
        """


@runtime_checkable
class Focusable(Protocol):
    """Components that can receive focus and display a hardware cursor.

    When focused, the component should emit `CURSOR_MARKER` at the cursor
    position in its render output.
    """

    focused: bool


def is_focusable(component: Component | None) -> bool:
    """Type guard to check if a component implements Focusable."""
    return component is not None and hasattr(component, "focused")


class Container(Component):
    """A component that contains other components."""

    def __init__(self) -> None:
        self.children: list[Component] = []

    def add_child(self, component: Component) -> None:
        self.children.append(component)

    def remove_child(self, component: Component) -> None:
        with contextlib.suppress(ValueError):
            self.children.remove(component)

    def clear(self) -> None:
        self.children = []

    def invalidate(self) -> None:
        for child in self.children:
            child.invalidate()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines
