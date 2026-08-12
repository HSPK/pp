"""The interface a custom editor component must satisfy.

Python port of `packages/tui/src/editor-component.ts`.

An extension can replace the built-in editor with its own (vim mode, emacs
mode, custom keybindings) as long as it provides these members. Upstream marks
most of the interface `?`-optional and callers probe with `"method" in
component`; only text access and input handling are truly required.

That required/optional split is load-bearing, so it is modelled explicitly
here rather than as one big interface:

- :class:`EditorComponent` is `runtime_checkable` and declares **only** the
  required members, so `isinstance(editor, EditorComponent)` answers "is this
  usable as an editor at all?". Putting the optional members in it would make
  `isinstance` reject every minimal editor upstream accepts.
- :class:`EditorComponentExtras` declares the optional members for type
  checking, and :func:`supports` is the runtime probe, standing in for
  upstream's `in` check.

Upstream also declares `EditorComponent extends Component`. A `Protocol`
cannot inherit from `Component` (an ABC), so `render`/`invalidate` are
restated here; the effect is the same, since a `Protocol` matches
structurally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .autocomplete import AutocompleteProvider


@runtime_checkable
class EditorComponent(Protocol):
    """The members every editor must provide."""

    def render(self, width: int) -> list[str]:
        """Render to lines for the given viewport width (from `Component`)."""
        ...

    def invalidate(self) -> None:
        """Drop cached rendering state (from `Component`)."""
        ...

    def get_text(self) -> str:
        """The current text content."""
        ...

    def set_text(self, text: str) -> None:
        """Replace the text content."""
        ...

    def handle_input(self, data: str) -> None:
        """Handle raw terminal input: key presses, paste sequences, and so on."""
        ...


class EditorComponentExtras(Protocol):
    """The members an editor may omit. Probe with :func:`supports` before calling."""

    on_submit: Callable[[str], None] | None
    """Called when the user submits, e.g. with Enter."""
    on_change: Callable[[str], None] | None
    """Called whenever the text changes."""
    border_color: Callable[[str], str] | None
    """Wraps the border in colour escapes."""

    def add_to_history(self, text: str) -> None:
        """Record `text` for up/down history navigation."""
        ...

    def insert_text_at_cursor(self, text: str) -> None:
        """Insert `text` at the cursor."""
        ...

    def get_expanded_text(self) -> str:
        """Text with markers (e.g. paste markers) expanded."""
        ...

    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        """Install the autocomplete provider."""
        ...

    def set_padding_x(self, padding: int) -> None:
        """Set horizontal padding."""
        ...

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        """Cap the autocomplete dropdown's visible items."""
        ...


REQUIRED_EDITOR_METHODS = ("render", "invalidate", "get_text", "set_text", "handle_input")
"""Members every editor must implement."""

OPTIONAL_EDITOR_METHODS = (
    "add_to_history",
    "insert_text_at_cursor",
    "get_expanded_text",
    "set_autocomplete_provider",
    "set_padding_x",
    "set_autocomplete_max_visible",
)
"""Members an editor may omit, mirroring the `?`-optional members upstream."""


def supports(editor: object, method: str) -> bool:
    """Whether `editor` implements an optional member. Mirrors TypeScript's `"m" in component`."""
    return callable(getattr(editor, method, None))


def get_expanded_text(editor: EditorComponent) -> str:
    """The editor's expanded text, falling back to `get_text()` as upstream documents."""
    if supports(editor, "get_expanded_text"):
        return editor.get_expanded_text()  # type: ignore[attr-defined]
    return editor.get_text()


__all__ = [
    "OPTIONAL_EDITOR_METHODS",
    "REQUIRED_EDITOR_METHODS",
    "EditorComponent",
    "EditorComponentExtras",
    "get_expanded_text",
    "supports",
]
