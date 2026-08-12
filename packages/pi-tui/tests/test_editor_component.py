"""Tests for `pi_tui.editor_component`.

No dedicated TS test file exists for `packages/tui/src/editor-component.ts`;
it is a pure interface declaration. These pin the required/optional split,
which is the part the Python port had to model explicitly: `isinstance` must
accept a minimal editor, and optional members must be probed, not assumed.
"""

from __future__ import annotations

import pytest
from pi_tui.editor_component import (
    OPTIONAL_EDITOR_METHODS,
    REQUIRED_EDITOR_METHODS,
    EditorComponent,
    get_expanded_text,
    supports,
)


class MinimalEditor:
    """Implements only the required members, as upstream permits."""

    def __init__(self) -> None:
        self.text = "hello"
        self.inputs: list[str] = []

    def render(self, width: int) -> list[str]:
        return [self.text[:width]]

    def invalidate(self) -> None:
        pass

    def get_text(self) -> str:
        return self.text

    def set_text(self, text: str) -> None:
        self.text = text

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)


class FullEditor(MinimalEditor):
    """Also implements the optional members."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[str] = []

    def add_to_history(self, text: str) -> None:
        self.history.append(text)

    def insert_text_at_cursor(self, text: str) -> None:
        self.text += text

    def get_expanded_text(self) -> str:
        return f"{self.text}[expanded]"

    def set_autocomplete_provider(self, provider: object) -> None:
        pass

    def set_padding_x(self, padding: int) -> None:
        pass

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        pass


def test_a_minimal_editor_satisfies_the_protocol():
    # The optional members must not be part of the runtime check, or every
    # editor upstream accepts would be rejected here.
    assert isinstance(MinimalEditor(), EditorComponent)


def test_a_full_editor_satisfies_the_protocol():
    assert isinstance(FullEditor(), EditorComponent)


def test_an_unrelated_object_does_not_satisfy_the_protocol():
    assert not isinstance(object(), EditorComponent)


def test_an_editor_missing_a_required_member_does_not_satisfy_the_protocol():
    class NoHandleInput:
        def render(self, width):
            return []

        def invalidate(self):
            pass

        def get_text(self):
            return ""

        def set_text(self, text):
            pass

    assert not isinstance(NoHandleInput(), EditorComponent)


@pytest.mark.parametrize("method", OPTIONAL_EDITOR_METHODS)
def test_supports_distinguishes_minimal_from_full_editors(method):
    assert supports(FullEditor(), method) is True
    assert supports(MinimalEditor(), method) is False


def test_supports_rejects_a_non_callable_attribute():
    class NotCallable(MinimalEditor):
        add_to_history = "surely not"

    assert supports(NotCallable(), "add_to_history") is False


def test_expanded_text_falls_back_to_get_text():
    assert get_expanded_text(MinimalEditor()) == "hello"


def test_expanded_text_uses_the_optional_member_when_present():
    assert get_expanded_text(FullEditor()) == "hello[expanded]"


def test_required_and_optional_member_lists_do_not_overlap():
    assert not set(REQUIRED_EDITOR_METHODS) & set(OPTIONAL_EDITOR_METHODS)
    assert all(hasattr(MinimalEditor, name) for name in REQUIRED_EDITOR_METHODS)
