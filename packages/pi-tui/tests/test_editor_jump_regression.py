"""Regression tests for the editor's backward character jump.

JavaScript clamps a negative `lastIndexOf` fromIndex to 0, so a backward jump
at column 0 still examines index 0 of the current line. The port returned -1
and searched previous lines, moving the cursor to the wrong line.
"""

from __future__ import annotations

from pi_tui.components.editor import Editor
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen
from test_editor import DEFAULT_EDITOR_THEME


def make_editor() -> Editor:
    """Build an editor on the real `TuiMainScreen`, not a permissive double.

    An earlier version of this file used a `_FakeTui` whose `__getattr__`
    returned a no-op lambda for every attribute. That double accepted calls the
    real TUI does not have, and made `tui.terminal` a lambda, so editor code
    reaching for terminal geometry could not be exercised here at all.
    `TuiMainScreen` over `FakeTerminal` is offline and just as cheap.
    """
    return Editor(TuiMainScreen(FakeTerminal()), DEFAULT_EDITOR_THEME)


def test_backward_char_jump_at_column_zero_checks_index_zero():
    """Regression: the search skipped the current line entirely at column 0.

    JavaScript clamps a negative `fromIndex` to 0, so `lastIndexOf("h", -1)`
    still examines index 0. The port returned -1 and searched previous lines,
    moving the cursor to the wrong line.
    """
    editor = make_editor()
    editor.set_text("hello\nhi")
    editor._state.cursor_line = 1
    editor._set_cursor_col(0)

    editor._jump_to_char("h", "backward")

    assert editor.get_cursor() == {"line": 1, "col": 0}


def test_backward_char_jump_at_column_zero_falls_through_when_absent():
    editor = make_editor()
    editor.set_text("xyz\nabc")
    editor._state.cursor_line = 1
    editor._set_cursor_col(0)

    editor._jump_to_char("x", "backward")

    # "abc" does not start with "x", so the search continues to the line above.
    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_backward_char_jump_within_a_line_is_unaffected():
    editor = make_editor()
    editor.set_text("abcabc")
    editor._set_cursor_col(5)

    editor._jump_to_char("a", "backward")

    assert editor.get_cursor() == {"line": 0, "col": 3}
