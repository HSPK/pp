"""Editor key handlers, cursor edge cases, paste handling and rendering.

Companion to `test_editor.py`; the cases here cover the rarer input paths of
`pi_tui.components.editor` and are cross-checked against
`packages/tui/src/components/editor.ts` (and the parts of
`packages/tui/test/editor.test.ts` that had not been ported yet: the
wrapped-visual-line navigation, editor-resize and paste-marker vertical
navigation cases).
"""

from __future__ import annotations

import re

from pi_tui.components.editor import (
    Editor,
    EditorOptions,
    EditorTheme,
    word_wrap_line,
)
from pi_tui.components.select_list import SelectListTheme
from pi_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingsManager,
    set_keybindings,
)
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen
from pi_tui.utils import strip_terminal_sequences, visible_width

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(text: str) -> str:
    return text


SELECT_LIST_THEME = SelectListTheme(
    selected_prefix=_identity,
    selected_text=_identity,
    description=_identity,
    scroll_info=_identity,
    no_match=_identity,
)

EDITOR_THEME = EditorTheme(border_color=_identity, select_list=SELECT_LIST_THEME)

UNDO = "\x1b[45;5u"
PAGE_UP = "\x1b[5~"
PAGE_DOWN = "\x1b[6~"
SHIFT_SPACE = "\x1b[32;2u"
ESCAPE = "\x1b"
PASTE_RE = re.compile(r"\[paste #\d+ \+\d+ lines\]")


class RecordingTui:
    """Minimal TUI double that counts `request_render()` calls."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self.terminal = FakeTerminal(columns=columns, rows=rows)
        self.render_requests = 0

    def request_render(self, force: bool = False) -> None:
        self.render_requests += 1


def make_editor(cols: int = 80, rows: int = 24, options: EditorOptions | None = None) -> Editor:
    return Editor(TuiMainScreen(FakeTerminal(columns=cols, rows=rows)), EDITOR_THEME, options)


def content_lines(rendered: list[str]) -> list[str]:
    return rendered[1:-1]


def position_cursor(editor: Editor, line: int, col: int) -> None:
    """Mirror of the TS test helper `positionCursor`."""
    for _ in range(20):
        editor.handle_input("\x1b[A")
    for _ in range(line):
        editor.handle_input("\x1b[B")
    editor.handle_input("\x01")
    for _ in range(col):
        editor.handle_input("\x1b[C")


# ---------------------------------------------------------------------------
# Options accessors
# ---------------------------------------------------------------------------


class TestOptionAccessors:
    def test_padding_x_is_applied_and_requests_a_render_once(self):
        tui = RecordingTui()
        editor = Editor(tui, EDITOR_THEME)
        assert editor.get_padding_x() == 0

        editor.set_padding_x(2)
        assert editor.get_padding_x() == 2
        assert tui.render_requests == 1

        # Setting the same value again is a no-op.
        editor.set_padding_x(2)
        assert tui.render_requests == 1

        editor.set_text("hi")
        rendered = editor.render(20)
        assert content_lines(rendered)[0].startswith("  ")
        assert visible_width(rendered[1]) == 20

    def test_padding_x_clamps_negative_values_to_zero(self):
        tui = RecordingTui()
        editor = Editor(tui, EDITOR_THEME, EditorOptions(padding_x=3))
        assert editor.get_padding_x() == 3

        editor.set_padding_x(-5)
        assert editor.get_padding_x() == 0
        assert tui.render_requests == 1

    def test_autocomplete_max_visible_is_clamped_to_3_20(self):
        tui = RecordingTui()
        editor = Editor(tui, EDITOR_THEME)
        assert editor.get_autocomplete_max_visible() == 5

        editor.set_autocomplete_max_visible(100)
        assert editor.get_autocomplete_max_visible() == 20
        assert tui.render_requests == 1

        editor.set_autocomplete_max_visible(1)
        assert editor.get_autocomplete_max_visible() == 3
        assert tui.render_requests == 2

        # Already clamped to the same value - no extra render.
        editor.set_autocomplete_max_visible(2)
        assert editor.get_autocomplete_max_visible() == 3
        assert tui.render_requests == 2

    def test_autocomplete_max_visible_option_is_clamped_at_construction(self):
        editor = make_editor(options=EditorOptions(autocomplete_max_visible=2))
        assert editor.get_autocomplete_max_visible() == 3

        editor = make_editor(options=EditorOptions(autocomplete_max_visible=50))
        assert editor.get_autocomplete_max_visible() == 20


# ---------------------------------------------------------------------------
# History bookkeeping
# ---------------------------------------------------------------------------


class TestHistoryBookkeeping:
    def test_blank_entries_are_not_added(self):
        editor = make_editor()
        editor.add_to_history("   \n  ")
        editor.handle_input("\x1b[A")
        assert editor.get_text() == ""

    def test_consecutive_duplicates_are_not_added(self):
        editor = make_editor()
        editor.add_to_history("same")
        editor.add_to_history("same")
        editor.add_to_history("other")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "other"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "same"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "same"

    def test_history_is_capped_at_100_entries(self):
        editor = make_editor()
        for i in range(101):
            editor.add_to_history(f"prompt {i}")

        for _ in range(120):
            editor.handle_input("\x1b[A")

        # "prompt 0" was dropped when the 101st entry pushed the list over 100.
        assert editor.get_text() == "prompt 1"

    def test_restoring_the_draft_emits_on_change(self):
        editor = make_editor()
        changes: list[str] = []
        editor.add_to_history("previous")
        editor.set_text("draft")
        editor.on_change = changes.append

        editor.handle_input("\x01")
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "previous"
        assert changes[-1] == "previous"

        editor.handle_input("\x1b[B")
        assert editor.get_text() == "draft"
        assert changes[-1] == "draft"


# ---------------------------------------------------------------------------
# on_change notifications
# ---------------------------------------------------------------------------


class TestOnChangeNotifications:
    def test_typing_newline_and_submit_emit_changes(self):
        editor = make_editor()
        changes: list[str] = []
        submitted: list[str] = []
        editor.on_change = changes.append
        editor.on_submit = submitted.append

        editor.handle_input("h")
        editor.handle_input("i")
        editor.handle_input("\n")
        assert changes == ["h", "hi", "hi\n"]

        editor.handle_input("\r")
        assert submitted == ["hi"]
        assert changes[-1] == ""

    def test_insert_text_at_cursor_emits_a_single_change(self):
        editor = make_editor()
        changes: list[str] = []
        editor.on_change = changes.append

        editor.insert_text_at_cursor("abc\ndef")
        assert changes == ["abc\ndef"]
        assert editor.get_cursor() == {"line": 1, "col": 3}

    def test_insert_text_at_cursor_ignores_empty_text(self):
        editor = make_editor()
        changes: list[str] = []
        editor.set_text("keep")
        editor.on_change = changes.append

        editor.insert_text_at_cursor("")
        assert changes == []
        assert editor.get_text() == "keep"

    def test_deletions_emit_changes(self):
        editor = make_editor()
        changes: list[str] = []
        editor.set_text("hello world")
        editor.on_change = changes.append

        editor.handle_input("\x7f")  # backspace
        assert changes[-1] == "hello worl"

        editor.handle_input("\x01")  # start of line
        editor.handle_input("\x04")  # forward delete
        assert changes[-1] == "ello worl"

        editor.handle_input("\x05")  # end of line
        editor.handle_input("\x15")  # delete to line start
        assert changes[-1] == ""

    def test_kill_and_yank_emit_changes(self):
        editor = make_editor()
        changes: list[str] = []
        editor.set_text("alpha beta")
        editor.on_change = changes.append

        editor.handle_input("\x01")
        editor.handle_input("\x0b")  # ctrl+k: kill to end of line
        assert changes[-1] == ""

        editor.handle_input("\x19")  # ctrl+y: yank
        assert changes[-1] == "alpha beta"

        editor.handle_input("\x17")  # ctrl+w: delete word backwards
        assert changes[-1] == "alpha "

        editor.handle_input("\x01")
        editor.handle_input("\x1bd")  # alt+d: delete word forwards
        assert changes[-1] == " "

        editor.handle_input(UNDO)
        assert changes[-1] == "alpha "

    def test_yank_pop_emits_the_intermediate_and_final_text(self):
        editor = make_editor()
        editor.set_text("one two")
        editor.handle_input("\x05")
        editor.handle_input("\x17")  # kill "two"
        editor.handle_input("\x17")  # kill "one " (separate entry: cursor moved? no - accumulates)
        editor.set_text("")

        # Two independent kill-ring entries.
        editor.set_text("aaa")
        editor.handle_input("\x05")
        editor.handle_input("\x15")  # kill "aaa"
        editor.handle_input("\x1b[C")  # break the kill chain
        editor.set_text("bbb")
        editor.handle_input("\x05")
        editor.handle_input("\x15")  # kill "bbb"

        changes: list[str] = []
        editor.on_change = changes.append

        editor.handle_input("\x19")  # yank -> "bbb"
        assert editor.get_text() == "bbb"
        editor.handle_input("\x1by")  # yank-pop -> "aaa"
        assert editor.get_text() == "aaa"
        assert changes == ["bbb", "", "aaa"]


# ---------------------------------------------------------------------------
# Rare key handlers
# ---------------------------------------------------------------------------


class TestRareKeyHandlers:
    def teardown_method(self, method):
        set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))

    def test_shift_space_inserts_a_regular_space(self):
        editor = make_editor()
        editor.handle_input("a")
        editor.handle_input(SHIFT_SPACE)
        editor.handle_input("b")
        assert editor.get_text() == "a b"

    def test_page_up_and_page_down_move_by_a_page(self):
        editor = make_editor(rows=24)  # page size = max(5, 24 * 0.3) = 7
        editor.set_text("\n".join(f"line {i}" for i in range(20)))
        assert editor.get_cursor() == {"line": 19, "col": 7}

        editor.handle_input(PAGE_UP)
        assert editor.get_cursor() == {"line": 12, "col": 7}

        editor.handle_input(PAGE_DOWN)
        assert editor.get_cursor() == {"line": 19, "col": 7}

    def test_page_up_clamps_at_the_first_line(self):
        editor = make_editor(rows=24)
        editor.set_text("\n".join(f"line {i}" for i in range(20)))
        for _ in range(5):
            editor.handle_input(PAGE_UP)
        assert editor.get_cursor()["line"] == 0

        for _ in range(5):
            editor.handle_input(PAGE_DOWN)
        assert editor.get_cursor()["line"] == 19

    def test_down_arrow_on_last_visual_line_jumps_to_line_end(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": 11}

    def test_enter_is_ignored_when_submit_is_disabled(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append
        editor.disable_submit = True

        editor.handle_input("h")
        editor.handle_input("\r")
        assert submitted == []
        assert editor.get_text() == "h"

    def test_backslash_newline_does_not_submit_when_submit_is_disabled(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append
        editor.disable_submit = True

        editor.handle_input("\\")
        editor.handle_input("\n")  # ctrl+j / newline
        assert submitted == []
        assert editor.get_text() == "\\\n"

    def test_alt_enter_after_backslash_inserts_a_newline(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append

        editor.handle_input("\\")
        editor.handle_input("\x1b\r")
        assert submitted == []
        assert editor.get_text() == "\\\n"

    def test_backslash_enter_submits_when_shift_enter_is_bound_to_submit(self):
        set_keybindings(
            KeybindingsManager(
                TUI_KEYBINDINGS,
                {"tui.input.submit": ["shift+enter"], "tui.input.newLine": ["enter"]},
            )
        )
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append

        for ch in "abc\\":
            editor.handle_input(ch)
        editor.handle_input("\r")

        # The trailing backslash is removed and the prompt submitted.
        assert submitted == ["abc"]
        assert editor.get_text() == ""

    def test_enter_inserts_a_newline_without_backslash_when_shift_enter_submits(self):
        set_keybindings(
            KeybindingsManager(
                TUI_KEYBINDINGS,
                {"tui.input.submit": ["shift+enter"], "tui.input.newLine": ["enter"]},
            )
        )
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append

        for ch in "abc":
            editor.handle_input(ch)
        editor.handle_input("\r")

        assert submitted == []
        assert editor.get_text() == "abc\n"

    def test_ctrl_c_is_left_to_the_parent(self):
        editor = make_editor()
        changes: list[str] = []
        editor.set_text("keep me")
        editor.on_change = changes.append

        editor.handle_input("\x03")
        assert editor.get_text() == "keep me"
        assert changes == []

    def test_submit_without_a_submit_handler_still_clears_the_editor(self):
        editor = make_editor()
        changes: list[str] = []
        editor.set_text("text")
        editor.on_change = changes.append

        editor.handle_input("\r")
        assert editor.get_text() == ""
        assert changes[-1] == ""

    def test_invalidate_does_not_change_the_rendered_output(self):
        editor = make_editor()
        editor.set_text("stable")
        before = editor.render(20)

        editor.invalidate()
        assert editor.render(20) == before
        assert editor.get_text() == "stable"

    def test_escape_cancels_jump_mode_and_is_otherwise_ignored(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")

        editor.handle_input("\x1d")  # ctrl+]: enter jump mode
        editor.handle_input(ESCAPE)  # cancels jump mode, moves nothing
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("o")
        assert editor.get_text() == "ohello world"


# ---------------------------------------------------------------------------
# Bracketed paste
# ---------------------------------------------------------------------------


class TestBracketedPaste:
    def test_paste_split_across_reads_is_buffered_and_trailing_input_replayed(self):
        editor = make_editor()

        editor.handle_input("\x1b[200~hello")
        assert editor.get_text() == ""  # still buffering

        editor.handle_input(" world\x1b[201~!")
        assert editor.get_text() == "hello world!"

    def test_paste_of_only_control_characters_inserts_nothing(self):
        editor = make_editor()
        changes: list[str] = []
        editor.on_change = changes.append

        editor.handle_input("\x1b[200~\x07\x01\x1b[201~")
        assert editor.get_text() == ""
        assert changes == []

    def test_empty_bracketed_paste_inserts_nothing(self):
        editor = make_editor()
        changes: list[str] = []
        editor.set_text("abc")
        editor.on_change = changes.append

        editor.handle_input("\x1b[200~\x1b[201~")
        assert editor.get_text() == "abc"
        assert changes == []

    def test_paste_decodes_csi_u_control_letters_and_keeps_unknown_sequences(self):
        editor = make_editor()
        # \x1b[74;5u is Ctrl+J (uppercase form) and decodes to a newline;
        # \x1b[300;5u is not a Ctrl+letter, so it is left as-is and only its
        # ESC byte is filtered out.
        editor.handle_input("\x1b[200~a\x1b[74;5ub\x1b[300;5uc\x1b[201~")
        assert editor.get_text() == "a\nb[300;5uc"

    def test_pasted_path_gets_a_separating_space_after_a_word_character(self):
        editor = make_editor()
        for ch in "file":
            editor.handle_input(ch)
        editor.handle_input("\x1b[200~/etc/hosts\x1b[201~")
        assert editor.get_text() == "file /etc/hosts"

    def test_pasted_path_after_a_space_is_not_padded_again(self):
        editor = make_editor()
        for ch in "file ":
            editor.handle_input(ch)
        editor.handle_input("\x1b[200~~/notes.md\x1b[201~")
        assert editor.get_text() == "file ~/notes.md"

    def test_long_single_line_paste_uses_a_character_count_marker(self):
        editor = make_editor()
        pasted = "x" * 1200
        editor.handle_input(f"\x1b[200~{pasted}\x1b[201~")

        assert editor.get_text() == "[paste #1 1200 chars]"
        assert editor.get_expanded_text() == pasted


# ---------------------------------------------------------------------------
# Cursor movement edge cases
# ---------------------------------------------------------------------------


class TestCursorMovementEdges:
    def test_word_left_at_line_start_moves_to_end_of_previous_line(self):
        editor = make_editor()
        editor.set_text("foo\nbar")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 1, "col": 0}

        editor.handle_input("\x1bb")  # alt+b
        assert editor.get_cursor() == {"line": 0, "col": 3}

    def test_word_right_at_line_end_moves_to_start_of_next_line(self):
        editor = make_editor()
        editor.set_text("foo\nbar")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x05")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x1bf")  # alt+f
        assert editor.get_cursor() == {"line": 1, "col": 0}

    def test_left_arrow_at_line_start_wraps_to_previous_line_end(self):
        editor = make_editor()
        editor.set_text("ab\ncd")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 1, "col": 0}

        editor.handle_input("\x1b[D")
        assert editor.get_cursor() == {"line": 0, "col": 2}

    def test_right_arrow_at_line_end_wraps_to_the_next_line_start(self):
        editor = make_editor()
        editor.set_text("ab\ncd")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x05")
        assert editor.get_cursor() == {"line": 0, "col": 2}

        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 1, "col": 0}

    def test_editing_keys_at_the_start_of_an_empty_buffer_are_noops(self):
        editor = make_editor()
        changes: list[str] = []
        editor.on_change = changes.append

        editor.handle_input("\x1b[D")  # left
        editor.handle_input("\x7f")  # backspace
        editor.handle_input("\x15")  # delete to line start
        editor.handle_input("\x1bd")  # delete word forward
        editor.handle_input("\x04")  # forward delete

        assert editor.get_text() == ""
        assert editor.get_cursor() == {"line": 0, "col": 0}
        # Cursor movement emits nothing; each of the four deletions reports the
        # (unchanged) empty text.
        assert changes == ["", "", "", ""]
        assert editor.get_lines() == [""]

    def test_word_movement_at_buffer_edges_is_a_noop(self):
        editor = make_editor()
        editor.set_text("word")
        editor.handle_input("\x01")
        editor.handle_input("\x1bb")
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x05")
        editor.handle_input("\x1bf")
        assert editor.get_cursor() == {"line": 0, "col": 4}

    def test_right_arrow_at_end_of_prompt_sets_the_preferred_column(self):
        editor = make_editor()
        editor.set_text("111111111x1111111111\n\n333333333_")

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x05")
        assert editor.get_cursor() == {"line": 0, "col": 20}

        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 10}

        # Right at the end of the buffer cannot move, but records the visual
        # column so the next vertical move keeps it.
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 2, "col": 10}

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 10}


# ---------------------------------------------------------------------------
# Vertical movement across wrapped / rewrapped lines
# ---------------------------------------------------------------------------


class TestWrappedLineNavigation:
    def test_moves_through_wrapped_visual_lines_without_getting_stuck(self):
        editor = make_editor(cols=15)
        editor.set_text("short\n123456789012345678901234567890")
        editor.render(15)
        assert editor.get_cursor() == {"line": 1, "col": 30}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor()["line"] == 1
        editor.handle_input("\x1b[A")
        assert editor.get_cursor()["line"] == 1
        editor.handle_input("\x1b[A")
        assert editor.get_cursor()["line"] == 0

    def test_resize_clamps_the_preferred_column_on_the_same_line(self):
        editor = make_editor()
        editor.set_text("12345678901234567890\n\n12345678901234567890")

        editor.handle_input("\x01")
        for _ in range(15):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 15}

        editor.render(12)  # narrower: layout width 11

        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor()["col"] == 4

    def test_resize_keeps_the_preferred_column_across_rewraps(self):
        editor = make_editor()
        editor.set_text("short\n12345678901234567890")

        editor.handle_input("\x01")
        for _ in range(15):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 1, "col": 15}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 5}

        editor.render(10)
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 8}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 5}

        editor.render(80)
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 15}

    def test_rewrapped_lines_target_fits_current_visual_column(self):
        editor = make_editor()
        editor.set_text("abcdefghijklmnopqr\n123456789012345678")

        position_cursor(editor, 0, 18)
        assert editor.get_cursor() == {"line": 0, "col": 18}

        editor.render(10)
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 8}

        editor.render(80)
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 8}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 8}

    def test_rewrapped_lines_target_shorter_than_current_visual_column(self):
        editor = make_editor()
        editor.set_text("abcdefghijklmnopqr\n123456789012345678\nab")

        position_cursor(editor, 0, 18)
        assert editor.get_cursor() == {"line": 0, "col": 18}
        editor.render(10)
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 8}

        editor.render(80)
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 2}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 1, "col": 8}


# ---------------------------------------------------------------------------
# Paste markers and vertical navigation
# ---------------------------------------------------------------------------


def _multi_visual_line_marker_editor() -> Editor:
    """Editor whose logical line 0 is `abcdefgh<marker>ijklmnopqr` at width 20.

    Word wrap splits the 21-char marker across two visual lines:
        VL1: "abcdefgh"            (start_col 0,  len 8)
        VL2: "[paste #1 +100"      (start_col 8,  len 15)  <- marker head
        VL3: "lines]ijklmnopqr"    (start_col 23, len 16)  <- marker tail
        VL4: "123456789012345678"  (logical line 1)
    """
    editor = make_editor(cols=20)
    for ch in "abcdefgh":
        editor.handle_input(ch)
    editor.handle_input("\x1b[200~" + ("line\n" * 100).rstrip("\n") + "\x1b[201~")
    for ch in "ijklmnopqr":
        editor.handle_input(ch)
    editor.handle_input("\n")
    for ch in "123456789012345678":
        editor.handle_input(ch)
    editor.render(20)
    return editor


class TestPasteMarkerVerticalNavigation:
    def test_snaps_to_the_marker_start_when_navigating_down_into_it(self):
        editor = make_editor()
        editor.set_text("12345678901234567890\n\nhello ")
        editor.handle_input("\x1b[200~" + "x" * 2000 + "\x1b[201~")
        editor.render(80)
        assert editor.get_lines()[2] == "hello [paste #1 2000 chars]"

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(10):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 10}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 0}

        # Sticky column 10 falls inside the marker (which starts at col 6), so
        # the cursor snaps to the marker start instead of landing inside it.
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 6}

    def test_preserves_the_sticky_column_when_passing_a_marker_line(self):
        editor = make_editor(cols=30)
        for ch in "1234567890123456":
            editor.handle_input(ch)
        editor.handle_input("\n")
        editor.handle_input("\n")
        editor.handle_input("\x1b[200~" + "x" * 2000 + "\x1b[201~")
        editor.handle_input("\n")
        editor.handle_input("\n")
        for ch in "abcdefghijklmnop":
            editor.handle_input(ch)
        editor.render(30)

        for _ in range(4):
            editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(10):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 10}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 0}
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 0}
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 3, "col": 0}
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 4, "col": 10}

    def test_does_not_get_stuck_moving_down_from_a_multi_visual_line_marker(self):
        editor = _multi_visual_line_marker_editor()
        marker = PASTE_RE.search(editor.get_text()).group(0)
        assert visible_width(marker) > 20
        marker_start = 8
        marker_end = marker_start + len(marker)

        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 6}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": marker_start}

        # Preferred column 6 lands past the marker tail on the third visual
        # line, i.e. on "i" right after the marker.
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": marker_end}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": marker_start}
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 6}

    def test_skips_marker_continuation_lines_when_preferred_column_is_in_the_tail(self):
        editor = _multi_visual_line_marker_editor()

        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(3):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": 8}

        # Visual column 3 of the third visual line is inside the marker tail
        # ("lines]"), so that visual line is skipped entirely.
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 3}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 8}
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 3}

    def test_skips_several_marker_continuation_lines_at_once(self):
        # Layout width 8 splits the 21-char marker over three visual lines:
        #   VL0 "abcdef"   (start 0)
        #   VL1 "[paste "  (start 6)   <- marker head
        #   VL2 "#1 +100 " (start 13)  <- marker continuation
        #   VL3 "lines]xy" (start 21)  <- marker continuation + content
        #   VL4 "z"        (start 29)
        editor = make_editor(cols=9)
        for ch in "abcdef":
            editor.handle_input(ch)
        editor.handle_input("\x1b[200~" + ("line\n" * 100).rstrip("\n") + "\x1b[201~")
        for ch in "xyz":
            editor.handle_input(ch)
        editor.handle_input("\n")
        for ch in "123456789012345678":
            editor.handle_input(ch)
        editor.render(9)
        assert len(editor.get_lines()[0]) == 30

        for _ in range(10):
            editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(3):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": 6}  # snapped to marker start

        # All remaining continuation lines of the marker are skipped in one
        # step, landing past the marker on the same logical line.
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": 30}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 3}

    def test_marker_running_to_the_end_of_the_buffer_keeps_the_cursor_snapped(self):
        editor = make_editor(cols=9)
        for ch in "abcdef":
            editor.handle_input(ch)
        editor.handle_input("\x1b[200~" + ("line\n" * 100).rstrip("\n") + "\x1b[201~")
        editor.render(9)
        assert editor.get_lines() == ["abcdef[paste #1 +100 lines]"]

        for _ in range(10):
            editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(3):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": 6}

        # There is nothing after the marker's continuation lines, so the
        # cursor stays snapped to the marker start.
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 0, "col": 6}


class TestPasteMarkerSegmentation:
    def test_marker_like_text_with_unknown_id_stays_navigable(self):
        editor = make_editor()
        editor.handle_input("\x1b[200~" + ("line\n" * 20).rstrip("\n") + "\x1b[201~")
        real_marker = PASTE_RE.search(editor.get_text()).group(0)

        fake = " [paste #9 +5 lines]"
        for ch in fake:
            editor.handle_input(ch)

        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        # The real marker is atomic ...
        assert editor.get_cursor() == {"line": 0, "col": len(real_marker)}
        editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[C")
        # ... while the typed marker with an unregistered id moves per character.
        assert editor.get_cursor() == {"line": 0, "col": len(real_marker) + 2}

    def test_marker_like_text_on_a_line_without_registered_markers(self):
        editor = make_editor()
        fake = "[paste #9 +5 lines]"
        for ch in fake:
            editor.handle_input(ch)
        editor.handle_input("\n")
        editor.handle_input("\x1b[200~" + ("line\n" * 20).rstrip("\n") + "\x1b[201~")

        # Line 1 holds a real marker, so the paste registry is non-empty, but
        # line 0 only contains marker-like text with an unregistered id.
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 1}


# ---------------------------------------------------------------------------
# Word wrap unit edge cases
# ---------------------------------------------------------------------------


class TestWordWrapEdgeCases:
    def test_empty_line_produces_a_single_empty_chunk(self):
        chunks = word_wrap_line("", 10)
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert (chunks[0].start_index, chunks[0].end_index) == (0, 0)

    def test_non_positive_width_produces_a_single_empty_chunk(self):
        chunks = word_wrap_line("hello", 0)
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_breaks_long_whitespace_runs_at_the_line_boundary(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,                         consectetur", 30)
        assert [c.text for c in chunks] == [
            "Lorem ipsum dolor sit ",
            "amet,                         ",
            "consectetur",
        ]

    def test_whitespace_spanning_full_lines_keeps_the_remainder(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,                                     consectetur", 30)
        assert [c.text for c in chunks] == [
            "Lorem ipsum dolor sit ",
            "amet,                         ",
            "            consectetur",
        ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_cursor_inverts_the_character_it_sits_on(self):
        editor = make_editor()
        editor.set_text("abc")
        editor.handle_input("\x01")

        line = content_lines(editor.render(20))[0]
        assert line.startswith("\x1b[7ma\x1b[0m")
        assert strip_terminal_sequences(line).rstrip() == "abc"
        assert visible_width(line) == 20

    def test_cursor_at_end_of_line_renders_an_inverted_space(self):
        editor = make_editor()
        editor.set_text("ab")

        line = content_lines(editor.render(20))[0]
        assert "ab\x1b[7m \x1b[0m" in line
        assert visible_width(line) == 20

    def test_cursor_overflowing_the_content_width_eats_one_padding_column(self):
        editor = Editor(TuiMainScreen(FakeTerminal(columns=80)), EDITOR_THEME, EditorOptions(padding_x=2))
        editor.set_text("x" * 16)  # exactly the content width at render width 20

        rendered = editor.render(20)
        line = content_lines(rendered)[0]
        assert line.startswith("  ") and line.endswith(" ")
        assert "\x1b[7m \x1b[0m" in line
        assert visible_width(line) == 20

    def test_scroll_borders_are_padded_out_to_the_full_width(self):
        editor = make_editor()
        editor.set_text("\n".join(f"line {i}" for i in range(30)))
        editor.render(40)
        for _ in range(10):
            editor.handle_input("\x1b[A")

        rendered = editor.render(40)
        top = strip_terminal_sequences(rendered[0])
        bottom = strip_terminal_sequences(rendered[-1])
        assert top.startswith("─── ↑ 19 more ") and top.endswith("─")
        assert bottom.startswith("─── ↓ 4 more ") and bottom.endswith("─")
        assert visible_width(top) == 40
        assert visible_width(bottom) == 40

    def test_cursor_is_rendered_inside_a_non_final_wrapped_chunk(self):
        editor = make_editor(cols=20)
        editor.set_text("aaaa bbbb cccc dddd eeee")
        editor.render(20)
        editor.handle_input("\x01")
        for _ in range(2):
            editor.handle_input("\x1b[C")

        rendered = content_lines(editor.render(20))
        assert len(rendered) >= 2
        assert "\x1b[7ma\x1b[0m" in rendered[0]
        assert "\x1b[7m" not in rendered[1]

    def test_unfocused_editor_omits_the_hardware_cursor_marker(self):
        from pi_tui.component import CURSOR_MARKER

        editor = make_editor()
        editor.set_text("ab")
        assert CURSOR_MARKER not in "".join(editor.render(20))

        editor.focused = True
        assert CURSOR_MARKER in "".join(editor.render(20))
