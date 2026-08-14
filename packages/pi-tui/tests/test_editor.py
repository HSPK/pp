"""Tests ported from packages/tui/test/editor.test.ts and
packages/tui/test/editor-history-keybindings.test.ts.

Notes on port-specific behavior (vs. the upstream TypeScript tests):

* Strings are indexed by Python code points, not UTF-16 code units. Cursor
  columns are therefore consistent with `len(text)`; for the emoji cases in
  `TestUnicodeTextEditing` this changes the internal indexing but not the
  observable text state that the tests assert on.
* Word segmentation uses the pi-tui ``iter_word_segments`` approximation
  (a whitespace/word-char/punctuation splitter with CJK-per-character
  segmentation). This matches the behavior already relied on by
  ``test_input.py`` and by ``word_navigation``'s docstring.
* Async autocomplete requests are dispatched via ``asyncio.get_running_loop()``.
  When ``handle_input`` is called from a sync test with no running loop, the
  request is simply skipped (matching how the TS editor's ``setTimeout``
  behaves without a debounce fire). Autocomplete-focused tests therefore run
  under ``pytest-asyncio`` and use ``flush_autocomplete()`` to drain pending
  microtasks/timers, mirroring the TS ``flushAutocomplete`` helper.
"""

from __future__ import annotations

import asyncio
import os
import re

from pi_tui.autocomplete import (
    AppliedCompletion,
    AutocompleteItem,
    AutocompleteSuggestions,
    CombinedAutocompleteProvider,
    SlashCommand,
)
from pi_tui.components.editor import (
    Editor,
    EditorOptions,
    EditorTheme,
    _Segment,
    word_wrap_line,
)
from pi_tui.components.select_list import SelectListTheme
from pi_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingsManager,
    set_keybindings,
)
from pi_tui.testing import FakeTerminal, wait_until
from pi_tui.tui_main_screen import TuiMainScreen
from pi_tui.utils import strip_terminal_sequences, visible_width

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(text: str) -> str:
    return text


DEFAULT_SELECT_LIST_THEME = SelectListTheme(
    selected_prefix=_identity,
    selected_text=_identity,
    description=_identity,
    scroll_info=_identity,
    no_match=_identity,
)

DEFAULT_EDITOR_THEME = EditorTheme(
    border_color=_identity,
    select_list=DEFAULT_SELECT_LIST_THEME,
)


def make_editor(cols: int = 80, rows: int = 24, options: EditorOptions | None = None) -> Editor:
    tui = TuiMainScreen(FakeTerminal(columns=cols, rows=rows))
    return Editor(tui, DEFAULT_EDITOR_THEME, options)


async def flush_autocomplete() -> None:
    """Approximates the TS ``await Promise.resolve(); await setImmediate``.

    Autocomplete work is spread across ``call_later(0, ...)`` for debouncing
    and ``create_task`` for chained provider calls, so we need to yield to
    the loop a few times to let everything settle.
    """
    for _ in range(6):
        await asyncio.sleep(0)


def apply_completion_replace_prefix(
    lines: list[str],
    cursor_line: int,
    cursor_col: int,
    item: AutocompleteItem,
    prefix: str,
) -> AppliedCompletion:
    """Reference apply_completion: replace ``prefix`` with ``item.value``.

    Mirrors the TS test helper `applyCompletion`.
    """
    line = lines[cursor_line] if cursor_line < len(lines) else ""
    before = line[: cursor_col - len(prefix)]
    after = line[cursor_col:]
    new_lines = list(lines)
    new_lines[cursor_line] = before + item.value + after
    return AppliedCompletion(
        lines=new_lines,
        cursor_line=cursor_line,
        cursor_col=cursor_col - len(prefix) + len(item.value),
    )


class FakeAutocompleteProvider:
    """Minimal AutocompleteProvider stand-in for tests."""

    def __init__(
        self,
        get_suggestions,
        *,
        trigger_characters: list[str] | None = None,
        apply=apply_completion_replace_prefix,
        should_trigger_file_completion=None,
    ) -> None:
        self.trigger_characters = list(trigger_characters) if trigger_characters else []
        self._get_suggestions = get_suggestions
        self._apply = apply
        self._should_trigger_file_completion = should_trigger_file_completion
        self.call_count = 0

    async def get_suggestions(self, lines, cursor_line, cursor_col, *, signal, force=False):
        self.call_count += 1
        result = self._get_suggestions(lines, cursor_line, cursor_col, force=force)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):
        return self._apply(lines, cursor_line, cursor_col, item, prefix)

    def should_trigger_file_completion(self, lines, cursor_line, cursor_col):
        # Default (no override) matches TS behavior of an absent optional
        # method: force-file completion is allowed to trigger.
        if self._should_trigger_file_completion is None:
            return True
        return self._should_trigger_file_completion(lines, cursor_line, cursor_col)


# ---------------------------------------------------------------------------
# Prompt history navigation
# ---------------------------------------------------------------------------


class TestPromptHistoryNavigation:
    def test_up_arrow_does_nothing_when_history_empty(self):
        editor = make_editor()
        editor.handle_input("\x1b[A")
        assert editor.get_text() == ""

    def test_up_arrow_shows_most_recent_history_when_empty(self):
        editor = make_editor()
        editor.add_to_history("first prompt")
        editor.add_to_history("second prompt")
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "second prompt"

    def test_up_arrow_cycles_through_history(self):
        editor = make_editor()
        editor.add_to_history("first")
        editor.add_to_history("second")
        editor.add_to_history("third")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "third"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "second"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "first"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "first"

    def test_jumps_to_start_before_entering_history_from_non_empty_draft(self):
        editor = make_editor()
        editor.add_to_history("prompt")
        editor.set_text("draft")
        editor.handle_input("\x1b[D")
        editor.handle_input("\x1b[D")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "draft"
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "prompt"

        editor.handle_input("\x1b[B")
        assert editor.get_text() == "draft"
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_down_arrow_restores_draft(self):
        editor = make_editor()
        editor.add_to_history("first")
        editor.set_text("my draft")
        editor.handle_input("\x01")  # go to start
        editor.handle_input("\x1b[A")  # enter history
        assert editor.get_text() == "first"
        editor.handle_input("\x1b[B")
        assert editor.get_text() == "my draft"

    def test_navigates_forward_through_history_with_down_arrow(self):
        editor = make_editor()
        editor.add_to_history("first")
        editor.add_to_history("second")
        editor.add_to_history("third")
        editor.set_text("draft")

        editor.handle_input("\x1b[A")  # start of draft
        editor.handle_input("\x1b[A")  # third
        editor.handle_input("\x1b[A")  # second
        editor.handle_input("\x1b[A")  # first

        editor.handle_input("\x1b[B")
        assert editor.get_text() == "second"
        editor.handle_input("\x1b[B")
        assert editor.get_text() == "third"
        editor.handle_input("\x1b[B")
        assert editor.get_text() == "draft"

    def test_exits_history_mode_when_typing_a_character(self):
        editor = make_editor()
        editor.add_to_history("old prompt")
        editor.handle_input("\x1b[A")
        editor.handle_input("x")
        assert editor.get_text() == "xold prompt"

    def test_exits_history_mode_on_set_text(self):
        editor = make_editor()
        editor.add_to_history("first")
        editor.add_to_history("second")
        editor.handle_input("\x1b[A")
        editor.set_text("")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "second"

    def test_does_not_add_empty_strings_to_history(self):
        editor = make_editor()
        editor.add_to_history("")
        editor.add_to_history("   ")
        editor.add_to_history("valid")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "valid"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "valid"

    def test_does_not_add_consecutive_duplicates_to_history(self):
        editor = make_editor()
        editor.add_to_history("same")
        editor.add_to_history("same")
        editor.add_to_history("same")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "same"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "same"

    def test_allows_non_consecutive_duplicates_in_history(self):
        editor = make_editor()
        editor.add_to_history("first")
        editor.add_to_history("second")
        editor.add_to_history("first")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "first"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "second"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "first"

    def test_uses_cursor_movement_instead_of_history_when_editor_has_content(self):
        editor = make_editor()
        editor.add_to_history("history item")
        editor.set_text("line1\nline2")

        editor.handle_input("\x1b[A")
        editor.handle_input("X")

        assert editor.get_text() == "line1X\nline2"

    def test_limits_history_to_100_entries(self):
        editor = make_editor()
        for i in range(105):
            editor.add_to_history(f"prompt {i}")

        for _ in range(100):
            editor.handle_input("\x1b[A")

        assert editor.get_text() == "prompt 5"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "prompt 5"

    def test_places_cursor_at_start_after_browsing_history_upward(self):
        editor = make_editor()
        editor.add_to_history("older entry")
        editor.add_to_history("line1\nline2\nline3")

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "line1\nline2\nline3"
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "older entry"
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_places_cursor_at_end_after_browsing_history_downward(self):
        editor = make_editor()
        editor.add_to_history("older entry")
        editor.add_to_history("line1\nline2\nline3")
        editor.add_to_history("newer entry")

        editor.handle_input("\x1b[A")  # newer entry
        editor.handle_input("\x1b[A")  # multi-line entry
        editor.handle_input("\x1b[A")  # older entry

        editor.handle_input("\x1b[B")
        assert editor.get_text() == "line1\nline2\nline3"
        assert editor.get_cursor() == {"line": 2, "col": 5}

        editor.handle_input("\x1b[B")
        assert editor.get_text() == "newer entry"

    def test_allows_opposite_direction_cursor_movement_in_multiline_history_entry(self):
        editor = make_editor()
        editor.add_to_history("line1\nline2\nline3")

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x1b[B")
        assert editor.get_text() == "line1\nline2\nline3"
        assert editor.get_cursor() == {"line": 1, "col": 0}

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "line1\nline2\nline3"
        assert editor.get_cursor() == {"line": 0, "col": 0}


# ---------------------------------------------------------------------------
# History keybinding overrides
# ---------------------------------------------------------------------------


class TestHistoryKeybindingsOverride:
    def teardown_method(self, method):
        set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))

    def test_browse_history_directly_without_first_moving_cursor(self):
        set_keybindings(
            KeybindingsManager(
                TUI_KEYBINDINGS,
                {
                    "tui.editor.historyPrevious": "ctrl+p",
                    "tui.editor.historyNext": "ctrl+n",
                },
            )
        )
        editor = make_editor()
        editor.add_to_history("older prompt")
        editor.add_to_history("newer\nmultiline prompt")
        editor.set_text("draft")
        editor.handle_input("\x1b[D")
        editor.handle_input("\x1b[D")

        editor.handle_input("\x10")  # Ctrl+P
        assert editor.get_text() == "newer\nmultiline prompt"
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x10")
        assert editor.get_text() == "older prompt"

        editor.handle_input("\x0e")  # Ctrl+N
        assert editor.get_text() == "newer\nmultiline prompt"
        assert editor.get_cursor() == {"line": 1, "col": 16}

        editor.handle_input("\x0e")
        assert editor.get_text() == "draft"
        assert editor.get_cursor() == {"line": 0, "col": 3}


# ---------------------------------------------------------------------------
# Public state accessors
# ---------------------------------------------------------------------------


class TestPublicStateAccessors:
    def test_returns_cursor_position(self):
        editor = make_editor()
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("a")
        editor.handle_input("b")
        editor.handle_input("c")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x1b[D")
        assert editor.get_cursor() == {"line": 0, "col": 2}

    def test_lines_are_defensive_copy(self):
        editor = make_editor()
        editor.set_text("a\nb")
        lines = editor.get_lines()
        assert lines == ["a", "b"]
        lines[0] = "mutated"
        assert editor.get_lines() == ["a", "b"]


# ---------------------------------------------------------------------------
# Backslash + Enter newline workaround
# ---------------------------------------------------------------------------


class TestBackslashEnterNewline:
    def test_inserts_backslash_immediately(self):
        editor = make_editor()
        editor.handle_input("\\")
        assert editor.get_text() == "\\"

    def test_converts_standalone_backslash_to_newline_on_enter(self):
        editor = make_editor()
        editor.handle_input("\\")
        editor.handle_input("\r")
        assert editor.get_text() == "\n"

    def test_inserts_backslash_normally_when_followed_by_other_chars(self):
        editor = make_editor()
        editor.handle_input("\\")
        editor.handle_input("x")
        assert editor.get_text() == "\\x"

    def test_does_not_trigger_newline_when_backslash_not_at_cursor(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = lambda text: submitted.append(text)

        editor.handle_input("\\")
        editor.handle_input("x")
        editor.handle_input("\r")

        assert len(submitted) == 1

    def test_only_removes_one_backslash_when_multiple(self):
        editor = make_editor()
        editor.handle_input("\\")
        editor.handle_input("\\")
        editor.handle_input("\\")
        assert editor.get_text() == "\\\\\\"

        editor.handle_input("\r")
        assert editor.get_text() == "\\\\\n"


# ---------------------------------------------------------------------------
# Kitty CSI-u handling
# ---------------------------------------------------------------------------


class TestKittyCsiU:
    def test_ignores_printable_csi_u_with_unsupported_modifiers(self):
        editor = make_editor()
        editor.handle_input("\x1b[99;9u")
        assert editor.get_text() == ""

    def test_inserts_shifted_csi_u_letters(self):
        editor = make_editor()
        editor.handle_input("\x1b[69;2u")
        assert editor.get_text() == "E"

    def test_inserts_shifted_xterm_modify_other_keys_letters(self):
        editor = make_editor()
        editor.handle_input("\x1b[27;2;69~")
        assert editor.get_text() == "E"


# ---------------------------------------------------------------------------
# Unicode text editing
# ---------------------------------------------------------------------------


class TestUnicodeTextEditing:
    def test_inserts_mixed_ascii_umlauts_and_emojis(self):
        editor = make_editor()
        for ch in ["H", "e", "l", "l", "o", " ", "ä", "ö", "ü", " ", "😀"]:
            editor.handle_input(ch)
        assert editor.get_text() == "Hello äöü 😀"

    def test_backspace_deletes_umlaut(self):
        editor = make_editor()
        editor.handle_input("ä")
        editor.handle_input("ö")
        editor.handle_input("ü")
        editor.handle_input("\x7f")
        assert editor.get_text() == "äö"

    def test_backspace_deletes_emoji_grapheme(self):
        editor = make_editor()
        editor.handle_input("😀")
        editor.handle_input("👍")
        editor.handle_input("\x7f")
        assert editor.get_text() == "😀"

    def test_inserts_after_cursor_move_over_umlauts(self):
        editor = make_editor()
        editor.handle_input("ä")
        editor.handle_input("ö")
        editor.handle_input("ü")
        editor.handle_input("\x1b[D")
        editor.handle_input("\x1b[D")
        editor.handle_input("x")
        assert editor.get_text() == "äxöü"

    def test_moves_cursor_across_emojis(self):
        editor = make_editor()
        editor.handle_input("😀")
        editor.handle_input("👍")
        editor.handle_input("🎉")
        editor.handle_input("\x1b[D")
        editor.handle_input("\x1b[D")
        editor.handle_input("x")
        assert editor.get_text() == "😀x👍🎉"

    def test_preserves_umlauts_across_line_breaks(self):
        editor = make_editor()
        for ch in ["ä", "ö", "ü", "\n", "Ä", "Ö", "Ü"]:
            editor.handle_input(ch)
        assert editor.get_text() == "äöü\nÄÖÜ"

    def test_set_text_replaces_with_unicode(self):
        editor = make_editor()
        editor.set_text("Hällö Wörld! 😀 äöüÄÖÜß")
        assert editor.get_text() == "Hällö Wörld! 😀 äöüÄÖÜß"

    def test_ctrl_a_moves_to_start_and_inserts(self):
        editor = make_editor()
        editor.handle_input("a")
        editor.handle_input("b")
        editor.handle_input("\x01")
        editor.handle_input("x")
        assert editor.get_text() == "xab"

    def test_ctrl_w_and_alt_backspace_delete_words(self):
        editor = make_editor()

        editor.set_text("foo bar baz")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo bar "

        editor.set_text("foo bar   ")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo "

        editor.set_text("foo bar...")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo bar"

        editor.set_text("foo.bar")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo."

        editor.set_text("foo:bar")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo:"

        editor.set_text("line one\nline two")
        editor.handle_input("\x17")
        assert editor.get_text() == "line one\nline "

        editor.set_text("line one\n")
        editor.handle_input("\x17")
        assert editor.get_text() == "line one"

        editor.set_text("foo 😀😀 bar")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo 😀😀 "
        editor.handle_input("\x17")
        assert editor.get_text() == "foo "

        editor.set_text("foo bar")
        editor.handle_input("\x1b\x7f")  # Alt+Backspace legacy
        assert editor.get_text() == "foo "

    def test_ctrl_left_right_word_navigation(self):
        editor = make_editor()
        editor.set_text("foo bar... baz")

        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 11}
        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 7}
        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 4}

        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 7}
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 10}
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 14}

        editor.set_text("   foo bar")
        editor.handle_input("\x01")
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 6}

        # ASCII punctuation inside word-like segments preserves old boundaries
        editor.set_text("foo.bar baz")
        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 8}
        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 4}
        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x01")
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 3}
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 4}
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 7}

    def test_stops_at_fullwidth_cjk_punctuation(self):
        # Upstream TS relies on Intl.Segmenter's ICU dictionary segmentation,
        # which groups "你好" and "世界" into two-character word segments;
        # this port's dictionary-free `iter_word_segments` (see utils.py)
        # treats each CJK character as its own word, matching the behavior
        # already documented in test_input.py. Ctrl+Left/Ctrl+Right therefore
        # steps one CJK char at a time here.
        editor = make_editor()
        editor.set_text("你好，世界")

        for expected in (4, 3, 2, 1, 0):
            editor.handle_input("\x1b[1;5D")
            assert editor.get_cursor() == {"line": 0, "col": expected}

        for expected in (1, 2, 3, 4, 5):
            editor.handle_input("\x1b[1;5C")
            assert editor.get_cursor() == {"line": 0, "col": expected}

    def test_mixed_cjk_and_ascii_word_movement(self):
        # See CJK note above: each CJK char is its own word segment in this
        # port. ASCII words still move at word granularity.
        editor = make_editor()
        editor.set_text("hello你好，world世界")

        for expected in (14, 13, 8, 7, 6, 5, 0):
            editor.handle_input("\x1b[1;5D")
            assert editor.get_cursor() == {"line": 0, "col": expected}

        for expected in (5, 6, 7, 8, 13, 14, 15):
            editor.handle_input("\x1b[1;5C")
            assert editor.get_cursor() == {"line": 0, "col": expected}


# ---------------------------------------------------------------------------
# Basic multi-line editing
# ---------------------------------------------------------------------------


class TestMultilineEditing:
    def test_enter_creates_newline(self):
        editor = make_editor()
        editor.handle_input("a")
        editor.handle_input("\n")
        editor.handle_input("b")
        assert editor.get_text() == "a\nb"
        assert editor.get_cursor() == {"line": 1, "col": 1}

    def test_backspace_at_line_start_joins_prev_line(self):
        editor = make_editor()
        editor.set_text("a\nb")
        editor.handle_input("\x01")  # start of line 1
        assert editor.get_cursor() == {"line": 1, "col": 0}
        editor.handle_input("\x7f")
        assert editor.get_text() == "ab"
        assert editor.get_cursor() == {"line": 0, "col": 1}

    def test_forward_delete_at_line_end_joins_next_line(self):
        editor = make_editor()
        editor.set_text("a\nb")
        editor.handle_input("\x1b[A")  # up to line 0
        editor.handle_input("\x05")  # end of line
        editor.handle_input("\x1b[3~")  # forward delete
        assert editor.get_text() == "ab"

    def test_vertical_movement_up_down(self):
        editor = make_editor()
        editor.set_text("hello\nworld")
        editor.handle_input("\x1b[A")  # up to line 0 same col (5, clamped to 5)
        assert editor.get_cursor() == {"line": 0, "col": 5}
        editor.handle_input("\x01")  # start
        for _ in range(3):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 3}
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 3}


# ---------------------------------------------------------------------------
# Kill ring
# ---------------------------------------------------------------------------


class TestKillRing:
    def test_ctrl_w_ctrl_y(self):
        editor = make_editor()
        editor.set_text("foo bar baz")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo bar "
        editor.handle_input("\x01")
        editor.handle_input("\x19")
        assert editor.get_text() == "bazfoo bar "

    def test_ctrl_u_saves_to_kill_ring(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x15")
        assert editor.get_text() == "world"
        editor.handle_input("\x19")
        assert editor.get_text() == "hello world"

    def test_ctrl_k_saves_to_kill_ring(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        editor.handle_input("\x0b")
        assert editor.get_text() == ""
        editor.handle_input("\x19")
        assert editor.get_text() == "hello world"

    def test_ctrl_y_noop_when_kill_ring_empty(self):
        editor = make_editor()
        editor.set_text("test")
        editor.handle_input("\x19")
        assert editor.get_text() == "test"

    def test_alt_y_cycles_kill_ring(self):
        editor = make_editor()
        editor.set_text("first")
        editor.handle_input("\x17")
        editor.set_text("second")
        editor.handle_input("\x17")
        editor.set_text("third")
        editor.handle_input("\x17")

        assert editor.get_text() == ""
        editor.handle_input("\x19")
        assert editor.get_text() == "third"
        editor.handle_input("\x1by")
        assert editor.get_text() == "second"
        editor.handle_input("\x1by")
        assert editor.get_text() == "first"
        editor.handle_input("\x1by")
        assert editor.get_text() == "third"

    def test_alt_y_noop_if_not_preceded_by_yank(self):
        editor = make_editor()
        editor.set_text("test")
        editor.handle_input("\x17")
        editor.set_text("other")
        editor.handle_input("x")
        assert editor.get_text() == "otherx"
        editor.handle_input("\x1by")
        assert editor.get_text() == "otherx"

    def test_alt_y_noop_when_ring_has_single_entry(self):
        editor = make_editor()
        editor.set_text("only")
        editor.handle_input("\x17")
        editor.handle_input("\x19")
        assert editor.get_text() == "only"
        editor.handle_input("\x1by")
        assert editor.get_text() == "only"

    def test_consecutive_ctrl_w_accumulates(self):
        editor = make_editor()
        editor.set_text("one two three")
        editor.handle_input("\x17")
        editor.handle_input("\x17")
        editor.handle_input("\x17")
        assert editor.get_text() == ""
        editor.handle_input("\x19")
        assert editor.get_text() == "one two three"

    def test_ctrl_u_multiline_accumulation(self):
        editor = make_editor()
        editor.set_text("line1\nline2\nline3")
        editor.handle_input("\x15")
        assert editor.get_text() == "line1\nline2\n"
        editor.handle_input("\x15")
        assert editor.get_text() == "line1\nline2"
        editor.handle_input("\x15")
        assert editor.get_text() == "line1\n"
        editor.handle_input("\x15")
        assert editor.get_text() == "line1"
        editor.handle_input("\x15")
        assert editor.get_text() == ""
        editor.handle_input("\x19")
        assert editor.get_text() == "line1\nline2\nline3"

    def test_backward_prepend_forward_append_during_accumulation(self):
        editor = make_editor()
        editor.set_text("prefix|suffix")
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x0b")
        editor.handle_input("\x0b")
        assert editor.get_text() == "prefix"
        editor.handle_input("\x19")
        assert editor.get_text() == "prefix|suffix"

    def test_non_delete_breaks_kill_accumulation(self):
        editor = make_editor()
        editor.set_text("foo bar baz")
        editor.handle_input("\x17")
        assert editor.get_text() == "foo bar "
        editor.handle_input("x")
        assert editor.get_text() == "foo bar x"
        editor.handle_input("\x17")
        assert editor.get_text() == "foo bar "
        editor.handle_input("\x19")
        assert editor.get_text() == "foo bar x"
        editor.handle_input("\x1by")
        assert editor.get_text() == "foo bar baz"

    def test_non_yank_breaks_alt_y_chain(self):
        editor = make_editor()
        editor.set_text("first")
        editor.handle_input("\x17")
        editor.set_text("second")
        editor.handle_input("\x17")
        editor.set_text("")

        editor.handle_input("\x19")
        assert editor.get_text() == "second"
        editor.handle_input("x")
        assert editor.get_text() == "secondx"
        editor.handle_input("\x1by")
        assert editor.get_text() == "secondx"

    def test_kill_ring_rotation_persists(self):
        editor = make_editor()
        editor.set_text("first")
        editor.handle_input("\x17")
        editor.set_text("second")
        editor.handle_input("\x17")
        editor.set_text("third")
        editor.handle_input("\x17")
        editor.set_text("")

        editor.handle_input("\x19")
        editor.handle_input("\x1by")
        assert editor.get_text() == "second"
        editor.handle_input("x")
        editor.set_text("")
        editor.handle_input("\x19")
        assert editor.get_text() == "second"

    def test_deletions_across_lines_coalesce(self):
        editor = make_editor()
        editor.set_text("1\n2\n3")
        editor.handle_input("\x17")
        assert editor.get_text() == "1\n2\n"
        editor.handle_input("\x17")
        assert editor.get_text() == "1\n2"
        editor.handle_input("\x17")
        assert editor.get_text() == "1\n"
        editor.handle_input("\x17")
        assert editor.get_text() == "1"
        editor.handle_input("\x17")
        assert editor.get_text() == ""
        editor.handle_input("\x19")
        assert editor.get_text() == "1\n2\n3"

    def test_ctrl_k_at_line_end_deletes_newline_and_coalesces(self):
        editor = make_editor()
        editor.set_text("")
        for ch in ["a", "b", "\n", "c", "d"]:
            editor.handle_input(ch)
        editor.handle_input("\x1b[A")
        editor.handle_input("\x05")
        editor.handle_input("\x0b")
        assert editor.get_text() == "abcd"
        editor.handle_input("\x0b")
        assert editor.get_text() == "ab"
        editor.handle_input("\x19")
        assert editor.get_text() == "ab\ncd"

    def test_yank_in_middle_of_text(self):
        editor = make_editor()
        editor.set_text("word")
        editor.handle_input("\x17")
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x19")
        assert editor.get_text() == "hello wordworld"

    def test_yank_pop_in_middle_of_text(self):
        editor = make_editor()
        editor.set_text("FIRST")
        editor.handle_input("\x17")
        editor.set_text("SECOND")
        editor.handle_input("\x17")

        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x19")
        assert editor.get_text() == "hello SECONDworld"
        editor.handle_input("\x1by")
        assert editor.get_text() == "hello FIRSTworld"

    def test_multiline_yank_and_yank_pop(self):
        editor = make_editor()
        editor.set_text("SINGLE")
        editor.handle_input("\x17")

        editor.set_text("A\nB")
        editor.handle_input("\x15")
        editor.handle_input("\x15")
        editor.handle_input("\x15")

        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x19")
        assert editor.get_text() == "hello A\nBworld"
        editor.handle_input("\x1by")
        assert editor.get_text() == "hello SINGLEworld"

    def test_alt_d_deletes_word_forward(self):
        editor = make_editor()
        editor.set_text("hello world test")
        editor.handle_input("\x01")

        editor.handle_input("\x1bd")
        assert editor.get_text() == " world test"
        editor.handle_input("\x1bd")
        assert editor.get_text() == " test"
        editor.handle_input("\x19")
        assert editor.get_text() == "hello world test"

    def test_alt_d_at_line_end_deletes_newline(self):
        editor = make_editor()
        editor.set_text("line1\nline2")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x05")
        editor.handle_input("\x1bd")
        assert editor.get_text() == "line1line2"
        editor.handle_input("\x19")
        assert editor.get_text() == "line1\nline2"


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

UNDO = "\x1b[45;5u"


class TestUndo:
    def test_noop_when_stack_empty(self):
        editor = make_editor()
        editor.handle_input(UNDO)
        assert editor.get_text() == ""

    def test_coalesces_consecutive_word_chars(self):
        editor = make_editor()
        for ch in "hello world":
            editor.handle_input(ch)
        assert editor.get_text() == "hello world"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello"
        editor.handle_input(UNDO)
        assert editor.get_text() == ""

    def test_undoes_spaces_one_at_a_time(self):
        editor = make_editor()
        for ch in "hello  ":
            editor.handle_input(ch)
        assert editor.get_text() == "hello  "
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello "
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello"
        editor.handle_input(UNDO)
        assert editor.get_text() == ""

    def test_undoes_newlines(self):
        editor = make_editor()
        for ch in "hello":
            editor.handle_input(ch)
        editor.handle_input("\n")
        for ch in "world":
            editor.handle_input(ch)
        assert editor.get_text() == "hello\nworld"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello\n"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello"
        editor.handle_input(UNDO)
        assert editor.get_text() == ""

    def test_undoes_backspace(self):
        editor = make_editor()
        for ch in "hello":
            editor.handle_input(ch)
        editor.handle_input("\x7f")
        assert editor.get_text() == "hell"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello"

    def test_undoes_forward_delete(self):
        editor = make_editor()
        for ch in "hello":
            editor.handle_input(ch)
        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[3~")
        assert editor.get_text() == "hllo"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello"

    def test_undoes_ctrl_w(self):
        editor = make_editor()
        for ch in "hello world":
            editor.handle_input(ch)
        editor.handle_input("\x17")
        assert editor.get_text() == "hello "
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"

    def test_undoes_ctrl_k(self):
        editor = make_editor()
        for ch in "hello world":
            editor.handle_input(ch)
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x0b")
        assert editor.get_text() == "hello "
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"
        editor.handle_input("|")
        assert editor.get_text() == "hello |world"

    def test_undoes_ctrl_u(self):
        editor = make_editor()
        for ch in "hello world":
            editor.handle_input(ch)
        editor.handle_input("\x01")
        for _ in range(6):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x15")
        assert editor.get_text() == "world"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"

    def test_undoes_yank(self):
        editor = make_editor()
        for ch in "hello ":
            editor.handle_input(ch)
        editor.handle_input("\x17")
        editor.handle_input("\x19")
        assert editor.get_text() == "hello "
        editor.handle_input(UNDO)
        assert editor.get_text() == ""

    def test_undoes_single_line_paste_atomically(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(5):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x1b[200~beep boop\x1b[201~")
        assert editor.get_text() == "hellobeep boop world"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"
        editor.handle_input("|")
        assert editor.get_text() == "hello| world"

    def test_undoes_multi_line_paste_atomically(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(5):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x1b[200~line1\nline2\nline3\x1b[201~")
        assert editor.get_text() == "helloline1\nline2\nline3 world"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"
        editor.handle_input("|")
        assert editor.get_text() == "hello| world"

    def test_undoes_insert_text_at_cursor_atomically(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(5):
            editor.handle_input("\x1b[C")

        editor.insert_text_at_cursor("/path/image.png")
        assert editor.get_text() == "hello/path/image.png world"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"
        editor.handle_input("|")
        assert editor.get_text() == "hello| world"

    def test_insert_text_at_cursor_handles_multiline(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(5):
            editor.handle_input("\x1b[C")

        editor.insert_text_at_cursor("line1\nline2\nline3")
        assert editor.get_text() == "helloline1\nline2\nline3 world"
        cursor = editor.get_cursor()
        assert cursor["line"] == 2
        assert cursor["col"] == 5
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"

    def test_insert_text_at_cursor_normalizes_crlf_and_cr(self):
        editor = make_editor()

        editor.insert_text_at_cursor("a\r\nb\r\nc")
        assert editor.get_text() == "a\nb\nc"
        editor.handle_input(UNDO)
        assert editor.get_text() == ""
        editor.insert_text_at_cursor("x\ry\rz")
        assert editor.get_text() == "x\ny\nz"

    def test_undoes_set_text_to_empty(self):
        editor = make_editor()
        for ch in "hello world":
            editor.handle_input(ch)
        editor.set_text("")
        assert editor.get_text() == ""
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"

    def test_clears_undo_stack_on_submit(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = lambda t: submitted.append(t)
        for ch in "hello":
            editor.handle_input(ch)
        editor.handle_input("\r")
        assert submitted == ["hello"]
        assert editor.get_text() == ""
        editor.handle_input(UNDO)
        assert editor.get_text() == ""

    def test_exits_history_browsing_on_undo(self):
        editor = make_editor()
        editor.add_to_history("hello")

        for ch in "world":
            editor.handle_input(ch)
        editor.handle_input("\x17")
        assert editor.get_text() == ""
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "hello"
        editor.handle_input(UNDO)
        assert editor.get_text() == ""
        editor.handle_input(UNDO)
        assert editor.get_text() == "world"

    def test_undo_restores_pre_history_state_after_multiple_navigations(self):
        editor = make_editor()
        editor.add_to_history("first")
        editor.add_to_history("second")
        editor.add_to_history("third")

        for ch in "current":
            editor.handle_input(ch)
        assert editor.get_text() == "current"

        editor.handle_input("\x17")  # Ctrl+W
        assert editor.get_text() == ""

        editor.handle_input("\x1b[A")
        assert editor.get_text() == "third"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "second"
        editor.handle_input("\x1b[A")
        assert editor.get_text() == "first"

        editor.handle_input(UNDO)
        assert editor.get_text() == ""
        editor.handle_input(UNDO)
        assert editor.get_text() == "current"

    def test_cursor_movement_starts_new_undo_unit(self):
        editor = make_editor()
        for ch in "hello world":
            editor.handle_input(ch)
        for _ in range(5):
            editor.handle_input("\x1b[D")
        for ch in "lol":
            editor.handle_input(ch)
        assert editor.get_text() == "hello lolworld"
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello world"
        editor.handle_input("|")
        assert editor.get_text() == "hello |world"

    def test_noop_delete_does_not_push_undo(self):
        editor = make_editor()
        for ch in "hello":
            editor.handle_input(ch)
        editor.handle_input("\x17")
        assert editor.get_text() == ""
        editor.handle_input("\x17")
        editor.handle_input("\x17")
        editor.handle_input(UNDO)
        assert editor.get_text() == "hello"

    def test_decodes_csi_u_ctrl_letter_inside_bracketed_paste(self):
        editor = make_editor()
        editor.handle_input("\x1b[200~line1\x1b[106;5uline2\x1b[106;5uline3\x1b[201~")
        assert editor.get_text() == "line1\nline2\nline3"


# ---------------------------------------------------------------------------
# Word wrapping (unit + rendered)
# ---------------------------------------------------------------------------


class TestWordWrapLineUnit:
    def test_wraps_word_to_next_line_when_ends_at_width(self):
        chunks = word_wrap_line("hello world test", 11)
        assert len(chunks) == 2
        assert chunks[0].text == "hello "
        assert chunks[1].text == "world test"

    def test_keeps_whitespace_at_width_boundary(self):
        chunks = word_wrap_line("hello world test", 12)
        assert len(chunks) == 2
        assert chunks[0].text == "hello world "
        assert chunks[1].text == "test"

    def test_unbreakable_word_filling_width_followed_by_space(self):
        chunks = word_wrap_line("aaaaaaaaaaaa aaaa", 12)
        assert len(chunks) == 2
        assert chunks[0].text == "aaaaaaaaaaaa"
        assert chunks[1].text == " aaaa"

    def test_wraps_when_word_fits_width_but_not_remaining(self):
        chunks = word_wrap_line("      aaaaaaaaaaaa", 12)
        assert len(chunks) == 2
        assert chunks[0].text == "      "
        assert chunks[1].text == "aaaaaaaaaaaa"

    def test_keeps_word_with_multispace_and_following_word_when_fits(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,    consectetur", 30)
        assert len(chunks) == 2
        assert chunks[0].text == "Lorem ipsum dolor sit "
        assert chunks[1].text == "amet,    consectetur"

    def test_splits_when_word_plus_multispace_plus_word_exceeds_width(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,               consectetur", 30)
        assert len(chunks) == 3
        assert chunks[0].text == "Lorem ipsum dolor sit "
        assert chunks[1].text == "amet,               "
        assert chunks[2].text == "consectetur"

    def test_force_break_when_wide_char_after_word_boundary_overflows(self):
        line = " " + "a" * 186 + "你"
        chunks = word_wrap_line(line, 187)
        for chunk in chunks:
            assert visible_width(chunk.text) <= 187
        reconstructed = "".join(line[c.start_index : c.end_index] for c in chunks)
        assert reconstructed == line

    def test_reconstruction_preserves_content_when_line_fits(self):
        chunks = word_wrap_line("short", 40)
        assert len(chunks) == 1
        assert chunks[0].text == "short"

    def test_keeps_word_with_multispace_and_following_word_when_fills_width_exactly(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,              consectetur", 30)
        assert len(chunks) == 2
        assert chunks[0].text == "Lorem ipsum dolor sit "
        assert chunks[1].text == "amet,              consectetur"

    def test_breaks_long_whitespace_at_line_boundary(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,                         consectetur", 30)
        assert len(chunks) == 3
        assert chunks[0].text == "Lorem ipsum dolor sit "
        assert chunks[1].text == "amet,                         "
        assert chunks[2].text == "consectetur"

    def test_breaks_long_whitespace_at_line_boundary_2(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,                          consectetur", 30)
        assert len(chunks) == 3
        assert chunks[0].text == "Lorem ipsum dolor sit "
        assert chunks[1].text == "amet,                         "
        assert chunks[2].text == " consectetur"

    def test_breaks_whitespace_spanning_full_lines(self):
        chunks = word_wrap_line("Lorem ipsum dolor sit amet,                                     consectetur", 30)
        assert len(chunks) == 3
        assert chunks[0].text == "Lorem ipsum dolor sit "
        assert chunks[1].text == "amet,                         "
        assert chunks[2].text == "            consectetur"

    def test_splits_oversized_atomic_segment_across_multiple_chunks(self):
        marker = "[paste #1 +20 lines]"
        line = f"A{marker}B"
        segments = [
            _Segment(segment="A", index=0),
            _Segment(segment=marker, index=1),
            _Segment(segment="B", index=1 + len(marker)),
        ]
        chunks = word_wrap_line(line, 10, segments)
        for chunk in chunks:
            assert visible_width(chunk.text) <= 10
        reconstructed = "".join(line[c.start_index : c.end_index] for c in chunks)
        assert reconstructed == line

    def test_splits_oversized_atomic_segment_at_start_of_line(self):
        marker = "[paste #1 +20 lines]"
        line = f"{marker}B"
        segments = [
            _Segment(segment=marker, index=0),
            _Segment(segment="B", index=len(marker)),
        ]
        chunks = word_wrap_line(line, 10, segments)
        for chunk in chunks:
            assert visible_width(chunk.text) <= 10
        assert "B" in chunks[-1].text
        reconstructed = "".join(line[c.start_index : c.end_index] for c in chunks)
        assert reconstructed == line

    def test_splits_oversized_atomic_segment_at_end_of_line(self):
        marker = "[paste #1 +20 lines]"
        line = f"A{marker}"
        segments = [
            _Segment(segment="A", index=0),
            _Segment(segment=marker, index=1),
        ]
        chunks = word_wrap_line(line, 10, segments)
        for chunk in chunks:
            assert visible_width(chunk.text) <= 10
        assert chunks[0].text == "A"
        reconstructed = "".join(line[c.start_index : c.end_index] for c in chunks)
        assert reconstructed == line

    def test_splits_consecutive_oversized_atomic_segments(self):
        m1 = "[paste #1 +20 lines]"
        m2 = "[paste #2 +30 lines]"
        line = f"{m1}{m2}"
        segments = [
            _Segment(segment=m1, index=0),
            _Segment(segment=m2, index=len(m1)),
        ]
        chunks = word_wrap_line(line, 10, segments)
        for chunk in chunks:
            assert visible_width(chunk.text) <= 10
        reconstructed = "".join(line[c.start_index : c.end_index] for c in chunks)
        assert reconstructed == line

    def test_wraps_normally_after_oversized_atomic_segment(self):
        marker = "[paste #1 +20 lines]"
        line = f"{marker} hello world"
        segments = [_Segment(segment=marker, index=0)]
        for offset, char in enumerate(" hello world"):
            segments.append(_Segment(segment=char, index=len(marker) + offset))

        chunks = word_wrap_line(line, 10, segments)
        for chunk in chunks:
            assert visible_width(chunk.text) <= 10
        assert chunks[-1].text == "world"
        reconstructed = "".join(line[c.start_index : c.end_index] for c in chunks)
        assert reconstructed == line


# ---------------------------------------------------------------------------
# Rendered wrapping and scroll indicators
# ---------------------------------------------------------------------------


def _content_lines(lines: list[str]) -> list[str]:
    return lines[1:-1]


class TestGraphemeAwareWrapping:
    def test_wide_emojis_fit_within_width(self):
        editor = make_editor()
        editor.set_text("Hello ✅ World")
        lines = editor.render(20)
        for line in _content_lines(lines):
            assert visible_width(line) == 20

    def test_long_text_with_emojis_wraps(self):
        editor = make_editor()
        editor.set_text("✅✅✅✅✅✅")
        lines = editor.render(10)
        for line in _content_lines(lines):
            assert visible_width(line) == 10

    def test_renders_isolated_thai_and_lao_am_clusters_without_width_drift(self):
        for text in ["ำabc", "ຳabc"]:
            editor = make_editor()
            width = 8
            editor.set_text(text)
            for line in editor.render(width):
                assert visible_width(line) == width, f"line width drift for {text!r}: {line}"

    def test_wraps_cjk_chars(self):
        editor = make_editor()
        editor.set_text("日本語テスト")
        width = 11  # +1 col reserved for cursor
        lines = editor.render(width)
        for line in _content_lines(lines):
            assert visible_width(line) == width
        content = [strip_terminal_sequences(line).rstrip() for line in _content_lines(lines)]
        assert content[0] == "日本語テス"
        assert content[1] == "ト"

    def test_cursor_at_end_before_wrap_wraps_on_next_char(self):
        width = 10
        for padding_x in (0, 1):
            editor = make_editor(cols=width + padding_x, options=EditorOptions(padding_x=padding_x))
            for _ in range(9):
                editor.handle_input("a")
            lines = editor.render(width + padding_x)
            content = _content_lines(lines)
            assert len(content) == 1
            assert content[0].endswith("\x1b[7m \x1b[0m")

            editor.handle_input("a")
            lines = editor.render(width + padding_x)
            content = _content_lines(lines)
            assert len(content) == 2

    def test_handles_mixed_ascii_and_wide_chars_in_wrapping(self):
        editor = make_editor()
        width = 15 + 1  # +1 col reserved for cursor
        editor.set_text("Test ✅ OK 日本")
        content = _content_lines(editor.render(width))
        assert len(content) == 1
        assert visible_width(content[0]) == width

    def test_renders_cursor_correctly_on_wide_characters(self):
        editor = make_editor()
        width = 20
        editor.set_text("A✅B")
        lines = editor.render(width)
        content_line = lines[1]
        assert "\x1b[7m" in content_line
        assert visible_width(content_line) == width

    def test_does_not_exceed_terminal_width_with_emoji_at_wrap_boundary(self):
        editor = make_editor()
        width = 11
        editor.set_text("0123456789✅")
        for line in _content_lines(editor.render(width)):
            assert visible_width(line) <= width


class TestScrollIndicators:
    def test_truncated_scroll_indicators_within_width(self):
        width = 10

        def border(text: str) -> str:
            return "\x1b[35m" + text + "\x1b[39m"

        theme = EditorTheme(border_color=border, select_list=DEFAULT_SELECT_LIST_THEME)
        editor = Editor(TuiMainScreen(FakeTerminal(columns=width)), theme)
        editor.set_text("\n".join(f"line {i}" for i in range(20)))

        editor.render(width)
        for _ in range(10):
            editor.handle_input("\x1b[A")

        lines = editor.render(width)
        top = lines[0]
        bottom = lines[-1]
        assert re.match(r"^─── ↑", strip_terminal_sequences(top))
        assert re.match(r"^─── ↓", strip_terminal_sequences(bottom))
        assert top == border(strip_terminal_sequences(top))
        assert bottom == border(strip_terminal_sequences(bottom))
        for line in lines:
            assert visible_width(line) == width


class TestWordWrappingRendered:
    def test_wraps_at_word_boundaries_instead_of_mid_word(self):
        editor = make_editor()
        editor.set_text("Hello world this is a test of word wrapping functionality")
        lines = editor.render(40)
        content = [strip_terminal_sequences(line).strip() for line in _content_lines(lines)]

        assert not content[0].endswith("-")
        for line in content:
            last_char = line.rstrip()[-1:]
            assert last_char == "" or re.match(r"[\w.,!?;:]", last_char)

    def test_does_not_start_lines_with_leading_whitespace_after_word_wrap(self):
        editor = make_editor()
        editor.set_text("Word1 Word2 Word3 Word4 Word5 Word6")
        lines = editor.render(20)
        for content_line in _content_lines(lines):
            line = strip_terminal_sequences(content_line)
            if line.lstrip():
                assert not re.match(r"^\s+\S", line.rstrip())

    def test_breaks_long_words_urls_at_character_level(self):
        editor = make_editor()
        width = 30
        editor.set_text("Check https://example.com/very/long/path/that/exceeds/width here")
        for line in _content_lines(editor.render(width)):
            assert visible_width(line) == width

    def test_preserves_multiple_spaces_within_words_on_same_line(self):
        editor = make_editor()
        editor.set_text("Word1   Word2    Word3")
        lines = editor.render(50)
        content_line = strip_terminal_sequences(lines[1]).strip()
        assert "Word1   Word2" in content_line

    def test_empty_string_renders_border_only(self):
        editor = make_editor()
        editor.set_text("")
        lines = editor.render(40)
        assert len(lines) == 3

    def test_single_word_fits_exactly(self):
        editor = make_editor()
        editor.set_text("1234567890")
        lines = editor.render(11)  # +1 for cursor
        assert len(lines) == 3
        content = strip_terminal_sequences(lines[1])
        assert "1234567890" in content


# ---------------------------------------------------------------------------
# Character jump (Ctrl+])
# ---------------------------------------------------------------------------


class TestCharacterJump:
    def test_jump_forward_to_first_char(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("o")
        assert editor.get_cursor() == {"line": 0, "col": 4}

    def test_jump_forward_to_next_after_cursor(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        for _ in range(4):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 4}
        editor.handle_input("\x1d")
        editor.handle_input("o")
        assert editor.get_cursor() == {"line": 0, "col": 7}

    def test_jump_forward_across_lines(self):
        editor = make_editor()
        editor.set_text("abc\ndef\nghi")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("g")
        assert editor.get_cursor() == {"line": 2, "col": 0}

    def test_jump_backward_same_line(self):
        editor = make_editor()
        editor.set_text("hello world")
        assert editor.get_cursor() == {"line": 0, "col": 11}
        editor.handle_input("\x1b\x1d")
        editor.handle_input("o")
        assert editor.get_cursor() == {"line": 0, "col": 7}

    def test_jump_backward_across_lines(self):
        editor = make_editor()
        editor.set_text("abc\ndef\nghi")
        assert editor.get_cursor() == {"line": 2, "col": 3}
        editor.handle_input("\x1b\x1d")
        editor.handle_input("a")
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_not_found_forward_leaves_cursor(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("z")
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_not_found_backward_leaves_cursor(self):
        editor = make_editor()
        editor.set_text("hello world")
        assert editor.get_cursor() == {"line": 0, "col": 11}
        editor.handle_input("\x1b\x1d")
        editor.handle_input("z")
        assert editor.get_cursor() == {"line": 0, "col": 11}

    def test_case_sensitive(self):
        editor = make_editor()
        editor.set_text("Hello World")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("h")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("W")
        assert editor.get_cursor() == {"line": 0, "col": 6}

    def test_cancel_jump_mode_by_repeat_ctrl_bracket(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("\x1d")
        editor.handle_input("o")
        assert editor.get_text() == "ohello world"

    def test_cancel_backward_jump_by_repeat(self):
        editor = make_editor()
        editor.set_text("hello world")
        assert editor.get_cursor() == {"line": 0, "col": 11}
        editor.handle_input("\x1b\x1d")
        editor.handle_input("\x1b\x1d")
        editor.handle_input("o")
        assert editor.get_text() == "hello worldo"

    def test_searches_for_special_chars(self):
        editor = make_editor()
        editor.set_text("foo(bar) = baz;")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("(")
        assert editor.get_cursor() == {"line": 0, "col": 3}
        editor.handle_input("\x1d")
        editor.handle_input("=")
        assert editor.get_cursor() == {"line": 0, "col": 9}

    def test_empty_text_graceful(self):
        editor = make_editor()
        editor.set_text("")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1d")
        editor.handle_input("x")
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_cancels_jump_mode_on_escape_and_processes_the_escape(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x1d")  # Ctrl+] enters jump mode
        editor.handle_input("\x1b")  # Escape cancels it

        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("o")
        assert editor.get_text() == "ohello world"

    def test_resets_last_action_when_jumping(self):
        editor = make_editor()
        editor.set_text("hello world")
        editor.handle_input("\x01")

        editor.handle_input("x")
        assert editor.get_text() == "xhello world"

        editor.handle_input("\x1d")
        editor.handle_input("o")

        editor.handle_input("Y")
        assert editor.get_text() == "xhellYo world"

        editor.handle_input(UNDO)
        assert editor.get_text() == "xhello world"


# ---------------------------------------------------------------------------
# Sticky column (vertical movement preferred column)
# ---------------------------------------------------------------------------


class TestStickyColumn:
    def test_preserves_target_column_when_moving_up_through_shorter_line(self):
        editor = make_editor()
        editor.set_text("2222222222x222\n\n1111111111_111111111111")
        assert editor.get_cursor() == {"line": 2, "col": 23}
        editor.handle_input("\x01")
        for _ in range(10):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 2, "col": 10}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 1, "col": 0}
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 10}

    def test_preserves_target_column_when_moving_down_through_shorter_line(self):
        editor = make_editor()
        editor.set_text("1111111111_111\n\n2222222222x222222222222")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(10):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 10}
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 1, "col": 0}
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 10}

    def test_resets_sticky_on_left_arrow(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x01")
        for _ in range(5):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 2, "col": 5}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 1, "col": 0}
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 5}

        editor.handle_input("\x1b[D")
        assert editor.get_cursor() == {"line": 0, "col": 4}
        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 4}

    def test_resets_sticky_on_typing(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x01")
        for _ in range(8):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 8}

        editor.handle_input("X")
        assert editor.get_cursor() == {"line": 0, "col": 9}
        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 9}

    def test_resets_sticky_on_right_arrow(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(5):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 5}

        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 5}

        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 2, "col": 6}
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 6}

    def test_resets_sticky_on_backspace(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x01")
        for _ in range(8):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 8}

        editor.handle_input("\x7f")
        assert editor.get_cursor() == {"line": 0, "col": 7}
        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 7}

    def test_resets_sticky_on_ctrl_a(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x01")
        for _ in range(8):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 1, "col": 0}

        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_resets_sticky_on_ctrl_e(self):
        editor = make_editor()
        editor.set_text("12345\n\n1234567890")
        editor.handle_input("\x01")
        for _ in range(3):
            editor.handle_input("\x1b[C")

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 3}

        editor.handle_input("\x05")
        assert editor.get_cursor() == {"line": 0, "col": 5}
        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 5}

    def test_resets_sticky_on_ctrl_left_word_movement(self):
        editor = make_editor()
        editor.set_text("hello world\n\nhello world")
        assert editor.get_cursor() == {"line": 2, "col": 11}

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 11}

        editor.handle_input("\x1b[1;5D")
        assert editor.get_cursor() == {"line": 0, "col": 6}
        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 6}

    def test_resets_sticky_on_ctrl_right_word_movement(self):
        editor = make_editor()
        editor.set_text("hello world\n\nhello world")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}

        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 0}

        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 2, "col": 5}
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 5}

    def test_resets_sticky_on_undo(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        editor.handle_input("\x01")
        for _ in range(8):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 8}

        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 2, "col": 8}

        editor.handle_input("X")
        assert editor.get_text() == "1234567890\n\n12345678X90"
        assert editor.get_cursor() == {"line": 2, "col": 9}

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 9}

        editor.handle_input(UNDO)
        assert editor.get_text() == "1234567890\n\n1234567890"
        assert editor.get_cursor() == {"line": 2, "col": 8}

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 8}

    def test_handles_set_text_resetting_sticky_column(self):
        editor = make_editor()
        editor.set_text("1234567890\n\n1234567890")
        editor.handle_input("\x01")
        for _ in range(8):
            editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[A")

        editor.set_text("abcdefghij\n\nabcdefghij")
        assert editor.get_cursor() == {"line": 2, "col": 10}

        editor.handle_input("\x1b[A")
        editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 10}

    def test_multiple_consecutive_up_downs_preserve_sticky(self):
        editor = make_editor()
        editor.set_text("1234567890\nab\ncd\nef\n1234567890")
        editor.handle_input("\x01")
        for _ in range(7):
            editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 4, "col": 7}

        for _ in range(4):
            editor.handle_input("\x1b[A")
        assert editor.get_cursor() == {"line": 0, "col": 7}

        for _ in range(4):
            editor.handle_input("\x1b[B")
        assert editor.get_cursor() == {"line": 4, "col": 7}


# ---------------------------------------------------------------------------
# Paste marker atomic behavior
# ---------------------------------------------------------------------------


def _paste_with_marker(editor: Editor) -> str:
    big = ("line\n" * 20).rstrip("\n")
    editor.handle_input(f"\x1b[200~{big}\x1b[201~")
    return editor.get_text()


def _big_paste(tag: str, n: int = 12) -> str:
    return "\n".join(f"{tag}{i}" for i in range(n))


PASTE_RE = re.compile(r"\[paste #\d+ \+\d+ lines\]")


class TestPasteMarkerAtomic:
    def test_creates_paste_marker_for_large_pastes(self):
        editor = make_editor()
        text = _paste_with_marker(editor)
        assert PASTE_RE.search(text)

    def test_treats_marker_as_single_unit_for_right_arrow(self):
        editor = make_editor()
        editor.handle_input("A")
        _paste_with_marker(editor)
        editor.handle_input("B")
        editor.handle_input("\x01")
        assert editor.get_cursor() == {"line": 0, "col": 0}
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 1}
        marker = PASTE_RE.search(editor.get_text()).group(0)
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker) + 1}

    def test_treats_marker_as_single_unit_for_left_arrow(self):
        editor = make_editor()
        editor.handle_input("A")
        _paste_with_marker(editor)
        editor.handle_input("B")
        marker = PASTE_RE.search(editor.get_text()).group(0)

        editor.handle_input("\x1b[D")
        assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}
        editor.handle_input("\x1b[D")
        assert editor.get_cursor() == {"line": 0, "col": 1}
        editor.handle_input("\x1b[D")
        assert editor.get_cursor() == {"line": 0, "col": 0}

    def test_treats_marker_as_single_unit_for_backspace(self):
        editor = make_editor()
        editor.handle_input("A")
        _paste_with_marker(editor)
        editor.handle_input("B")
        marker = PASTE_RE.search(editor.get_text()).group(0)

        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}
        editor.handle_input("\x7f")
        assert editor.get_text() == "AB"
        assert editor.get_cursor() == {"line": 0, "col": 1}

    def test_treats_marker_as_single_unit_for_forward_delete(self):
        editor = make_editor()
        editor.handle_input("A")
        _paste_with_marker(editor)
        editor.handle_input("B")

        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[3~")
        assert editor.get_text() == "AB"
        assert editor.get_cursor() == {"line": 0, "col": 1}

    def test_undo_restores_marker_after_backspace(self):
        editor = make_editor()
        editor.handle_input("A")
        _paste_with_marker(editor)
        editor.handle_input("B")
        text_before = editor.get_text()

        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        editor.handle_input("\x1b[C")
        editor.handle_input("\x7f")
        assert editor.get_text() == "AB"

        editor.handle_input(UNDO)
        assert editor.get_text() == text_before

    def test_undo_after_marker_deletion_restores_registry(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = lambda t: submitted.append(t)

        paste = _big_paste("alpha")
        editor.handle_input(f"\x1b[200~{paste}\x1b[201~")
        editor.handle_input("\x7f")
        editor.handle_input(UNDO)
        editor.handle_input("\r")
        assert submitted == [paste]

    def test_undo_after_deleting_first_of_two_markers_restores_registry(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = lambda t: submitted.append(t)

        paste_a = _big_paste("alpha")
        paste_b = _big_paste("beta")
        editor.handle_input(f"\x1b[200~{paste_a}\x1b[201~")
        editor.handle_input(f"\x1b[200~{paste_b}\x1b[201~")
        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        editor.handle_input("\x7f")
        editor.handle_input(UNDO)
        editor.handle_input("\r")
        assert submitted == [paste_a + paste_b]

    def test_renumbers_registry_in_ascending_order(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = lambda t: submitted.append(t)

        paste_a = _big_paste("alpha")
        paste_b = _big_paste("beta")
        paste_c = _big_paste("gamma")
        editor.handle_input(f"\x1b[200~{paste_a}\x1b[201~")
        editor.handle_input("\x01")
        editor.handle_input(f"\x1b[200~{paste_b}\x1b[201~")
        editor.handle_input("\x01")
        editor.handle_input(f"\x1b[200~{paste_c}\x1b[201~")
        editor.handle_input("\x05")
        editor.handle_input("\x7f")
        editor.handle_input("\r")
        assert submitted == [paste_c + paste_b]

    def test_undo_after_set_text_restores_markers_and_registry(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = lambda t: submitted.append(t)

        paste = _big_paste("alpha")
        editor.handle_input(f"\x1b[200~{paste}\x1b[201~")
        editor.set_text("replacement")
        editor.handle_input(UNDO)
        editor.handle_input("\r")
        assert submitted == [paste]

    def test_multiple_markers_in_same_line(self):
        editor = make_editor()
        _paste_with_marker(editor)
        editor.handle_input(" ")
        _paste_with_marker(editor)

        text = editor.get_text()
        markers = PASTE_RE.findall(text)
        assert len(markers) == 2

        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": len(markers[0])}
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": len(markers[0]) + 1}
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {
            "line": 0,
            "col": len(markers[0]) + 1 + len(markers[1]),
        }

    def test_typed_marker_like_text_is_not_atomic(self):
        editor = make_editor()
        fake = "[paste #99 +5 lines]"
        for ch in fake:
            editor.handle_input(ch)
        assert editor.get_text() == fake

        editor.handle_input("\x01")
        editor.handle_input("\x1b[C")
        assert editor.get_cursor() == {"line": 0, "col": 1}

    def test_treats_marker_as_single_unit_for_word_movement(self):
        editor = make_editor()
        editor.handle_input("X")
        editor.handle_input(" ")
        _paste_with_marker(editor)
        editor.handle_input(" ")
        editor.handle_input("Y")

        marker = PASTE_RE.search(editor.get_text()).group(0)

        editor.handle_input("\x01")
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 1}
        editor.handle_input("\x1b[1;5C")
        assert editor.get_cursor() == {"line": 0, "col": 2 + len(marker)}

    def test_does_not_crash_when_marker_wider_than_render_width(self):
        editor = make_editor()
        big = ("line\n" * 47).rstrip("\n")
        editor.handle_input(f"\x1b[200~{big}\x1b[201~")
        marker = PASTE_RE.search(editor.get_text()).group(0)
        assert visible_width(marker) > 8

        lines = editor.render(8)
        for line in lines:
            assert visible_width(line) <= 8

    def test_does_not_crash_when_text_plus_marker_exceeds_width_with_cursor_on_marker(self):
        editor = make_editor()
        for _ in range(35):
            editor.handle_input("b")
        big = ("line\n" * 27).rstrip("\n")
        editor.handle_input(f"\x1b[200~{big}\x1b[201~")
        for _ in range(4):
            editor.handle_input("b")
        for _ in range(5):
            editor.handle_input("\x1b[D")

        render_width = 54
        for line in editor.render(render_width):
            assert visible_width(line) <= render_width

    def test_word_wrap_line_rechecks_overflow_after_backtracking(self):
        editor = make_editor()
        editor.handle_input(" ")
        for _ in range(35):
            editor.handle_input("b")
        big = ("line\n" * 27).rstrip("\n")
        editor.handle_input(f"\x1b[200~{big}\x1b[201~")
        for _ in range(4):
            editor.handle_input("b")

        render_width = 54
        for line in editor.render(render_width):
            assert visible_width(line) <= render_width

    def test_expands_pasted_content_literally(self):
        editor = make_editor()
        pasted = "\n".join(
            [
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
                "line 7",
                "line 8",
                "line 9",
                "line 10",
                "tokens $1 $2 $& $$ $` $' end",
            ]
        )
        editor.handle_input(f"\x1b[200~{pasted}\x1b[201~")
        assert PASTE_RE.search(editor.get_text())
        assert editor.get_expanded_text() == pasted

    def test_submits_large_paste_literally(self):
        editor = make_editor()
        # Same payload as the TypeScript test: the last line carries regex
        # replacement tokens ($1, $&, $$, $`, $') that must survive expansion.
        pasted = "\n".join(
            [
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
                "line 7",
                "line 8",
                "line 9",
                "line 10",
                "tokens $1 $2 $& $$ $` $' end",
            ]
        )
        submitted: list[str] = []
        editor.on_submit = lambda t: submitted.append(t)

        editor.handle_input(f"\x1b[200~{pasted}\x1b[201~")
        editor.handle_input("\r")
        assert submitted == [pasted]


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------


class TestAutocomplete:
    async def test_no_suggestions_during_paste(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            return None

        provider = FakeAutocompleteProvider(get_suggestions)
        editor.set_autocomplete_provider(provider)
        editor.handle_input("\x1b[200~look at @node_modules/react/index.js please\x1b[201~")
        assert editor.get_text() == "look at @node_modules/react/index.js please"
        await flush_autocomplete()
        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False

    async def test_undo_autocomplete(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            text = lines[0] if lines else ""
            prefix = text[:cursor_col]
            if prefix == "di":
                return AutocompleteSuggestions(
                    items=[AutocompleteItem(value="dist/", label="dist/")],
                    prefix="di",
                )
            return None

        provider = FakeAutocompleteProvider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        editor.handle_input("d")
        editor.handle_input("i")
        assert editor.get_text() == "di"

        editor.handle_input("\t")
        await flush_autocomplete()
        assert editor.get_text() == "dist/"
        assert editor.is_showing_autocomplete() is False

        editor.handle_input(UNDO)
        assert editor.get_text() == "di"

    async def test_auto_apply_single_force_file_suggestion(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            if not force:
                return None
            text = lines[0] if lines else ""
            prefix = text[:cursor_col]
            if prefix == "Work":
                return AutocompleteSuggestions(
                    items=[AutocompleteItem(value="Workspace/", label="Workspace/")],
                    prefix="Work",
                )
            return None

        provider = FakeAutocompleteProvider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        for ch in "Work":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush_autocomplete()
        assert editor.get_text() == "Workspace/"
        assert editor.is_showing_autocomplete() is False

        editor.handle_input(UNDO)
        assert editor.get_text() == "Work"

    async def test_shows_menu_when_force_file_has_multiple(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            if not force:
                return None
            text = lines[0] if lines else ""
            prefix = text[:cursor_col]
            if prefix == "src":
                return AutocompleteSuggestions(
                    items=[
                        AutocompleteItem(value="src/", label="src/"),
                        AutocompleteItem(value="src.txt", label="src.txt"),
                    ],
                    prefix="src",
                )
            return None

        provider = FakeAutocompleteProvider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        for ch in "src":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush_autocomplete()
        assert editor.get_text() == "src"
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\t")
        assert editor.get_text() == "src/"
        assert editor.is_showing_autocomplete() is False

    async def test_debounces_trigger_autocomplete(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            text = (lines[0] if lines else "")[:cursor_col]
            return AutocompleteSuggestions(
                items=[AutocompleteItem(value="@main.py", label="main.py")],
                prefix=text,
            )

        provider = FakeAutocompleteProvider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        editor.handle_input("@")
        editor.handle_input("m")
        editor.handle_input("a")
        editor.handle_input("i")

        # No calls yet (debounced)
        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False

        # The debounce is a real timer; wait for the request it schedules
        # rather than for a duration that a loaded machine may not respect.
        await wait_until(
            lambda: provider.call_count == 1,
            message="the debounced autocomplete request never fired",
        )
        await flush_autocomplete()

        assert provider.call_count == 1
        assert editor.is_showing_autocomplete() is True

    async def test_debounces_hash_autocomplete(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            text = (lines[0] if lines else "")[:cursor_col]
            return AutocompleteSuggestions(
                items=[AutocompleteItem(value="#2983", label="#2983")],
                prefix=text,
            )

        provider = FakeAutocompleteProvider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        for ch in "#298":
            editor.handle_input(ch)

        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False

        # The debounce is a real timer; wait for the request it schedules
        # rather than for a duration that a loaded machine may not respect.
        await wait_until(
            lambda: provider.call_count == 1,
            message="the debounced autocomplete request never fired",
        )
        await flush_autocomplete()

        assert provider.call_count == 1
        assert editor.is_showing_autocomplete() is True

    async def test_debounces_custom_trigger_characters_autocomplete(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            prefix = (lines[0] if lines else "")[:cursor_col]
            return AutocompleteSuggestions(
                items=[AutocompleteItem(value="$skill-name", label="skill-name")],
                prefix=prefix,
            )

        provider = FakeAutocompleteProvider(get_suggestions, trigger_characters=["$"])
        editor.set_autocomplete_provider(provider)

        for ch in "$sk":
            editor.handle_input(ch)

        assert provider.call_count == 0
        # The debounce is a real timer; wait for the request it schedules
        # rather than for a duration that a loaded machine may not respect.
        await wait_until(
            lambda: provider.call_count == 1,
            message="the debounced autocomplete request never fired",
        )
        await flush_autocomplete()

        assert provider.call_count == 1
        assert editor.is_showing_autocomplete() is True

    async def test_resets_custom_trigger_characters_when_provider_changes(self):
        editor = make_editor()

        def first_suggestions(lines, cursor_line, cursor_col, *, force):
            return AutocompleteSuggestions(
                items=[AutocompleteItem(value="$skill-name", label="skill-name")],
                prefix="$",
            )

        def second_suggestions(lines, cursor_line, cursor_col, *, force):
            return AutocompleteSuggestions(
                items=[AutocompleteItem(value="$skill-name", label="skill-name")],
                prefix="$",
            )

        editor.set_autocomplete_provider(FakeAutocompleteProvider(first_suggestions, trigger_characters=["$"]))
        second = FakeAutocompleteProvider(second_suggestions)
        editor.set_autocomplete_provider(second)

        editor.handle_input("$")
        editor.handle_input("s")
        # A negative assertion: no request may be scheduled at all. There is no
        # outcome to wait for, so this keeps a real wait past the debounce.
        await asyncio.sleep(0.06)
        await flush_autocomplete()

        assert second.call_count == 0
        assert editor.is_showing_autocomplete() is False


class TestSlashArgumentCompletion:
    """`Autocomplete` cases driving `/command <arg>` completion."""

    @staticmethod
    def _provider(values: list[str], *, filter_by_prefix: bool, command: str = "argtest"):
        pattern = re.compile(rf"^/{command}\s+(\S+)$")

        def get_suggestions(lines, cursor_line, cursor_col, *, force):
            text = lines[0] if lines else ""
            before_cursor = text[:cursor_col]
            match = pattern.match(before_cursor)
            if match is None:
                return None
            argument_text = match.group(1)
            items = [AutocompleteItem(value=v, label=v) for v in values]
            if filter_by_prefix:
                items = [item for item in items if item.value.startswith(argument_text)]
                if not items:
                    return None
            return AutocompleteSuggestions(items=items, prefix=argument_text)

        return FakeAutocompleteProvider(get_suggestions)

    async def test_applies_exact_typed_slash_argument_value_on_enter(self):
        editor = make_editor()
        editor.set_autocomplete_provider(self._provider(["one", "two", "three"], filter_by_prefix=True))

        for ch in "/argtest two":
            editor.handle_input(ch)

        assert editor.get_text() == "/argtest two"
        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert editor.get_text() == "/argtest two"

    async def test_selects_first_prefix_match_on_enter_when_not_exact(self):
        editor = make_editor()
        editor.set_autocomplete_provider(self._provider(["two", "three", "twelve"], filter_by_prefix=True))

        for ch in "/argtest t":
            editor.handle_input(ch)

        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert editor.get_text() == "/argtest two"

    async def test_highlights_unique_prefix_match_as_user_types(self):
        editor = make_editor()
        editor.set_autocomplete_provider(self._provider(["one", "two", "three"], filter_by_prefix=False))

        for ch in "/argtest tw":
            editor.handle_input(ch)

        assert editor.get_text() == "/argtest tw"
        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert editor.get_text() == "/argtest two"

    async def test_selects_first_prefix_match_when_multiple_items_match(self):
        editor = make_editor()
        editor.set_autocomplete_provider(self._provider(["one", "two", "three"], filter_by_prefix=False))

        for ch in "/argtest t":
            editor.handle_input(ch)

        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert editor.get_text() == "/argtest two"

    async def test_builtin_style_command_argument_completion_path(self):
        editor = make_editor()
        editor.set_autocomplete_provider(
            self._provider(
                ["gpt-4o", "gpt-4o-mini", "claude-sonnet"],
                filter_by_prefix=True,
                command="model",
            )
        )

        for ch in "/model gpt-4o-mini":
            editor.handle_input(ch)

        assert editor.get_text() == "/model gpt-4o-mini"
        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert editor.get_text() == "/model gpt-4o-mini"

    async def test_awaits_async_slash_command_argument_completions(self):
        editor = make_editor()

        async def get_argument_completions(prefix: str):
            if prefix.startswith("s"):
                return [AutocompleteItem(value="skill-a", label="skill-a")]
            return None

        provider = CombinedAutocompleteProvider(
            [
                SlashCommand(
                    name="load-skills",
                    description="Load skills",
                    get_argument_completions=get_argument_completions,
                )
            ],
            os.getcwd(),
        )
        editor.set_autocomplete_provider(provider)
        editor.set_text("/load-skills ")

        editor.handle_input("s")
        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\t")
        assert editor.get_text() == "/load-skills skill-a"
        assert editor.is_showing_autocomplete() is False

    async def test_ignores_invalid_slash_command_argument_completion_results(self):
        editor = make_editor()

        async def get_argument_completions(prefix: str):
            return "not-a-list"  # type: ignore[return-value]

        provider = CombinedAutocompleteProvider(
            [
                SlashCommand(
                    name="load-skills",
                    description="Load skills",
                    get_argument_completions=get_argument_completions,
                )
            ],
            os.getcwd(),
        )
        editor.set_autocomplete_provider(provider)
        editor.set_text("/load-skills ")

        editor.handle_input("s")
        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "/load-skills s"

    async def test_no_argument_completions_when_command_has_no_completer(self):
        editor = make_editor()

        async def get_argument_completions(prefix: str):
            return [AutocompleteItem(value="claude-opus", label="claude-opus")]

        provider = CombinedAutocompleteProvider(
            [
                SlashCommand(name="help", description="Show help"),
                SlashCommand(
                    name="model",
                    description="Switch model",
                    get_argument_completions=get_argument_completions,
                ),
            ],
            os.getcwd(),
        )
        editor.set_autocomplete_provider(provider)

        for ch in "/he":
            editor.handle_input(ch)
        await flush_autocomplete()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\t")
        assert editor.get_text() == "/help "
        assert editor.is_showing_autocomplete() is False
