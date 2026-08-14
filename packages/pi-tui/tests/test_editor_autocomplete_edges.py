"""Autocomplete edge cases for the editor component.

Companion to the `TestAutocomplete` class in `test_editor.py`. These cases
cover the menu key handling, request lifecycle (debounce, supersede, abort,
stale results) and trigger-character handling of
`pi_tui.components.editor`, cross-checked against
`packages/tui/src/components/editor.ts` and the autocomplete cases of
`packages/tui/test/editor.test.ts`.

No test waits out a real debounce timer: where the editor would schedule a
`call_later`, the debounce constant is monkeypatched to 0 ms so the request is
dispatched immediately and `flush()` (a few zero-delay yields) is enough to let
it run.
"""

from __future__ import annotations

import asyncio

import pi_tui.components.editor as editor_module
from pi_tui.autocomplete import (
    AppliedCompletion,
    AutocompleteItem,
    AutocompleteSuggestions,
)
from pi_tui.components.editor import Editor, EditorTheme
from pi_tui.components.select_list import SelectListTheme
from pi_tui.testing import FakeTerminal
from pi_tui.tui_main_screen import TuiMainScreen
from pi_tui.utils import strip_terminal_sequences, visible_width

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(text: str) -> str:
    return text


EDITOR_THEME = EditorTheme(
    border_color=_identity,
    select_list=SelectListTheme(
        selected_prefix=_identity,
        selected_text=_identity,
        description=_identity,
        scroll_info=_identity,
        no_match=_identity,
    ),
)

ESCAPE = "\x1b"


def make_editor(cols: int = 80, rows: int = 24) -> Editor:
    return Editor(TuiMainScreen(FakeTerminal(columns=cols, rows=rows)), EDITOR_THEME)


async def flush() -> None:
    """Let already-scheduled autocomplete callbacks/tasks run."""
    for _ in range(10):
        await asyncio.sleep(0)


def instant_debounce(monkeypatch) -> None:
    """Make the attachment-autocomplete debounce fire without a real timer."""
    monkeypatch.setattr(editor_module, "_ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS", 0)


def apply_completion(
    lines: list[str],
    cursor_line: int,
    cursor_col: int,
    item: AutocompleteItem,
    prefix: str,
) -> AppliedCompletion:
    """Replace `prefix` before the cursor with `item.value` (TS test helper)."""
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


class Provider:
    """Autocomplete provider double driven by a plain callable."""

    def __init__(
        self,
        get_suggestions,
        *,
        trigger_characters: list[str] | None = None,
        should_trigger_file_completion=None,
    ) -> None:
        self.trigger_characters = list(trigger_characters) if trigger_characters else []
        self._get_suggestions = get_suggestions
        self._should_trigger = should_trigger_file_completion
        self.call_count = 0

    async def get_suggestions(self, lines, cursor_line, cursor_col, *, signal, force=False):
        self.call_count += 1
        result = self._get_suggestions(lines, cursor_line, cursor_col, force=force, signal=signal)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):
        return apply_completion(lines, cursor_line, cursor_col, item, prefix)

    def should_trigger_file_completion(self, lines, cursor_line, cursor_col):
        if self._should_trigger is None:
            return True
        return self._should_trigger(lines, cursor_line, cursor_col)


def items(*values: str) -> list[AutocompleteItem]:
    return [AutocompleteItem(value=value, label=value) for value in values]


def slash_command_provider() -> Provider:
    """Provider offering `/model` and `/help` while the line starts with `/`."""

    def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
        before = (lines[cursor_line] if cursor_line < len(lines) else "")[:cursor_col]
        if not before.startswith("/"):
            return None
        matches = [c for c in items("/model", "/help") if c.value.startswith(before)]
        if not matches:
            return None
        return AutocompleteSuggestions(items=matches, prefix=before)

    return Provider(get_suggestions)


# ---------------------------------------------------------------------------
# Menu key handling
# ---------------------------------------------------------------------------


class TestAutocompleteMenuKeys:
    async def test_escape_closes_the_menu_and_keeps_the_text(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        editor.handle_input("/")
        await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input(ESCAPE)
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "/"

    async def test_arrow_keys_move_the_selection(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        editor.handle_input("/")
        await flush()

        editor.handle_input("\x1b[B")  # down: second entry
        editor.handle_input("\t")
        assert editor.get_text() == "/help"
        assert editor.is_showing_autocomplete() is False

    async def test_up_arrow_moves_the_selection_back(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        editor.handle_input("/")
        await flush()

        editor.handle_input("\x1b[B")
        editor.handle_input("\x1b[A")
        editor.handle_input("\t")
        assert editor.get_text() == "/model"

    async def test_tab_apply_emits_on_change(self):
        editor = make_editor()
        changes: list[str] = []
        editor.on_change = changes.append
        editor.set_autocomplete_provider(slash_command_provider())

        editor.handle_input("/")
        await flush()
        editor.handle_input("\t")
        assert changes[-1] == "/model"

    async def test_enter_applies_an_argument_completion_without_submitting(self):
        editor = make_editor()
        submitted: list[str] = []
        changes: list[str] = []
        editor.on_submit = submitted.append
        editor.on_change = changes.append

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            before = (lines[0] if lines else "")[:cursor_col]
            if not before.startswith("/argtest "):
                return None
            argument = before[len("/argtest ") :]
            if not argument or " " in argument:
                return None
            matches = [i for i in items("two", "three", "twelve") if i.value.startswith(argument)]
            if not matches:
                return None
            return AutocompleteSuggestions(items=matches, prefix=argument)

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "/argtest tw":
            editor.handle_input(ch)
            await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        # "tw" uniquely prefixes "two"; Enter applies it and does not submit.
        assert editor.get_text() == "/argtest two"
        assert changes[-1] == "/argtest two"
        assert submitted == []
        assert editor.is_showing_autocomplete() is False

    async def test_enter_applies_a_completion_without_an_on_change_handler(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            before = (lines[0] if lines else "")[:cursor_col]
            if not before.startswith("/argtest "):
                return None
            argument = before[len("/argtest ") :]
            matches = [i for i in items("two", "three") if i.value.startswith(argument)]
            return AutocompleteSuggestions(items=matches, prefix=argument) if matches else None

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "/argtest tw":
            editor.handle_input(ch)
            await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert editor.get_text() == "/argtest two"
        assert submitted == []

    async def test_enter_on_a_slash_command_applies_it_and_submits(self):
        editor = make_editor()
        submitted: list[str] = []
        editor.on_submit = submitted.append
        editor.set_autocomplete_provider(slash_command_provider())

        for ch in "/mod":
            editor.handle_input(ch)
            await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\r")
        assert submitted == ["/model"]
        assert editor.get_text() == ""

    async def test_menu_is_rendered_below_the_editor(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        editor.handle_input("/")
        await flush()

        rendered = editor.render(40)
        text = "\n".join(strip_terminal_sequences(line) for line in rendered)
        assert "/model" in text
        assert "/help" in text
        for line in rendered:
            assert visible_width(line) == 40

    async def test_exact_prefix_match_is_preselected(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            if not force:
                return None
            before = (lines[0] if lines else "")[:cursor_col]
            if before != "two":
                return None
            return AutocompleteSuggestions(items=items("three", "two"), prefix="two")

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "two":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\t")
        # Exact match wins over the first list entry.
        assert editor.get_text() == "two"

    async def test_empty_prefix_keeps_the_default_highlight(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            if not force:
                return None
            return AutocompleteSuggestions(items=items("alpha", "beta"), prefix="")

        editor.set_autocomplete_provider(Provider(get_suggestions))

        editor.handle_input("\t")
        await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\t")
        assert editor.get_text() == "alpha"


# ---------------------------------------------------------------------------
# Triggering
# ---------------------------------------------------------------------------


class TestAutocompleteTriggering:
    async def test_tab_without_a_provider_is_a_noop(self):
        editor = make_editor()
        for ch in "ab":
            editor.handle_input(ch)

        editor.handle_input("\t")
        await flush()
        assert editor.get_text() == "ab"
        assert editor.is_showing_autocomplete() is False

    async def test_tab_completes_a_slash_command_without_a_debounce(self):
        editor = make_editor()
        provider = slash_command_provider()
        editor.set_autocomplete_provider(provider)
        editor.set_text("/mod")

        editor.handle_input("\t")
        await flush()
        # Explicit Tab in a slash-command context is dispatched immediately
        # (no debounce) and opens the menu; it is not a force-file completion,
        # so a lone match is shown rather than auto-applied.
        assert provider.call_count == 1
        assert editor.is_showing_autocomplete() is True
        assert editor.get_text() == "/mod"

        editor.handle_input("\t")
        assert editor.get_text() == "/model"

    async def test_force_file_completion_can_be_declined_by_the_provider(self):
        editor = make_editor()
        provider = Provider(
            lambda *a, **kw: AutocompleteSuggestions(items=items("abc.txt"), prefix="abc"),
            should_trigger_file_completion=lambda lines, line, col: False,
        )
        editor.set_autocomplete_provider(provider)

        for ch in "abc":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush()

        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "abc"

    async def test_slash_menu_only_triggers_on_the_first_line(self):
        editor = make_editor()
        provider = slash_command_provider()
        editor.set_autocomplete_provider(provider)

        editor.handle_input("x")
        editor.handle_input("\n")
        editor.handle_input("/")
        await flush()

        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "x\n/"

    async def test_trigger_character_inside_a_word_does_not_open_the_menu(self, monkeypatch):
        instant_debounce(monkeypatch)
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            prefix = (lines[0] if lines else "")[:cursor_col]
            return AutocompleteSuggestions(items=items("@main.py"), prefix=prefix)

        provider = Provider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        # "@" right after a word character (here "a") is an email-style
        # address, not an attachment trigger.
        editor.handle_input("a")
        editor.handle_input("@")
        await flush()

        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "a@"

    async def test_custom_trigger_character_is_registered(self, monkeypatch):
        instant_debounce(monkeypatch)
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            prefix = (lines[0] if lines else "")[:cursor_col]
            return AutocompleteSuggestions(items=items("$skill-name"), prefix=prefix)

        provider = Provider(get_suggestions, trigger_characters=["$"])
        editor.set_autocomplete_provider(provider)

        editor.handle_input("$")
        editor.handle_input("s")
        await flush()

        assert provider.call_count >= 1
        assert editor.is_showing_autocomplete() is True

    async def test_invalid_trigger_characters_are_ignored(self, monkeypatch):
        instant_debounce(monkeypatch)
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            prefix = (lines[0] if lines else "")[:cursor_col]
            return AutocompleteSuggestions(items=items("whatever"), prefix=prefix)

        # "/" is reserved for the slash menu, "ab" is not a single character
        # and " " is whitespace - none of them become trigger characters.
        provider = Provider(get_suggestions, trigger_characters=["/", "ab", " "])
        editor.set_autocomplete_provider(provider)

        for ch in "hello /f":
            editor.handle_input(ch)
        await flush()

        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False

    async def test_trigger_characters_reset_when_the_provider_changes(self, monkeypatch):
        instant_debounce(monkeypatch)
        editor = make_editor()

        first = Provider(
            lambda *a, **kw: AutocompleteSuggestions(items=items("$skill"), prefix="$"),
            trigger_characters=["$"],
        )
        second = Provider(lambda *a, **kw: AutocompleteSuggestions(items=items("$skill"), prefix="$"))

        editor.set_autocomplete_provider(first)
        editor.set_autocomplete_provider(second)

        editor.handle_input("$")
        editor.handle_input("s")
        await flush()

        assert second.call_count == 0
        assert editor.is_showing_autocomplete() is False

    def test_autocomplete_is_skipped_without_a_running_event_loop(self):
        editor = make_editor()
        provider = slash_command_provider()
        editor.set_autocomplete_provider(provider)

        # Called from sync code: there is no loop to schedule the request on.
        editor.handle_input("/")

        assert provider.call_count == 0
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "/"


# ---------------------------------------------------------------------------
# Keeping an open menu in sync
# ---------------------------------------------------------------------------


class TestAutocompleteUpdates:
    async def test_typing_keeps_a_force_triggered_menu_open(self):
        editor = make_editor()
        all_files = items("readme.md", "package.json", "src/", "dist/")

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            prefix = (lines[0] if lines else "")[:cursor_col]
            if not (force or "/" in prefix or prefix.startswith(".")):
                return None
            matches = [f for f in all_files if f.value.lower().startswith(prefix.lower())]
            return AutocompleteSuggestions(items=matches, prefix=prefix) if matches else None

        editor.set_autocomplete_provider(Provider(get_suggestions))

        editor.handle_input("\t")
        await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("r")
        await flush()
        assert editor.get_text() == "r"
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("e")
        await flush()
        assert editor.get_text() == "re"
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\t")
        assert editor.get_text() == "readme.md"
        assert editor.is_showing_autocomplete() is False

    async def test_menu_closes_when_an_update_returns_no_suggestions(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            prefix = (lines[0] if lines else "")[:cursor_col]
            if prefix != "/mo":
                return None
            return AutocompleteSuggestions(items=items("/model"), prefix=prefix)

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "/mo":
            editor.handle_input(ch)
            await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("z")
        await flush()
        assert editor.get_text() == "/moz"
        assert editor.is_showing_autocomplete() is False

    async def test_backspacing_a_slash_command_to_empty_hides_the_menu(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        editor.handle_input("/")
        await flush()
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\x7f")
        await flush()
        assert editor.get_text() == ""
        assert editor.is_showing_autocomplete() is False

    async def test_backspace_retriggers_a_closed_menu_in_slash_context(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        for ch in "/mode":
            editor.handle_input(ch)
            await flush()
        editor.handle_input(ESCAPE)
        assert editor.is_showing_autocomplete() is False

        editor.handle_input("\x7f")
        await flush()
        assert editor.get_text() == "/mod"
        assert editor.is_showing_autocomplete() is True

    async def test_forward_delete_updates_and_retriggers_the_menu(self):
        editor = make_editor()
        editor.set_autocomplete_provider(slash_command_provider())

        for ch in "/modelx":
            editor.handle_input(ch)
            await flush()
        assert editor.is_showing_autocomplete() is False  # "/modelx" matches nothing

        editor.handle_input("\x1b[D")  # left, cursor before "x"
        await flush()
        editor.handle_input("\x04")  # forward delete removes "x"
        await flush()
        assert editor.get_text() == "/model"
        assert editor.is_showing_autocomplete() is True

        editor.handle_input("\x7f")
        await flush()
        editor.handle_input("\x04")  # nothing after the cursor: joins nothing
        await flush()
        assert editor.get_text() == "/mode"
        assert editor.is_showing_autocomplete() is True

    async def test_cursor_move_requeries_the_open_menu(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            before = (lines[0] if lines else "")[:cursor_col]
            if not before.startswith("/"):
                return None
            if " " in before:
                return AutocompleteSuggestions(
                    items=items("repo", "message", "help"),
                    prefix=before[before.index(" ") + 1 :],
                )
            return AutocompleteSuggestions(items=items("cmd"), prefix=before)

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "/cmd ":
            editor.handle_input(ch)
            await flush()
        assert editor.get_text() == "/cmd "
        assert editor.is_showing_autocomplete() is True
        at_argument = "\n".join(strip_terminal_sequences(line) for line in editor.render(80))
        assert "repo" in at_argument

        editor.handle_input("\x1b[D")  # left, back into the command name
        await flush()

        after_move = "\n".join(strip_terminal_sequences(line) for line in editor.render(80))
        assert "repo" not in after_move
        assert "message" not in after_move

    async def test_auto_applied_single_suggestion_emits_on_change(self):
        editor = make_editor()
        changes: list[str] = []
        editor.on_change = changes.append

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            if not force:
                return None
            prefix = (lines[0] if lines else "")[:cursor_col]
            if prefix != "Work":
                return None
            return AutocompleteSuggestions(items=items("Workspace/"), prefix="Work")

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "Work":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush()

        assert editor.get_text() == "Workspace/"
        assert changes[-1] == "Workspace/"


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------


class TestAutocompleteRequestLifecycle:
    async def test_provider_errors_are_swallowed(self):
        editor = make_editor()

        def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            raise RuntimeError("provider exploded")

        provider = Provider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        for ch in "abc":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush()

        assert provider.call_count == 1
        assert editor.is_showing_autocomplete() is False
        assert editor.get_text() == "abc"

    async def test_result_is_discarded_when_the_text_changed_meanwhile(self):
        editor = make_editor()
        gate = asyncio.Event()

        async def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            await gate.wait()
            return AutocompleteSuggestions(items=items("abc.txt"), prefix="ab")

        editor.set_autocomplete_provider(Provider(get_suggestions))

        for ch in "ab":
            editor.handle_input(ch)
        editor.handle_input("\t")
        await flush()
        assert editor.is_showing_autocomplete() is False  # still in flight

        editor.handle_input("c")  # text no longer matches the request snapshot
        gate.set()
        await flush()

        assert editor.get_text() == "abc"
        assert editor.is_showing_autocomplete() is False

    async def test_typing_aborts_the_in_flight_request(self, monkeypatch):
        instant_debounce(monkeypatch)
        editor = make_editor()
        gate = asyncio.Event()
        signals: list[asyncio.Event] = []

        async def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            signals.append(signal)
            await gate.wait()
            return AutocompleteSuggestions(items=items("@main.py"), prefix="@ma")

        editor.set_autocomplete_provider(Provider(get_suggestions))

        editor.handle_input("@")
        editor.handle_input("m")
        await flush()
        assert len(signals) == 1
        assert signals[0].is_set() is False

        editor.handle_input("a")  # continues typing: the first request is aborted
        await flush()
        assert signals[0].is_set() is True

        gate.set()
        await flush()
        assert editor.get_text() == "@ma"

    async def test_superseded_requests_never_reach_the_provider(self):
        editor = make_editor()
        gate = asyncio.Event()

        async def get_suggestions(lines, cursor_line, cursor_col, *, force, signal):
            await gate.wait()
            return AutocompleteSuggestions(items=items("ab.txt", "abc.txt"), prefix="ab")

        provider = Provider(get_suggestions)
        editor.set_autocomplete_provider(provider)

        for ch in "ab":
            editor.handle_input(ch)

        editor.handle_input("\t")
        await flush()
        editor.handle_input("\t")
        await flush()
        editor.handle_input("\t")
        await flush()
        assert provider.call_count == 1  # only the first request started

        gate.set()
        await flush()

        # The second request is superseded while queued behind the first and
        # is dropped without calling the provider again; the third one runs.
        assert provider.call_count == 2
        assert editor.is_showing_autocomplete() is True
