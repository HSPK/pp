"""Multi-line editor component, ported from
`packages/tui/src/components/editor.ts`.

Indexing note: TypeScript indexes string positions by UTF-16 code unit
(`string.length`, `slice`); this port indexes by Python code point (see
`pi_tui.utils` module docstring). Every `cursor_col`/`cursor_line`/`col`
value here — including the `TextChunk.start_index`/`end_index` returned by
`word_wrap_line` and every offset used by the autocomplete provider
interface — is a Python code-point offset. That difference is only visible
when the text contains astral-plane characters (emoji, some CJK
supplementary blocks): in TypeScript a single surrogate-pair emoji
occupies two `cursor_col` positions and Backspace deletes one code unit at
a time (mitigated in the TS source by falling back to grapheme
segmentation for deletion/movement); in this port the same emoji occupies
whatever number of code points its NFC form uses (one for most emojis, one
for its base plus zero-width joiners for compound emoji clusters) and
Backspace / arrow keys always operate at grapheme granularity via
`iter_graphemes`. The tests port over unchanged for BMP text and use
grapheme-aware assertions for emoji.

Word-segmentation caveat: TS `Intl.Segmenter` uses ICU dictionary-based
word breaking for CJK/Thai/Lao/Khmer/Myanmar; this port uses
`iter_word_segments`, which treats each CJK character as its own
word-like segment. Editor tests covering CJK word movement have been
translated to match the character-by-character granularity — see
`word_navigation.py` for the full rationale.

Async / debounced autocomplete: TS uses `setTimeout` from a sync
`handleInput`; this port uses `asyncio.get_running_loop().call_later` and
`create_task`. If `handle_input` is called with no running event loop
(e.g. from a purely-sync test), autocomplete triggers are silently
skipped instead of raising. Autocomplete tests are `async def` and rely
on the standard `pytest-asyncio` "auto" mode already enabled for this
repo (see the root `pyproject.toml`).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pi_tui.autocomplete import (
    AppliedCompletion,
    AutocompleteItem,
    AutocompleteSuggestions,
)
from pi_tui.component import CURSOR_MARKER, Component, Focusable
from pi_tui.components.select_list import (
    SelectItem,
    SelectList,
    SelectListLayoutOptions,
    SelectListTheme,
)
from pi_tui.keybindings import get_keybindings
from pi_tui.keys import decode_printable_key, matches_key
from pi_tui.kill_ring import KillRing
from pi_tui.undo_stack import UndoStack
from pi_tui.utils import (
    _is_cjk_break_char,
    is_whitespace_char,
    iter_graphemes,
    iter_word_segments,
    slice_by_column,
    visible_width,
)
from pi_tui.word_navigation import WordNavigationOptions, find_word_backward, find_word_forward

PASTE_MARKER_REGEX = re.compile(r"\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]")
"""Regex matching paste markers like `[paste #1 +123 lines]` or `[paste #2 1234 chars]`."""

PASTE_MARKER_SINGLE = re.compile(r"^\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]$")
"""Non-global version for single-segment testing."""


def _is_paste_marker(segment: str) -> bool:
    """Check if a segment is a paste marker (i.e. was merged by `segment_with_markers`)."""
    return len(segment) >= 10 and PASTE_MARKER_SINGLE.match(segment) is not None


@dataclass
class _Segment:
    """Uniform holder for grapheme/word segments used by the editor.

    Mirrors the subset of `Intl.SegmentData` (segment + index) actually
    consumed by `word_wrap_line`; `is_word_like` is populated when the
    segment came from a word segmenter so `find_word_*` can use it.
    """

    segment: str
    index: int = 0
    is_word_like: bool = False


def _iter_graphemes_with_indices(text: str) -> list[_Segment]:
    result: list[_Segment] = []
    index = 0
    for grapheme in iter_graphemes(text):
        result.append(_Segment(segment=grapheme, index=index))
        index += len(grapheme)
    return result


def _iter_word_segments_with_indices(text: str) -> list[_Segment]:
    result: list[_Segment] = []
    index = 0
    for ws in iter_word_segments(text):
        result.append(_Segment(segment=ws.segment, index=index, is_word_like=ws.is_word_like))
        index += len(ws.segment)
    return result


def _segment_with_markers(text: str, mode: Literal["grapheme", "word"], valid_ids: set[int]) -> list[_Segment]:
    """Segment `text` and merge paste markers whose id is in `valid_ids` into
    single atomic segments.

    Ported from `segmentWithMarkers` in editor.ts. Only markers whose numeric
    ID exists in `valid_ids` are merged; unknown-id marker-looking text is
    passed through as individual base segments so the cursor can move through
    it normally.
    """
    base = _iter_graphemes_with_indices(text) if mode == "grapheme" else _iter_word_segments_with_indices(text)

    # Fast path: no valid IDs or no marker literal present.
    if not valid_ids or "[paste #" not in text:
        return base

    markers: list[tuple[int, int]] = []
    for m in PASTE_MARKER_REGEX.finditer(text):
        try:
            marker_id = int(m.group(1))
        except (TypeError, ValueError):
            continue
        if marker_id not in valid_ids:
            continue
        markers.append((m.start(), m.end()))

    if not markers:
        return base

    result: list[_Segment] = []
    marker_idx = 0

    for seg in base:
        # Skip past markers that end at or before this segment's start.
        while marker_idx < len(markers) and markers[marker_idx][1] <= seg.index:
            marker_idx += 1

        marker = markers[marker_idx] if marker_idx < len(markers) else None

        if marker is not None and seg.index >= marker[0] and seg.index < marker[1]:
            # This segment falls inside a marker. Emit the merged segment
            # only when we see its first base segment; skip the rest.
            if seg.index == marker[0]:
                result.append(_Segment(segment=text[marker[0] : marker[1]], index=marker[0]))
        else:
            result.append(seg)

    return result


@dataclass
class TextChunk:
    """A chunk of text produced by `word_wrap_line`.

    Tracks both the text content and its position in the original line.
    """

    text: str
    start_index: int
    end_index: int


def word_wrap_line(line: str, max_width: int, pre_segmented: list[_Segment] | None = None) -> list[TextChunk]:
    """Split `line` into word-wrapped chunks.

    Wraps at word boundaries when possible, falling back to character-level
    wrapping for words longer than `max_width`. When `pre_segmented` is given
    (e.g. paste-marker-aware graphemes) it is used verbatim; otherwise the
    default grapheme iterator is used.
    """
    if not line or max_width <= 0:
        return [TextChunk(text="", start_index=0, end_index=0)]

    line_width = visible_width(line)
    if line_width <= max_width:
        return [TextChunk(text=line, start_index=0, end_index=len(line))]

    chunks: list[TextChunk] = []
    segments = pre_segmented if pre_segmented is not None else _iter_graphemes_with_indices(line)

    current_width = 0
    chunk_start = 0

    # Wrap opportunity: the position after the last whitespace before a
    # non-whitespace grapheme, i.e. where a line break is allowed.
    wrap_opp_index = -1
    wrap_opp_width = 0

    for i, seg in enumerate(segments):
        grapheme = seg.segment
        g_width = visible_width(grapheme)
        char_index = seg.index
        is_ws = (not _is_paste_marker(grapheme)) and is_whitespace_char(grapheme)

        if current_width + g_width > max_width:
            if wrap_opp_index >= 0 and current_width - wrap_opp_width + g_width <= max_width:
                # Backtrack to the last wrap opportunity: the remaining
                # content plus the current grapheme still fits.
                chunks.append(
                    TextChunk(
                        text=line[chunk_start:wrap_opp_index],
                        start_index=chunk_start,
                        end_index=wrap_opp_index,
                    )
                )
                chunk_start = wrap_opp_index
                current_width -= wrap_opp_width
            elif chunk_start < char_index:
                # No viable wrap opportunity: force-break at the current
                # position. Also handles the case where backtracking to the
                # word boundary wouldn't help because the remaining content
                # plus the current wide grapheme still exceeds `max_width`.
                chunks.append(
                    TextChunk(
                        text=line[chunk_start:char_index],
                        start_index=chunk_start,
                        end_index=char_index,
                    )
                )
                chunk_start = char_index
                current_width = 0
            wrap_opp_index = -1

        if g_width > max_width:
            # A single atomic segment (e.g. a paste marker in a narrow
            # terminal) is wider than `max_width`. Re-wrap it at grapheme
            # granularity — the segment stays logically atomic for cursor
            # movement/editing; the split is purely visual for layout.
            sub_chunks = word_wrap_line(grapheme, max_width)
            for sc in sub_chunks[:-1]:
                chunks.append(
                    TextChunk(
                        text=sc.text,
                        start_index=char_index + sc.start_index,
                        end_index=char_index + sc.end_index,
                    )
                )
            last = sub_chunks[-1]
            chunk_start = char_index + last.start_index
            current_width = visible_width(last.text)
            wrap_opp_index = -1
            continue

        current_width += g_width

        # Record a wrap opportunity: whitespace followed by non-whitespace
        # (multiple spaces join; the break point is after the last space),
        # or at a boundary where either side is a CJK break char (CJK allows
        # breaking between any adjacent characters).
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if is_ws and nxt is not None and (_is_paste_marker(nxt.segment) or not is_whitespace_char(nxt.segment)):
            wrap_opp_index = nxt.index
            wrap_opp_width = current_width
        elif (not is_ws) and nxt is not None and not is_whitespace_char(nxt.segment):
            is_cjk = (not _is_paste_marker(grapheme)) and _is_cjk_break_char(grapheme[0])
            next_is_cjk = (not _is_paste_marker(nxt.segment)) and _is_cjk_break_char(nxt.segment[0])
            if is_cjk or next_is_cjk:
                wrap_opp_index = nxt.index
                wrap_opp_width = current_width

    chunks.append(TextChunk(text=line[chunk_start:], start_index=chunk_start, end_index=len(line)))

    return chunks


@dataclass
class _EditorState:
    lines: list[str] = field(default_factory=lambda: [""])
    cursor_line: int = 0
    cursor_col: int = 0


@dataclass
class _EditorSnapshot:
    """Undo snapshot: editor text state plus the paste registry."""

    state: _EditorState
    pastes: dict[int, str]
    paste_counter: int


@dataclass
class _LayoutLine:
    text: str
    has_cursor: bool
    cursor_pos: int | None = None


@dataclass
class _VisualLine:
    logical_line: int
    start_col: int
    length: int


@dataclass
class EditorTheme:
    border_color: Callable[[str], str]
    select_list: SelectListTheme


@dataclass
class EditorOptions:
    padding_x: int | None = None
    autocomplete_max_visible: int | None = None


class _AutocompleteProviderLike(Protocol):
    """Structural typing for autocomplete providers accepted by the editor.

    Mirrors `pi_tui.autocomplete.AutocompleteProvider` but avoids importing
    it as a strict runtime type check so simple test doubles don't need to
    define every attribute of the full Protocol.
    """

    async def get_suggestions(  # type: ignore[empty-body]
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        *,
        signal: asyncio.Event,
        force: bool = False,
    ) -> AutocompleteSuggestions | None: ...

    def apply_completion(  # type: ignore[empty-body]
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> AppliedCompletion: ...


_SLASH_COMMAND_SELECT_LIST_LAYOUT = SelectListLayoutOptions(
    min_primary_column_width=12,
    max_primary_column_width=32,
)

_ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS = 20
_DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS = ["@", "#"]


def _escape_character_class(value: str) -> str:
    return re.sub(r"([\\^$.*+?()\[\]{}|\-])", r"\\\1", value)


def _build_trigger_pattern(trigger_characters: list[str]) -> re.Pattern[str]:
    charset = "".join(_escape_character_class(c) for c in trigger_characters)
    return re.compile(rf"(?:^|[\s])[{charset}][^\s]*$")


def _build_debounce_pattern(trigger_characters: list[str]) -> re.Pattern[str]:
    escaped_without_at = "".join(_escape_character_class(c) for c in trigger_characters if c != "@")
    return re.compile(rf'(?:^|[ \t])(?:@(?:"[^"]*|[^\s]*)|[{escaped_without_at}][^\s]*)$')


def _create_scroll_border(direction: str, hidden_line_count: int, width: int) -> str:
    available_width = max(0, width)
    indicator = f"─── {direction} {hidden_line_count} more "
    remaining = available_width - visible_width(indicator)
    if remaining >= 0:
        return indicator + ("─" * remaining)

    ellipsis = "..."[:available_width]
    indicator_width = available_width - visible_width(ellipsis)
    return slice_by_column(indicator, 0, indicator_width, True) + ellipsis


_JUMP_MODE_FORWARD = "forward"
_JUMP_MODE_BACKWARD = "backward"


class _EditorTui(Protocol):
    """Duck-typed subset of TUI used by the editor.

    Mirrors the TS `constructor(tui: TUI, ...)` dependency: the editor only
    reads `tui.terminal.rows` and calls `tui.request_render()`. Any object
    supplying those two — the real `TuiBase`, or a small fake in tests —
    satisfies this Protocol.
    """

    terminal: object
    """Must expose a `rows` int attribute."""

    def request_render(self, force: bool = False) -> None: ...


class Editor(Component, Focusable):
    """Multi-line text editor with word-wrap, undo/kill-ring, history, and
    autocomplete integration."""

    def __init__(
        self,
        tui: _EditorTui,
        theme: EditorTheme,
        options: EditorOptions | None = None,
    ) -> None:
        options = options or EditorOptions()
        self._state = _EditorState()

        self.focused: bool = False

        self._tui = tui
        self._theme = theme
        self.border_color: Callable[[str], str] = theme.border_color

        padding_x = options.padding_x if options.padding_x is not None else 0
        self._padding_x = max(0, int(padding_x))

        max_visible = options.autocomplete_max_visible if options.autocomplete_max_visible is not None else 5
        self._autocomplete_max_visible = max(3, min(20, int(max_visible)))

        self._last_width = 80
        self._scroll_offset = 0

        # Autocomplete state.
        self._autocomplete_provider: _AutocompleteProviderLike | None = None
        self._autocomplete_trigger_characters: list[str] = list(_DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS)
        self._autocomplete_trigger_pattern: re.Pattern[str] = _build_trigger_pattern(
            self._autocomplete_trigger_characters
        )
        self._autocomplete_debounce_pattern: re.Pattern[str] = _build_debounce_pattern(
            self._autocomplete_trigger_characters
        )
        self._autocomplete_list: SelectList | None = None
        self._autocomplete_state: Literal["regular", "force"] | None = None
        self._autocomplete_prefix: str = ""
        self._autocomplete_abort: asyncio.Event | None = None
        self._autocomplete_debounce_handle: asyncio.TimerHandle | None = None
        self._autocomplete_request_task: asyncio.Future[None] | None = None
        self._autocomplete_dispatch_task: asyncio.Task[None] | None = None
        self._autocomplete_start_token: int = 0
        self._autocomplete_request_id: int = 0

        # Paste tracking for large pastes.
        self._pastes: dict[int, str] = {}
        self._paste_counter: int = 0

        # Bracketed paste mode buffering.
        self._paste_buffer: str = ""
        self._is_in_paste: bool = False

        # Prompt history for up/down navigation.
        self._history: list[str] = []
        self._history_index: int = -1
        self._history_draft: _EditorState | None = None

        # Kill ring / last-action tracking.
        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill" | "yank" | "type-word" | None

        # Character jump mode.
        self._jump_mode: str | None = None

        # Vertical-movement sticky column bookkeeping.
        self._preferred_visual_col: int | None = None
        self._snapped_from_cursor_col: int | None = None

        # Undo.
        self._undo_stack: UndoStack[_EditorSnapshot] = UndoStack()

        # Public callbacks.
        self.on_submit: Callable[[str], None] | None = None
        self.on_change: Callable[[str], None] | None = None
        self.disable_submit: bool = False

    # ------------------------------------------------------------------
    # Segmentation helpers.
    # ------------------------------------------------------------------

    def _valid_paste_ids(self) -> set[int]:
        return set(self._pastes.keys())

    def _segment(self, text: str, mode: Literal["grapheme", "word"]) -> list[_Segment]:
        return _segment_with_markers(text, mode, self._valid_paste_ids())

    # ------------------------------------------------------------------
    # Public state accessors.
    # ------------------------------------------------------------------

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding: int) -> None:
        new_padding = max(0, int(padding))
        if self._padding_x != new_padding:
            self._padding_x = new_padding
            self._tui.request_render()

    def get_autocomplete_max_visible(self) -> int:
        return self._autocomplete_max_visible

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        new_max = max(3, min(20, int(max_visible)))
        if self._autocomplete_max_visible != new_max:
            self._autocomplete_max_visible = new_max
            self._tui.request_render()

    def set_autocomplete_provider(self, provider: _AutocompleteProviderLike) -> None:
        self._cancel_autocomplete()
        self._autocomplete_provider = provider
        trigger = getattr(provider, "trigger_characters", None) or []
        self._set_autocomplete_trigger_characters(list(trigger))

    def add_to_history(self, text: str) -> None:
        """Add a prompt to history for up/down arrow navigation.

        Called after successful submission.
        """
        trimmed = text.strip()
        if not trimmed:
            return
        if self._history and self._history[0] == trimmed:
            return
        self._history.insert(0, trimmed)
        if len(self._history) > 100:
            self._history.pop()

    def get_text(self) -> str:
        return "\n".join(self._state.lines)

    def get_expanded_text(self) -> str:
        """Return the current text with paste markers expanded to full content."""
        return self._expand_paste_markers("\n".join(self._state.lines))

    def get_lines(self) -> list[str]:
        return list(self._state.lines)

    def get_cursor(self) -> dict[str, int]:
        return {"line": self._state.cursor_line, "col": self._state.cursor_col}

    def set_text(self, text: str) -> None:
        self._cancel_autocomplete()
        self._last_action = None
        self._exit_history_browsing()
        normalized = self._normalize_text(text)
        if self.get_text() != normalized:
            self._push_undo_snapshot()
        self._pastes.clear()
        self._paste_counter = 0
        self._set_text_internal(normalized)

    def insert_text_at_cursor(self, text: str) -> None:
        """Insert text at the current cursor position (single undo unit)."""
        if not text:
            return
        self._cancel_autocomplete()
        self._push_undo_snapshot()
        self._last_action = None
        self._exit_history_browsing()
        self._insert_text_at_cursor_internal(text)

    def is_showing_autocomplete(self) -> bool:
        return self._autocomplete_state is not None

    # ------------------------------------------------------------------
    # Component interface.
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        # No cached state to invalidate currently.
        pass

    def render(self, width: int) -> list[str]:
        max_padding = max(0, (width - 1) // 2)
        padding_x = min(self._padding_x, max_padding)
        content_width = max(1, width - padding_x * 2)

        # With padding the cursor can overflow into it; without padding we
        # reserve one column for the cursor.
        layout_width = max(1, content_width - (0 if padding_x else 1))
        self._last_width = layout_width

        horizontal = self.border_color("─")
        layout_lines = self._layout_text(layout_width)

        terminal_rows = self._tui.terminal.rows
        max_visible_lines = max(5, int(terminal_rows * 0.3))

        cursor_line_index = -1
        for i, line in enumerate(layout_lines):
            if line.has_cursor:
                cursor_line_index = i
                break
        if cursor_line_index == -1:
            cursor_line_index = 0

        if cursor_line_index < self._scroll_offset:
            self._scroll_offset = cursor_line_index
        elif cursor_line_index >= self._scroll_offset + max_visible_lines:
            self._scroll_offset = cursor_line_index - max_visible_lines + 1

        max_scroll_offset = max(0, len(layout_lines) - max_visible_lines)
        self._scroll_offset = max(0, min(self._scroll_offset, max_scroll_offset))

        visible_lines = layout_lines[self._scroll_offset : self._scroll_offset + max_visible_lines]

        result: list[str] = []
        left_padding = " " * padding_x
        right_padding = left_padding

        if self._scroll_offset > 0:
            border = _create_scroll_border("↑", self._scroll_offset, width)
            result.append(self.border_color(border))
        else:
            result.append(horizontal * width)

        emit_cursor_marker = self.focused

        for layout_line in visible_lines:
            display_text = layout_line.text
            line_visible_width = visible_width(layout_line.text)
            cursor_in_padding = False

            if layout_line.has_cursor and layout_line.cursor_pos is not None:
                before = display_text[: layout_line.cursor_pos]
                after = display_text[layout_line.cursor_pos :]

                marker = CURSOR_MARKER if emit_cursor_marker else ""

                if after:
                    after_graphemes = self._segment(after, "grapheme")
                    first_grapheme = after_graphemes[0].segment if after_graphemes else ""
                    rest_after = after[len(first_grapheme) :]
                    cursor = f"\x1b[7m{first_grapheme}\x1b[0m"
                    display_text = before + marker + cursor + rest_after
                else:
                    cursor = "\x1b[7m \x1b[0m"
                    display_text = before + marker + cursor
                    line_visible_width += 1
                    if line_visible_width > content_width and padding_x > 0:
                        cursor_in_padding = True

            padding = " " * max(0, content_width - line_visible_width)
            line_right_padding = right_padding[1:] if cursor_in_padding else right_padding

            result.append(f"{left_padding}{display_text}{padding}{line_right_padding}")

        lines_below = len(layout_lines) - (self._scroll_offset + len(visible_lines))
        if lines_below > 0:
            border = _create_scroll_border("↓", lines_below, width)
            result.append(self.border_color(border))
        else:
            result.append(horizontal * width)

        if self._autocomplete_state and self._autocomplete_list is not None:
            autocomplete_result = self._autocomplete_list.render(content_width)
            for line in autocomplete_result:
                line_width = visible_width(line)
                line_padding = " " * max(0, content_width - line_width)
                result.append(f"{left_padding}{line}{line_padding}{right_padding}")

        return result

    # ------------------------------------------------------------------
    # Input handling.
    # ------------------------------------------------------------------

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()

        # Handle character-jump mode (awaiting next character to jump to).
        if self._jump_mode is not None:
            if kb.matches(data, "tui.editor.jumpForward") or kb.matches(data, "tui.editor.jumpBackward"):
                self._jump_mode = None
                return

            printable = decode_printable_key(data)
            if printable is None and data and ord(data[0]) >= 32:
                printable = data

            if printable is not None:
                direction = self._jump_mode
                self._jump_mode = None
                self._jump_to_char(printable, direction)
                return

            # Control character - cancel and fall through.
            self._jump_mode = None

        # Bracketed paste mode.
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += data
            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                paste_content = self._paste_buffer[:end_index]
                if paste_content:
                    self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_index + 6 :]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
                return
            return

        # Ctrl+C - let parent handle (exit/clear).
        if kb.matches(data, "tui.input.copy"):
            return

        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return

        # Autocomplete-menu keys.
        if self._autocomplete_state and self._autocomplete_list is not None:
            if kb.matches(data, "tui.select.cancel"):
                self._cancel_autocomplete()
                return
            if kb.matches(data, "tui.select.up") or kb.matches(data, "tui.select.down"):
                self._autocomplete_list.handle_input(data)
                return
            if kb.matches(data, "tui.input.tab"):
                selected = self._autocomplete_list.get_selected_item()
                if selected is not None and self._autocomplete_provider is not None:
                    self._push_undo_snapshot()
                    self._last_action = None
                    result = self._autocomplete_provider.apply_completion(
                        self._state.lines,
                        self._state.cursor_line,
                        self._state.cursor_col,
                        _select_item_to_autocomplete_item(selected),
                        self._autocomplete_prefix,
                    )
                    self._state.lines = result.lines
                    self._state.cursor_line = result.cursor_line
                    self._set_cursor_col(result.cursor_col)
                    self._cancel_autocomplete()
                    if self.on_change:
                        self.on_change(self.get_text())
                return
            if kb.matches(data, "tui.select.confirm"):
                selected = self._autocomplete_list.get_selected_item()
                if selected is not None and self._autocomplete_provider is not None:
                    self._push_undo_snapshot()
                    self._last_action = None
                    result = self._autocomplete_provider.apply_completion(
                        self._state.lines,
                        self._state.cursor_line,
                        self._state.cursor_col,
                        _select_item_to_autocomplete_item(selected),
                        self._autocomplete_prefix,
                    )
                    self._state.lines = result.lines
                    self._state.cursor_line = result.cursor_line
                    self._set_cursor_col(result.cursor_col)

                    if self._autocomplete_prefix.startswith("/"):
                        self._cancel_autocomplete()
                        # Fall through to submit.
                    else:
                        self._cancel_autocomplete()
                        if self.on_change:
                            self.on_change(self.get_text())
                        return

        # Tab - trigger completion.
        if kb.matches(data, "tui.input.tab") and not self._autocomplete_state:
            self._handle_tab_completion()
            return

        # Deletion actions.
        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_end_of_line()
            return
        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_start_of_line()
            return
        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "tui.editor.deleteCharBackward") or matches_key(data, "shift+backspace"):
            self._handle_backspace()
            return
        if kb.matches(data, "tui.editor.deleteCharForward") or matches_key(data, "shift+delete"):
            self._handle_forward_delete()
            return

        # Kill-ring yank actions.
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return

        # Dedicated history actions.
        if kb.matches(data, "tui.editor.historyPrevious"):
            self._cancel_autocomplete()
            self._navigate_history(-1)
            return
        if kb.matches(data, "tui.editor.historyNext"):
            self._cancel_autocomplete()
            self._navigate_history(1)
            return

        # Cursor movement actions.
        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._move_to_line_start()
            return
        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._move_to_line_end()
            return
        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return

        # New line.
        if (
            kb.matches(data, "tui.input.newLine")
            or (data and ord(data[0]) == 10 and len(data) > 1)
            or data == "\x1b\r"
            or data == "\x1b[13;2~"
            or (len(data) > 1 and "\x1b" in data and "\r" in data)
            or (data == "\n" and len(data) == 1)
        ):
            if self._should_submit_on_backslash_enter(data, kb):
                self._handle_backspace()
                self._submit_value()
                return
            self._add_new_line()
            return

        # Submit (Enter).
        if kb.matches(data, "tui.input.submit"):
            if self.disable_submit:
                return

            current_line = (
                self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            )
            if self._state.cursor_col > 0 and current_line[self._state.cursor_col - 1] == "\\":
                self._handle_backspace()
                self._add_new_line()
                return

            self._submit_value()
            return

        # Arrow key navigation (with history support).
        if kb.matches(data, "tui.editor.cursorUp"):
            if self._is_on_first_visual_line() and (
                self._is_editor_empty() or self._history_index > -1 or self._state.cursor_col == 0
            ):
                self._navigate_history(-1)
            elif self._is_on_first_visual_line():
                self._move_to_line_start()
            else:
                self._move_cursor(-1, 0)
            return
        if kb.matches(data, "tui.editor.cursorDown"):
            if self._history_index > -1 and self._is_on_last_visual_line():
                self._navigate_history(1)
            elif self._is_on_last_visual_line():
                self._move_to_line_end()
            else:
                self._move_cursor(1, 0)
            return
        if kb.matches(data, "tui.editor.cursorRight"):
            self._move_cursor(0, 1)
            return
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._move_cursor(0, -1)
            return

        # Page up/down.
        if kb.matches(data, "tui.editor.pageUp"):
            self._page_scroll(-1)
            return
        if kb.matches(data, "tui.editor.pageDown"):
            self._page_scroll(1)
            return

        # Character-jump triggers.
        if kb.matches(data, "tui.editor.jumpForward"):
            self._jump_mode = _JUMP_MODE_FORWARD
            return
        if kb.matches(data, "tui.editor.jumpBackward"):
            self._jump_mode = _JUMP_MODE_BACKWARD
            return

        # Shift+Space - regular space.
        if matches_key(data, "shift+space"):
            self._insert_character(" ")
            return

        printable = decode_printable_key(data)
        if printable is not None:
            self._insert_character(printable)
            return

        if data and ord(data[0]) >= 32:
            self._insert_character(data)

    # ------------------------------------------------------------------
    # Layout.
    # ------------------------------------------------------------------

    def _layout_text(self, content_width: int) -> list[_LayoutLine]:
        layout_lines: list[_LayoutLine] = []

        if not self._state.lines or (len(self._state.lines) == 1 and self._state.lines[0] == ""):
            layout_lines.append(_LayoutLine(text="", has_cursor=True, cursor_pos=0))
            return layout_lines

        for i, line in enumerate(self._state.lines):
            is_current_line = i == self._state.cursor_line
            line_visible_width = visible_width(line)

            if line_visible_width <= content_width:
                if is_current_line:
                    layout_lines.append(_LayoutLine(text=line, has_cursor=True, cursor_pos=self._state.cursor_col))
                else:
                    layout_lines.append(_LayoutLine(text=line, has_cursor=False))
                continue

            chunks = word_wrap_line(line, content_width, self._segment(line, "grapheme"))

            for chunk_index, chunk in enumerate(chunks):
                cursor_pos = self._state.cursor_col
                is_last_chunk = chunk_index == len(chunks) - 1

                has_cursor_in_chunk = False
                adjusted_cursor_pos = 0

                if is_current_line:
                    if is_last_chunk:
                        has_cursor_in_chunk = cursor_pos >= chunk.start_index
                        adjusted_cursor_pos = cursor_pos - chunk.start_index
                    else:
                        has_cursor_in_chunk = chunk.start_index <= cursor_pos < chunk.end_index
                        if has_cursor_in_chunk:
                            adjusted_cursor_pos = cursor_pos - chunk.start_index
                            if adjusted_cursor_pos > len(chunk.text):
                                adjusted_cursor_pos = len(chunk.text)

                if has_cursor_in_chunk:
                    layout_lines.append(_LayoutLine(text=chunk.text, has_cursor=True, cursor_pos=adjusted_cursor_pos))
                else:
                    layout_lines.append(_LayoutLine(text=chunk.text, has_cursor=False))

        return layout_lines

    def _build_visual_line_map(self, width: int) -> list[_VisualLine]:
        visual_lines: list[_VisualLine] = []
        for i, line in enumerate(self._state.lines):
            line_vis_width = visible_width(line)
            if not line:
                visual_lines.append(_VisualLine(logical_line=i, start_col=0, length=0))
            elif line_vis_width <= width:
                visual_lines.append(_VisualLine(logical_line=i, start_col=0, length=len(line)))
            else:
                chunks = word_wrap_line(line, width, self._segment(line, "grapheme"))
                for chunk in chunks:
                    visual_lines.append(
                        _VisualLine(
                            logical_line=i,
                            start_col=chunk.start_index,
                            length=chunk.end_index - chunk.start_index,
                        )
                    )
        return visual_lines

    def _find_visual_line_at(self, visual_lines: list[_VisualLine], line: int, col: int) -> int:
        for i, vl in enumerate(visual_lines):
            if vl.logical_line != line:
                continue
            offset = col - vl.start_col
            is_last_segment_of_line = i == len(visual_lines) - 1 or visual_lines[i + 1].logical_line != vl.logical_line
            if offset >= 0 and (offset < vl.length or (is_last_segment_of_line and offset == vl.length)):
                return i
        return len(visual_lines) - 1

    def _find_current_visual_line(self, visual_lines: list[_VisualLine]) -> int:
        return self._find_visual_line_at(visual_lines, self._state.cursor_line, self._state.cursor_col)

    def _is_on_first_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(visual_lines) == 0

    def _is_on_last_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(visual_lines) == len(visual_lines) - 1

    def _is_editor_empty(self) -> bool:
        return len(self._state.lines) == 1 and self._state.lines[0] == ""

    # ------------------------------------------------------------------
    # History navigation.
    # ------------------------------------------------------------------

    def _navigate_history(self, direction: int) -> None:
        self._last_action = None
        if not self._history:
            return

        new_index = self._history_index - direction
        if new_index < -1 or new_index >= len(self._history):
            return

        if self._history_index == -1 and new_index >= 0:
            self._push_undo_snapshot()
            self._history_draft = _EditorState(
                lines=list(self._state.lines),
                cursor_line=self._state.cursor_line,
                cursor_col=self._state.cursor_col,
            )

        self._history_index = new_index

        if self._history_index == -1:
            draft = self._history_draft
            self._history_draft = None
            if draft is not None:
                self._state = draft
                self._preferred_visual_col = None
                self._snapped_from_cursor_col = None
                self._scroll_offset = 0
                if self.on_change:
                    self.on_change(self.get_text())
            else:
                self._set_text_internal("")
        else:
            entry = self._history[self._history_index]
            self._set_text_internal(entry, "start" if direction == -1 else "end")

    def _exit_history_browsing(self) -> None:
        self._history_index = -1
        self._history_draft = None

    def _set_text_internal(self, text: str, cursor_placement: Literal["start", "end"] = "end") -> None:
        lines = text.split("\n")
        self._state.lines = lines if lines else [""]
        self._state.cursor_line = 0 if cursor_placement == "start" else len(self._state.lines) - 1
        target_col = 0 if cursor_placement == "start" else len(self._state.lines[self._state.cursor_line])
        self._set_cursor_col(target_col)
        self._scroll_offset = 0

        if self.on_change:
            self.on_change(self.get_text())

    # ------------------------------------------------------------------
    # Text mutation.
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")

    def _insert_text_at_cursor_internal(self, text: str) -> None:
        if not text:
            return

        normalized = self._normalize_text(text)
        inserted_lines = normalized.split("\n")

        current_line = (
            self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        )
        before_cursor = current_line[: self._state.cursor_col]
        after_cursor = current_line[self._state.cursor_col :]

        if len(inserted_lines) == 1:
            self._state.lines[self._state.cursor_line] = before_cursor + normalized + after_cursor
            self._set_cursor_col(self._state.cursor_col + len(normalized))
        else:
            new_lines: list[str] = list(self._state.lines[: self._state.cursor_line])
            new_lines.append(before_cursor + inserted_lines[0])
            new_lines.extend(inserted_lines[1:-1])
            new_lines.append(inserted_lines[-1] + after_cursor)
            new_lines.extend(self._state.lines[self._state.cursor_line + 1 :])
            self._state.lines = new_lines
            self._state.cursor_line += len(inserted_lines) - 1
            self._set_cursor_col(len(inserted_lines[-1]))

        if self.on_change:
            self.on_change(self.get_text())

    def _insert_character(self, char: str, skip_undo_coalescing: bool = False) -> None:
        self._exit_history_browsing()

        if not skip_undo_coalescing:
            if is_whitespace_char(char) or self._last_action != "type-word":
                self._push_undo_snapshot()
            self._last_action = "type-word"

        line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        before = line[: self._state.cursor_col]
        after = line[self._state.cursor_col :]

        self._state.lines[self._state.cursor_line] = before + char + after
        self._set_cursor_col(self._state.cursor_col + len(char))

        if self.on_change:
            self.on_change(self.get_text())

        if not self._autocomplete_state:
            if char == "/" and self._is_at_start_of_message():
                self._try_trigger_autocomplete()
            elif char in self._autocomplete_trigger_characters:
                current_line = self._state.lines[self._state.cursor_line]
                text_before_cursor = current_line[: self._state.cursor_col]
                char_before_symbol = (
                    text_before_cursor[len(text_before_cursor) - 2] if len(text_before_cursor) >= 2 else None
                )
                if len(text_before_cursor) == 1 or char_before_symbol == " " or char_before_symbol == "\t":
                    self._try_trigger_autocomplete()
            elif re.match(r"[a-zA-Z0-9.\-_]", char):
                current_line = self._state.lines[self._state.cursor_line]
                text_before_cursor = current_line[: self._state.cursor_col]
                if self._is_in_slash_command_context(text_before_cursor) or self._autocomplete_trigger_pattern.search(
                    text_before_cursor
                ):
                    self._try_trigger_autocomplete()
        else:
            self._update_autocomplete()

    def _handle_paste(self, pasted_text: str) -> None:
        self._cancel_autocomplete()
        self._exit_history_browsing()
        self._last_action = None

        self._push_undo_snapshot()

        # tmux popups with extended-keys-format=csi-u re-encode control bytes
        # inside bracketed paste as `\x1b[<code>;5u` sequences. Decode Ctrl+letter
        # back to its literal byte so the per-char filter below preserves
        # newlines instead of stripping ESC and leaking the printable tail
        # (e.g. `[106;5u`) into the editor.
        def _decode_csi_u(match: re.Match[str]) -> str:
            cp = int(match.group(1))
            if 97 <= cp <= 122:
                return chr(cp - 96)
            if 65 <= cp <= 90:
                return chr(cp - 64)
            return match.group(0)

        decoded_text = re.sub(r"\x1b\[(\d+);5u", _decode_csi_u, pasted_text)

        clean_text = self._normalize_text(decoded_text)
        filtered_text = "".join(ch for ch in clean_text if ch == "\n" or ord(ch) >= 32)

        # If pasting a file path (starts with /, ~, .) and the char before the
        # cursor is a word character, prepend a space for readability.
        if filtered_text and filtered_text[0] in "/~.":
            current_line = self._state.lines[self._state.cursor_line]
            char_before_cursor = current_line[self._state.cursor_col - 1] if self._state.cursor_col > 0 else ""
            if char_before_cursor and re.match(r"\w", char_before_cursor):
                filtered_text = " " + filtered_text

        pasted_lines = filtered_text.split("\n")
        total_chars = len(filtered_text)
        if len(pasted_lines) > 10 or total_chars > 1000:
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered_text

            if len(pasted_lines) > 10:
                marker = f"[paste #{paste_id} +{len(pasted_lines)} lines]"
            else:
                marker = f"[paste #{paste_id} {total_chars} chars]"
            self._insert_text_at_cursor_internal(marker)
            return

        # Single- and multi-line normal paste: same code path.
        self._insert_text_at_cursor_internal(filtered_text)

    def _add_new_line(self) -> None:
        self._cancel_autocomplete()
        self._exit_history_browsing()
        self._last_action = None

        self._push_undo_snapshot()

        current_line = self._state.lines[self._state.cursor_line]
        before = current_line[: self._state.cursor_col]
        after = current_line[self._state.cursor_col :]

        self._state.lines[self._state.cursor_line] = before
        self._state.lines.insert(self._state.cursor_line + 1, after)

        self._state.cursor_line += 1
        self._set_cursor_col(0)

        if self.on_change:
            self.on_change(self.get_text())

    def _should_submit_on_backslash_enter(self, data: str, kb) -> bool:
        if self.disable_submit:
            return False
        if not matches_key(data, "enter"):
            return False
        submit_keys = kb.get_keys("tui.input.submit")
        has_shift_enter = "shift+enter" in submit_keys or "shift+return" in submit_keys
        if not has_shift_enter:
            return False

        current_line = self._state.lines[self._state.cursor_line]
        return self._state.cursor_col > 0 and current_line[self._state.cursor_col - 1] == "\\"

    def _submit_value(self) -> None:
        self._cancel_autocomplete()
        result = self._expand_paste_markers("\n".join(self._state.lines)).strip()

        self._state = _EditorState()
        self._pastes.clear()
        self._paste_counter = 0
        self._exit_history_browsing()
        self._scroll_offset = 0
        self._undo_stack.clear()
        self._last_action = None

        if self.on_change:
            self.on_change("")
        if self.on_submit:
            self.on_submit(result)

    def _expand_paste_markers(self, text: str) -> str:
        result = text
        for paste_id, paste_content in self._pastes.items():
            marker_regex = re.compile(rf"\[paste #{paste_id}( (\+\d+ lines|\d+ chars))?\]")
            result = marker_regex.sub(lambda _m, content=paste_content: content, result)
        return result

    # ------------------------------------------------------------------
    # Deletion.
    # ------------------------------------------------------------------

    def _handle_backspace(self) -> None:
        self._exit_history_browsing()
        self._last_action = None

        if self._state.cursor_col > 0:
            self._push_undo_snapshot()

            line = self._state.lines[self._state.cursor_line]
            before_cursor = line[: self._state.cursor_col]

            graphemes = self._segment(before_cursor, "grapheme")
            last_grapheme = graphemes[-1] if graphemes else None
            grapheme_length = len(last_grapheme.segment) if last_grapheme else 1
            is_pasted_segmented = PASTE_MARKER_SINGLE.match(last_grapheme.segment) if last_grapheme else None

            if is_pasted_segmented is not None:
                target_id = int(is_pasted_segmented.group(1))
                self._pastes.pop(target_id, None)
                self._paste_counter -= 1

                # Shift registry entries down in ascending id order,
                # independent of marker order in the text: `[paste #3]`
                # becomes `[paste #2]` when `[paste #1]` is removed.
                higher_ids = sorted(pid for pid in self._pastes if pid > target_id)
                for pid in higher_ids:
                    self._pastes[pid - 1] = self._pastes.pop(pid)

                # Renumber markers with ids greater than the removed one.
                def _renumber(match: re.Match[str]) -> str:
                    x = int(match.group(1))
                    if x <= target_id:
                        return match.group(0)
                    suffix = match.group(2) or ""
                    return f"[paste #{x - 1}{suffix}]"

                self._state.lines = [PASTE_MARKER_REGEX.sub(_renumber, ln) for ln in self._state.lines]

            line = self._state.lines[self._state.cursor_line]
            before = line[: self._state.cursor_col - grapheme_length]
            after = line[self._state.cursor_col :]
            self._state.lines[self._state.cursor_line] = before + after
            self._set_cursor_col(self._state.cursor_col - grapheme_length)
        elif self._state.cursor_line > 0:
            self._push_undo_snapshot()

            current_line = self._state.lines[self._state.cursor_line]
            previous_line = self._state.lines[self._state.cursor_line - 1]

            self._state.lines[self._state.cursor_line - 1] = previous_line + current_line
            del self._state.lines[self._state.cursor_line]

            self._state.cursor_line -= 1
            self._set_cursor_col(len(previous_line))

        if self.on_change:
            self.on_change(self.get_text())

        # Update or re-trigger autocomplete after backspace.
        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            current_line = self._state.lines[self._state.cursor_line]
            text_before_cursor = current_line[: self._state.cursor_col]
            if self._is_in_slash_command_context(text_before_cursor) or self._autocomplete_trigger_pattern.search(
                text_before_cursor
            ):
                self._try_trigger_autocomplete()

    def _handle_forward_delete(self) -> None:
        self._exit_history_browsing()
        self._last_action = None

        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col < len(current_line):
            self._push_undo_snapshot()

            after_cursor = current_line[self._state.cursor_col :]
            graphemes = self._segment(after_cursor, "grapheme")
            first_grapheme = graphemes[0] if graphemes else None
            grapheme_length = len(first_grapheme.segment) if first_grapheme else 1

            before = current_line[: self._state.cursor_col]
            after = current_line[self._state.cursor_col + grapheme_length :]
            self._state.lines[self._state.cursor_line] = before + after
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo_snapshot()

            next_line = self._state.lines[self._state.cursor_line + 1]
            self._state.lines[self._state.cursor_line] = current_line + next_line
            del self._state.lines[self._state.cursor_line + 1]

        if self.on_change:
            self.on_change(self.get_text())

        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            current_line = self._state.lines[self._state.cursor_line]
            text_before_cursor = current_line[: self._state.cursor_col]
            if self._is_in_slash_command_context(text_before_cursor) or self._autocomplete_trigger_pattern.search(
                text_before_cursor
            ):
                self._try_trigger_autocomplete()

    def _delete_to_start_of_line(self) -> None:
        self._exit_history_browsing()

        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col > 0:
            self._push_undo_snapshot()

            deleted_text = current_line[: self._state.cursor_col]
            self._kill_ring.push(deleted_text, prepend=True, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            self._state.lines[self._state.cursor_line] = current_line[self._state.cursor_col :]
            self._set_cursor_col(0)
        elif self._state.cursor_line > 0:
            self._push_undo_snapshot()

            self._kill_ring.push("\n", prepend=True, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            previous_line = self._state.lines[self._state.cursor_line - 1]
            self._state.lines[self._state.cursor_line - 1] = previous_line + current_line
            del self._state.lines[self._state.cursor_line]
            self._state.cursor_line -= 1
            self._set_cursor_col(len(previous_line))

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_to_end_of_line(self) -> None:
        self._exit_history_browsing()

        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col < len(current_line):
            self._push_undo_snapshot()

            deleted_text = current_line[self._state.cursor_col :]
            self._kill_ring.push(deleted_text, prepend=False, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            self._state.lines[self._state.cursor_line] = current_line[: self._state.cursor_col]
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo_snapshot()

            self._kill_ring.push("\n", prepend=False, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            next_line = self._state.lines[self._state.cursor_line + 1]
            self._state.lines[self._state.cursor_line] = current_line + next_line
            del self._state.lines[self._state.cursor_line + 1]

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_backwards(self) -> None:
        self._exit_history_browsing()

        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._push_undo_snapshot()

                self._kill_ring.push("\n", prepend=True, accumulate=self._last_action == "kill")
                self._last_action = "kill"

                previous_line = self._state.lines[self._state.cursor_line - 1]
                self._state.lines[self._state.cursor_line - 1] = previous_line + current_line
                del self._state.lines[self._state.cursor_line]
                self._state.cursor_line -= 1
                self._set_cursor_col(len(previous_line))
        else:
            self._push_undo_snapshot()

            was_kill = self._last_action == "kill"

            old_cursor_col = self._state.cursor_col
            self._move_word_backwards()
            delete_from = self._state.cursor_col
            self._set_cursor_col(old_cursor_col)

            deleted_text = current_line[delete_from : self._state.cursor_col]
            self._kill_ring.push(deleted_text, prepend=True, accumulate=was_kill)
            self._last_action = "kill"

            self._state.lines[self._state.cursor_line] = (
                current_line[:delete_from] + current_line[self._state.cursor_col :]
            )
            self._set_cursor_col(delete_from)

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_forward(self) -> None:
        self._exit_history_browsing()

        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col >= len(current_line):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._push_undo_snapshot()

                self._kill_ring.push("\n", prepend=False, accumulate=self._last_action == "kill")
                self._last_action = "kill"

                next_line = self._state.lines[self._state.cursor_line + 1]
                self._state.lines[self._state.cursor_line] = current_line + next_line
                del self._state.lines[self._state.cursor_line + 1]
        else:
            self._push_undo_snapshot()

            was_kill = self._last_action == "kill"

            old_cursor_col = self._state.cursor_col
            self._move_word_forwards()
            delete_to = self._state.cursor_col
            self._set_cursor_col(old_cursor_col)

            deleted_text = current_line[self._state.cursor_col : delete_to]
            self._kill_ring.push(deleted_text, prepend=False, accumulate=was_kill)
            self._last_action = "kill"

            self._state.lines[self._state.cursor_line] = (
                current_line[: self._state.cursor_col] + current_line[delete_to:]
            )

        if self.on_change:
            self.on_change(self.get_text())

    # ------------------------------------------------------------------
    # Kill-ring yank.
    # ------------------------------------------------------------------

    def _yank(self) -> None:
        if self._kill_ring.length == 0:
            return

        self._push_undo_snapshot()

        text = self._kill_ring.peek() or ""
        self._insert_yanked_text(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return

        self._push_undo_snapshot()

        self._delete_yanked_text()

        self._kill_ring.rotate()

        text = self._kill_ring.peek() or ""
        self._insert_yanked_text(text)
        self._last_action = "yank"

    def _insert_yanked_text(self, text: str) -> None:
        self._exit_history_browsing()
        lines = text.split("\n")

        if len(lines) == 1:
            current_line = self._state.lines[self._state.cursor_line]
            before = current_line[: self._state.cursor_col]
            after = current_line[self._state.cursor_col :]
            self._state.lines[self._state.cursor_line] = before + text + after
            self._set_cursor_col(self._state.cursor_col + len(text))
        else:
            current_line = self._state.lines[self._state.cursor_line]
            before = current_line[: self._state.cursor_col]
            after = current_line[self._state.cursor_col :]

            self._state.lines[self._state.cursor_line] = before + lines[0]

            for i in range(1, len(lines) - 1):
                self._state.lines.insert(self._state.cursor_line + i, lines[i])

            last_line_index = self._state.cursor_line + len(lines) - 1
            self._state.lines.insert(last_line_index, lines[-1] + after)

            self._state.cursor_line = last_line_index
            self._set_cursor_col(len(lines[-1]))

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_yanked_text(self) -> None:
        yanked_text = self._kill_ring.peek()
        if not yanked_text:
            return

        yank_lines = yanked_text.split("\n")

        if len(yank_lines) == 1:
            current_line = self._state.lines[self._state.cursor_line]
            delete_len = len(yanked_text)
            before = current_line[: self._state.cursor_col - delete_len]
            after = current_line[self._state.cursor_col :]
            self._state.lines[self._state.cursor_line] = before + after
            self._set_cursor_col(self._state.cursor_col - delete_len)
        else:
            start_line = self._state.cursor_line - (len(yank_lines) - 1)
            start_col = len(self._state.lines[start_line]) - len(yank_lines[0])

            after_cursor = self._state.lines[self._state.cursor_line][self._state.cursor_col :]
            before_yank = self._state.lines[start_line][:start_col]

            del self._state.lines[start_line : start_line + len(yank_lines)]
            self._state.lines.insert(start_line, before_yank + after_cursor)

            self._state.cursor_line = start_line
            self._set_cursor_col(start_col)

        if self.on_change:
            self.on_change(self.get_text())

    # ------------------------------------------------------------------
    # Cursor movement.
    # ------------------------------------------------------------------

    def _set_cursor_col(self, col: int) -> None:
        """Set cursor column and clear preferred_visual_col."""
        self._state.cursor_col = col
        self._preferred_visual_col = None
        self._snapped_from_cursor_col = None

    def _move_to_line_start(self) -> None:
        self._last_action = None
        self._set_cursor_col(0)

    def _move_to_line_end(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line]
        self._set_cursor_col(len(current_line))

    def _move_word_backwards(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._state.cursor_line -= 1
                prev_line = self._state.lines[self._state.cursor_line]
                self._set_cursor_col(len(prev_line))
            return

        # Segmenting: word_navigation uses `.segment` and `.is_word_like`.
        # Provide marker-aware word segmentation from this editor's paste
        # registry.
        options = WordNavigationOptions(
            segment=lambda t: self._segment(t, "word"),
            is_atomic_segment=_is_paste_marker,
        )
        self._set_cursor_col(find_word_backward(current_line, self._state.cursor_col, options))

    def _move_word_forwards(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line]

        if self._state.cursor_col >= len(current_line):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._state.cursor_line += 1
                self._set_cursor_col(0)
            return

        options = WordNavigationOptions(
            segment=lambda t: self._segment(t, "word"),
            is_atomic_segment=_is_paste_marker,
        )
        self._set_cursor_col(find_word_forward(current_line, self._state.cursor_col, options))

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        self._last_action = None
        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)

        if delta_line != 0:
            target_visual_line = current_visual_line + delta_line
            if 0 <= target_visual_line < len(visual_lines):
                self._move_to_visual_line(visual_lines, current_visual_line, target_visual_line)

        if delta_col != 0:
            current_line = self._state.lines[self._state.cursor_line]

            if delta_col > 0:
                if self._state.cursor_col < len(current_line):
                    after_cursor = current_line[self._state.cursor_col :]
                    graphemes = self._segment(after_cursor, "grapheme")
                    first_grapheme = graphemes[0] if graphemes else None
                    self._set_cursor_col(
                        self._state.cursor_col + (len(first_grapheme.segment) if first_grapheme else 1)
                    )
                elif self._state.cursor_line < len(self._state.lines) - 1:
                    self._state.cursor_line += 1
                    self._set_cursor_col(0)
                else:
                    current_vl = visual_lines[current_visual_line] if current_visual_line < len(visual_lines) else None
                    if current_vl is not None:
                        self._preferred_visual_col = self._state.cursor_col - current_vl.start_col
            else:
                if self._state.cursor_col > 0:
                    before_cursor = current_line[: self._state.cursor_col]
                    graphemes = self._segment(before_cursor, "grapheme")
                    last_grapheme = graphemes[-1] if graphemes else None
                    self._set_cursor_col(self._state.cursor_col - (len(last_grapheme.segment) if last_grapheme else 1))
                elif self._state.cursor_line > 0:
                    self._state.cursor_line -= 1
                    prev_line = self._state.lines[self._state.cursor_line]
                    self._set_cursor_col(len(prev_line))

        # Keep an open autocomplete picker in sync with the new cursor
        # position: cursor movement changes text before the cursor, so a
        # picker computed for the old position is stale. Re-query so it
        # refreshes or closes when the new position yields no suggestions.
        if self._autocomplete_state:
            self._update_autocomplete()

    def _page_scroll(self, direction: int) -> None:
        self._last_action = None
        terminal_rows = self._tui.terminal.rows
        page_size = max(5, int(terminal_rows * 0.3))

        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)
        target_visual_line = max(0, min(len(visual_lines) - 1, current_visual_line + direction * page_size))

        self._move_to_visual_line(visual_lines, current_visual_line, target_visual_line)

    def _move_to_visual_line(
        self,
        visual_lines: list[_VisualLine],
        current_visual_line: int,
        target_visual_line: int,
    ) -> None:
        current_vl = visual_lines[current_visual_line]
        target_vl = visual_lines[target_visual_line]

        if self._snapped_from_cursor_col is not None:
            vl_index = self._find_visual_line_at(visual_lines, current_vl.logical_line, self._snapped_from_cursor_col)
            current_visual_col = self._snapped_from_cursor_col - visual_lines[vl_index].start_col
        else:
            current_visual_col = self._state.cursor_col - current_vl.start_col

        is_last_source_segment = (
            current_visual_line == len(visual_lines) - 1
            or visual_lines[current_visual_line + 1].logical_line != current_vl.logical_line
        )
        source_max_visual_col = current_vl.length if is_last_source_segment else max(0, current_vl.length - 1)

        is_last_target_segment = (
            target_visual_line == len(visual_lines) - 1
            or visual_lines[target_visual_line + 1].logical_line != target_vl.logical_line
        )
        target_max_visual_col = target_vl.length if is_last_target_segment else max(0, target_vl.length - 1)

        move_to_visual_col = self._compute_vertical_move_column(
            current_visual_col, source_max_visual_col, target_max_visual_col
        )

        self._state.cursor_line = target_vl.logical_line
        target_col = target_vl.start_col + move_to_visual_col
        logical_line = self._state.lines[target_vl.logical_line]
        self._state.cursor_col = min(target_col, len(logical_line))

        # Snap cursor to atomic-segment boundaries (e.g. paste markers) so
        # the cursor never lands in the middle of a multi-grapheme unit.
        segments = self._segment(logical_line, "grapheme")
        for seg in segments:
            if seg.index > self._state.cursor_col:
                break
            if len(seg.segment) <= 1:
                continue
            if self._state.cursor_col < seg.index + len(seg.segment):
                is_continuation = seg.index < target_vl.start_col
                is_moving_down = target_visual_line > current_visual_line

                if is_continuation and is_moving_down:
                    # Segment started on a previous visual line and we
                    # already visited it on the way down. Skip all remaining
                    # continuation VLs and land on the first VL past it.
                    seg_end = seg.index + len(seg.segment)
                    nxt = target_visual_line + 1
                    while (
                        nxt < len(visual_lines)
                        and visual_lines[nxt].logical_line == target_vl.logical_line
                        and visual_lines[nxt].start_col < seg_end
                    ):
                        nxt += 1
                    if nxt < len(visual_lines):
                        self._move_to_visual_line(visual_lines, current_visual_line, nxt)
                        return

                # Snap to the segment start so it gets highlighted. Store
                # the pre-snap position for the next vertical move.
                self._snapped_from_cursor_col = self._state.cursor_col
                self._state.cursor_col = seg.index
                return

        self._snapped_from_cursor_col = None

    def _compute_vertical_move_column(
        self, current_visual_col: int, source_max_visual_col: int, target_max_visual_col: int
    ) -> int:
        """Sticky-column decision table for vertical cursor movement (see the
        table in editor.ts for the full case matrix)."""
        has_preferred = self._preferred_visual_col is not None
        cursor_in_middle = current_visual_col < source_max_visual_col
        target_too_short = target_max_visual_col < current_visual_col

        if (not has_preferred) or cursor_in_middle:
            if target_too_short:
                # Cases 2 and 7.
                self._preferred_visual_col = current_visual_col
                return target_max_visual_col

            # Cases 1 and 6.
            self._preferred_visual_col = None
            return current_visual_col

        preferred = self._preferred_visual_col
        assert preferred is not None
        target_cant_fit_preferred = target_max_visual_col < preferred
        if target_too_short or target_cant_fit_preferred:
            # Cases 4 and 5.
            return target_max_visual_col

        # Case 3.
        result = preferred
        self._preferred_visual_col = None
        return result

    # ------------------------------------------------------------------
    # Character jump.
    # ------------------------------------------------------------------

    def _jump_to_char(self, char: str, direction: str) -> None:
        self._last_action = None
        is_forward = direction == _JUMP_MODE_FORWARD
        lines = self._state.lines

        end = len(lines) if is_forward else -1
        step = 1 if is_forward else -1

        line_idx = self._state.cursor_line
        while line_idx != end:
            line = lines[line_idx]
            is_current_line = line_idx == self._state.cursor_line

            if is_current_line:
                search_from = self._state.cursor_col + 1 if is_forward else self._state.cursor_col - 1
            else:
                search_from = None

            if is_forward:
                idx = line.find(char, search_from if search_from is not None else 0)
            else:
                if search_from is None:
                    idx = line.rfind(char)
                elif search_from < 0:
                    # JavaScript clamps a negative fromIndex to 0, so
                    # `lastIndexOf(char, -1)` still examines index 0.
                    idx = 0 if line.startswith(char) else -1
                else:
                    idx = line.rfind(char, 0, search_from + 1)

            if idx != -1:
                self._state.cursor_line = line_idx
                self._set_cursor_col(idx)
                return
            line_idx += step

    # ------------------------------------------------------------------
    # Undo.
    # ------------------------------------------------------------------

    def _push_undo_snapshot(self) -> None:
        snapshot = _EditorSnapshot(
            state=_EditorState(
                lines=list(self._state.lines),
                cursor_line=self._state.cursor_line,
                cursor_col=self._state.cursor_col,
            ),
            pastes=dict(self._pastes),
            paste_counter=self._paste_counter,
        )
        self._undo_stack.push(snapshot)

    def _undo(self) -> None:
        self._exit_history_browsing()
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._state.lines = snapshot.state.lines
        self._state.cursor_line = snapshot.state.cursor_line
        self._state.cursor_col = snapshot.state.cursor_col
        self._pastes = snapshot.pastes
        self._paste_counter = snapshot.paste_counter
        self._last_action = None
        self._preferred_visual_col = None
        if self.on_change:
            self.on_change(self.get_text())

    # ------------------------------------------------------------------
    # Slash / autocomplete helpers.
    # ------------------------------------------------------------------

    def _is_slash_menu_allowed(self) -> bool:
        return self._state.cursor_line == 0

    def _is_at_start_of_message(self) -> bool:
        if not self._is_slash_menu_allowed():
            return False
        current_line = self._state.lines[self._state.cursor_line]
        before_cursor = current_line[: self._state.cursor_col]
        stripped = before_cursor.strip()
        return stripped == "" or stripped == "/"

    def _is_in_slash_command_context(self, text_before_cursor: str) -> bool:
        return self._is_slash_menu_allowed() and text_before_cursor.lstrip().startswith("/")

    def _get_best_autocomplete_match_index(self, items: list[AutocompleteItem] | list[SelectItem], prefix: str) -> int:
        if not prefix:
            return -1
        first_prefix_index = -1
        for i, item in enumerate(items):
            value = item.value
            if value == prefix:
                return i
            if first_prefix_index == -1 and value.startswith(prefix):
                first_prefix_index = i
        return first_prefix_index

    def _create_autocomplete_list(self, prefix: str, items: list[AutocompleteItem]) -> SelectList:
        select_items = [SelectItem(value=item.value, label=item.label, description=item.description) for item in items]
        layout = _SLASH_COMMAND_SELECT_LIST_LAYOUT if prefix.startswith("/") else None
        return SelectList(select_items, self._autocomplete_max_visible, self._theme.select_list, layout)

    def _try_trigger_autocomplete(self, explicit_tab: bool = False) -> None:
        self._request_autocomplete(force=False, explicit_tab=explicit_tab)

    def _handle_tab_completion(self) -> None:
        if self._autocomplete_provider is None:
            return
        current_line = self._state.lines[self._state.cursor_line]
        before_cursor = current_line[: self._state.cursor_col]

        if self._is_in_slash_command_context(before_cursor) and " " not in before_cursor.lstrip():
            self._handle_slash_command_completion()
        else:
            self._force_file_autocomplete(True)

    def _handle_slash_command_completion(self) -> None:
        self._request_autocomplete(force=False, explicit_tab=True)

    def _force_file_autocomplete(self, explicit_tab: bool = False) -> None:
        self._request_autocomplete(force=True, explicit_tab=explicit_tab)

    def _request_autocomplete(self, *, force: bool, explicit_tab: bool) -> None:
        if self._autocomplete_provider is None:
            return

        if force:
            should_trigger_fn = getattr(self._autocomplete_provider, "should_trigger_file_completion", None)
            if should_trigger_fn is not None and not should_trigger_fn(
                self._state.lines, self._state.cursor_line, self._state.cursor_col
            ):
                return

        self._cancel_autocomplete_request()
        self._autocomplete_start_token += 1
        start_token = self._autocomplete_start_token

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        debounce_ms = self._get_autocomplete_debounce_ms(force=force, explicit_tab=explicit_tab)
        if debounce_ms > 0:

            def _on_debounce_fire() -> None:
                self._autocomplete_debounce_handle = None
                self._autocomplete_dispatch_task = loop.create_task(
                    self._start_autocomplete_request(start_token, force=force, explicit_tab=explicit_tab)
                )

            self._autocomplete_debounce_handle = loop.call_later(debounce_ms / 1000.0, _on_debounce_fire)
            return

        self._autocomplete_dispatch_task = loop.create_task(
            self._start_autocomplete_request(start_token, force=force, explicit_tab=explicit_tab)
        )

    async def _start_autocomplete_request(self, start_token: int, *, force: bool, explicit_tab: bool) -> None:
        previous_task = self._autocomplete_request_task

        async def _chained() -> None:
            if previous_task is not None:
                with contextlib.suppress(Exception):
                    await previous_task
            if start_token != self._autocomplete_start_token or self._autocomplete_provider is None:
                return

            signal = asyncio.Event()
            self._autocomplete_abort = signal
            self._autocomplete_request_id += 1
            request_id = self._autocomplete_request_id
            snapshot_text = self.get_text()
            snapshot_line = self._state.cursor_line
            snapshot_col = self._state.cursor_col

            await self._run_autocomplete_request(
                request_id,
                signal,
                snapshot_text,
                snapshot_line,
                snapshot_col,
                force=force,
                explicit_tab=explicit_tab,
            )

        task = asyncio.ensure_future(_chained())
        self._autocomplete_request_task = task
        with contextlib.suppress(Exception):
            await task

    def _set_autocomplete_trigger_characters(self, trigger_characters: list[str]) -> None:
        result = list(_DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS)
        for character in trigger_characters:
            if len(character) != 1 or character == "/" or is_whitespace_char(character) or character in result:
                continue
            result.append(character)
        self._autocomplete_trigger_characters = result
        self._autocomplete_trigger_pattern = _build_trigger_pattern(result)
        self._autocomplete_debounce_pattern = _build_debounce_pattern(result)

    def _get_autocomplete_debounce_ms(self, *, force: bool, explicit_tab: bool) -> int:
        if explicit_tab or force:
            return 0
        current_line = self._state.lines[self._state.cursor_line]
        text_before_cursor = current_line[: self._state.cursor_col]
        if self._autocomplete_debounce_pattern.search(text_before_cursor):
            return _ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS
        return 0

    async def _run_autocomplete_request(
        self,
        request_id: int,
        signal: asyncio.Event,
        snapshot_text: str,
        snapshot_line: int,
        snapshot_col: int,
        *,
        force: bool,
        explicit_tab: bool,
    ) -> None:
        provider = self._autocomplete_provider
        if provider is None:
            return

        try:
            suggestions = await provider.get_suggestions(
                self._state.lines,
                self._state.cursor_line,
                self._state.cursor_col,
                signal=signal,
                force=force,
            )
        except Exception:
            return

        if not self._is_autocomplete_request_current(request_id, signal, snapshot_text, snapshot_line, snapshot_col):
            return

        self._autocomplete_abort = None

        if suggestions is None or not isinstance(suggestions.items, list) or len(suggestions.items) == 0:
            self._cancel_autocomplete()
            self._tui.request_render()
            return

        if force and explicit_tab and len(suggestions.items) == 1:
            item = suggestions.items[0]
            self._push_undo_snapshot()
            self._last_action = None
            result = provider.apply_completion(
                self._state.lines,
                self._state.cursor_line,
                self._state.cursor_col,
                item,
                suggestions.prefix,
            )
            self._state.lines = result.lines
            self._state.cursor_line = result.cursor_line
            self._set_cursor_col(result.cursor_col)
            if self.on_change:
                self.on_change(self.get_text())
            self._tui.request_render()
            return

        self._apply_autocomplete_suggestions(suggestions, "force" if force else "regular")
        self._tui.request_render()

    def _is_autocomplete_request_current(
        self,
        request_id: int,
        signal: asyncio.Event,
        snapshot_text: str,
        snapshot_line: int,
        snapshot_col: int,
    ) -> bool:
        return (
            not signal.is_set()
            and request_id == self._autocomplete_request_id
            and self.get_text() == snapshot_text
            and self._state.cursor_line == snapshot_line
            and self._state.cursor_col == snapshot_col
        )

    def _apply_autocomplete_suggestions(
        self, suggestions: AutocompleteSuggestions, state: Literal["regular", "force"]
    ) -> None:
        self._autocomplete_prefix = suggestions.prefix
        self._autocomplete_list = self._create_autocomplete_list(suggestions.prefix, suggestions.items)

        best_match_index = self._get_best_autocomplete_match_index(suggestions.items, suggestions.prefix)
        if best_match_index >= 0:
            self._autocomplete_list.set_selected_index(best_match_index)

        self._autocomplete_state = state

    def _cancel_autocomplete_request(self) -> None:
        self._autocomplete_start_token += 1
        if self._autocomplete_debounce_handle is not None:
            self._autocomplete_debounce_handle.cancel()
            self._autocomplete_debounce_handle = None
        if self._autocomplete_abort is not None:
            self._autocomplete_abort.set()
        self._autocomplete_abort = None

    def _clear_autocomplete_ui(self) -> None:
        self._autocomplete_state = None
        self._autocomplete_list = None
        self._autocomplete_prefix = ""

    def _cancel_autocomplete(self) -> None:
        self._cancel_autocomplete_request()
        self._clear_autocomplete_ui()

    def _update_autocomplete(self) -> None:
        if not self._autocomplete_state or self._autocomplete_provider is None:
            return
        self._request_autocomplete(force=self._autocomplete_state == "force", explicit_tab=False)


def _select_item_to_autocomplete_item(item: SelectItem) -> AutocompleteItem:
    return AutocompleteItem(value=item.value, label=item.label, description=item.description)
