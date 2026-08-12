"""Word-boundary cursor navigation for `packages/tui/src/word-navigation.ts`.

Uses `pi_tui.utils.iter_word_segments` in place of the TypeScript source's
shared `Intl.Segmenter(undefined, { granularity: "word" })` instance. See
`pi_tui.utils` for the approximation this implies.

Indexing note: the TypeScript source indexes `cursor` positions by UTF-16
code unit (`string.length`/`slice`). This port indexes by Python code point
instead (see `pi_tui.utils` module docstring for the general rationale) so a
`cursor` value here is a code-point offset, not a UTF-16 offset.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pi_tui.utils import PUNCTUATION_REGEX, WordSegment, is_whitespace_char, iter_word_segments

_PUNCTUATION_GLOBAL_REGEX = re.compile(PUNCTUATION_REGEX.pattern)


@dataclass
class WordNavigationOptions:
    """Options for word navigation functions.

    When omitted, uses the default `iter_word_segments` word segmentation.
    """

    segment: Callable[[str], Iterable[WordSegment]] | None = None
    is_atomic_segment: Callable[[str], bool] | None = None


def find_word_backward(text: str, cursor: int, options: WordNavigationOptions | None = None) -> int:
    """Find the cursor position after moving one word backward from `cursor`.

    Skips trailing whitespace, then stops at the next word/punctuation
    boundary. Pure function - does not mutate any state.
    """
    if cursor <= 0:
        return 0

    text_before_cursor = text[:cursor]
    segment_fn = options.segment if options else None
    is_atomic = options.is_atomic_segment if options else None
    segments = list(segment_fn(text_before_cursor)) if segment_fn else iter_word_segments(text_before_cursor)
    new_cursor = cursor

    def last_segment_text() -> str:
        return segments[-1].segment if segments else ""

    # Skip trailing whitespace.
    while segments and not (is_atomic and is_atomic(last_segment_text())) and is_whitespace_char(last_segment_text()):
        new_cursor -= len(segments.pop().segment)

    if not segments:
        return new_cursor

    last = segments[-1]

    if is_atomic and is_atomic(last.segment):
        # Skip one atomic segment.
        new_cursor -= len(last.segment)
    elif last.is_word_like:
        # Skip inside one word-like segment, preserving ASCII punctuation boundaries.
        segment = last.segment
        matches = list(_PUNCTUATION_GLOBAL_REGEX.finditer(segment))
        if not matches:
            new_cursor -= len(segment)
        else:
            last_match = matches[-1]
            new_cursor -= len(segment) - last_match.end()
    else:
        # Skip non-word non-whitespace run (punctuation).
        while (
            segments
            and not (is_atomic and is_atomic(last_segment_text()))
            and not segments[-1].is_word_like
            and not is_whitespace_char(last_segment_text())
        ):
            new_cursor -= len(segments.pop().segment)

    return new_cursor


def find_word_forward(text: str, cursor: int, options: WordNavigationOptions | None = None) -> int:
    """Find the cursor position after moving one word forward from `cursor`.

    Skips leading whitespace, then stops at the next word/punctuation
    boundary. Pure function - does not mutate any state.
    """
    if cursor >= len(text):
        return len(text)

    text_after_cursor = text[cursor:]
    segment_fn = options.segment if options else None
    is_atomic = options.is_atomic_segment if options else None
    segments = list(segment_fn(text_after_cursor)) if segment_fn else iter_word_segments(text_after_cursor)
    index = 0
    new_cursor = cursor

    def current() -> WordSegment | None:
        return segments[index] if index < len(segments) else None

    # Skip leading whitespace.
    seg = current()
    while seg is not None and not (is_atomic and is_atomic(seg.segment)) and is_whitespace_char(seg.segment):
        new_cursor += len(seg.segment)
        index += 1
        seg = current()

    if seg is None:
        return new_cursor

    if is_atomic and is_atomic(seg.segment):
        # Skip one atomic segment.
        new_cursor += len(seg.segment)
    elif seg.is_word_like:
        # Skip inside one word-like segment, preserving ASCII punctuation boundaries.
        match = PUNCTUATION_REGEX.search(seg.segment)
        new_cursor += match.start() if match else len(seg.segment)
    else:
        # Skip non-word non-whitespace run (punctuation).
        while (
            seg is not None
            and not (is_atomic and is_atomic(seg.segment))
            and not seg.is_word_like
            and not is_whitespace_char(seg.segment)
        ):
            new_cursor += len(seg.segment)
            index += 1
            seg = current()

    return new_cursor
