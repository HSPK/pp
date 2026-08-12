"""Fullscreen transcript search matching.

Python port of `packages/tui/src/alt-screen-search.ts`.

The interesting part is that a match is not a substring of any one rendered
line. Rendered output carries ANSI sequences, wraps across rows and pads with
whitespace, so searching the raw lines finds nothing useful. Instead the lines
are flattened into a whitespace-normalised corpus that remembers, for every
character, which row and column range it came from. A match in the corpus is
then mapped back into per-row highlight segments.

TypeScript uses `Intl.Segmenter` to walk the line by grapheme; this port uses
`iter_graphemes`, which is the same boundary logic (see `utils.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .utils import iter_graphemes, strip_terminal_sequences, visible_width


@dataclass
class AltScreenSearchSegment:
    """One highlight run: a column range on a single rendered row."""

    row: int
    start_col: int
    end_col: int


@dataclass
class AltScreenSearchMatch:
    """A whole match, which may span several rows."""

    segments: list[AltScreenSearchSegment] = field(default_factory=list)


@dataclass
class _SearchCorpus:
    text: str = ""
    # One entry per character of `text`; `None` marks a synthetic separator that
    # maps to no rendered cell.
    source: list[AltScreenSearchSegment | None] = field(default_factory=list)


_WHITESPACE_RUN = re.compile(r"\s+", re.UNICODE)
_ALL_WHITESPACE = re.compile(r"^\s+$", re.UNICODE)


def _append_mapped_text(text: str, span: AltScreenSearchSegment | None, corpus: _SearchCorpus) -> None:
    corpus.text += text
    corpus.source.extend([span] * len(text))


def _build_search_corpus(lines: list[str]) -> _SearchCorpus:
    """Flatten rendered rows into searchable text plus a per-character origin map.

    Every run of whitespace -- including a row boundary -- collapses to a single
    space, and only when something follows it. That is what lets a query match
    across a wrap without the user having to know where the wrap fell.
    """
    corpus = _SearchCorpus()
    pending_separator = False

    for row, raw in enumerate(lines):
        line = strip_terminal_sequences(raw or "")
        column = 0
        for grapheme in iter_graphemes(line):
            width = visible_width(grapheme)
            if _ALL_WHITESPACE.match(grapheme):
                if corpus.text:
                    pending_separator = True
                column += width
                continue
            if pending_separator:
                _append_mapped_text(" ", None, corpus)
                pending_separator = False
            _append_mapped_text(
                grapheme,
                AltScreenSearchSegment(row=row, start_col=column, end_col=column + width),
                corpus,
            )
            column += width
        if corpus.text:
            pending_separator = True

    return corpus


def _normalize_query(query: str) -> str:
    return _WHITESPACE_RUN.sub(" ", query).strip()


def find_alt_screen_search_matches(lines: list[str], query: str) -> list[AltScreenSearchMatch]:
    """Locate `query` in rendered `lines`, case-insensitively.

    Port of `findAltScreenSearchMatches`. Adjacent characters on the same row
    are merged into one segment so a plain match highlights as a single run
    rather than one segment per character.
    """
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    corpus = _build_search_corpus(lines)
    expression = re.compile(re.escape(normalized_query), re.IGNORECASE | re.UNICODE)
    matches: list[AltScreenSearchMatch] = []

    for match in expression.finditer(corpus.text):
        segments: list[AltScreenSearchSegment] = []
        for index in range(match.start(), match.end()):
            span = corpus.source[index]
            if span is None:
                continue
            previous = segments[-1] if segments else None
            if previous is not None and previous.row == span.row and span.start_col <= previous.end_col:
                previous.end_col = max(previous.end_col, span.end_col)
            else:
                segments.append(AltScreenSearchSegment(row=span.row, start_col=span.start_col, end_col=span.end_col))
        if segments:
            matches.append(AltScreenSearchMatch(segments=segments))

    return matches


def get_alt_screen_search_match_key(match: AltScreenSearchMatch) -> str:
    """Stable identity for a match, used to keep the cursor on the same hit.

    Port of `getAltScreenSearchMatchKey`.
    """
    if not match.segments:
        return ""
    first = match.segments[0]
    last = match.segments[-1]
    return f"{first.row}:{first.start_col}:{last.row}:{last.end_col}"
