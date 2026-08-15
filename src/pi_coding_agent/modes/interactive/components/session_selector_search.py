"""Session-selector query parsing, matching and sorting.

Ported from ``packages/coding-agent/src/modes/interactive/components/session-selector-search.ts``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pi_tui.fuzzy import fuzzy_match

if TYPE_CHECKING:
    from ....core.session_manager import SessionInfo

SortMode = Literal["threaded", "recent", "relevance"]
NameFilter = Literal["all", "named"]
TokenKind = Literal["fuzzy", "phrase"]

_WHITESPACE_RUN_RE = re.compile(r"\s+")


@dataclass
class SearchToken:
    kind: TokenKind
    value: str


@dataclass
class ParsedSearchQuery:
    mode: Literal["tokens", "regex"]
    tokens: list[SearchToken] = field(default_factory=list)
    regex: re.Pattern[str] | None = None
    error: str | None = None
    """When set, parsing failed and the query should match nothing."""


@dataclass
class MatchResult:
    matches: bool
    score: float = 0.0
    """Lower is better; only meaningful when ``matches`` is True."""


def _normalize_whitespace_lower(text: str) -> str:
    return _WHITESPACE_RUN_RE.sub(" ", text.lower()).strip()


def get_session_search_text(session: SessionInfo) -> str:
    return f"{session.id} {session.name or ''} {session.all_messages_text} {session.cwd}"


def has_session_name(session: SessionInfo) -> bool:
    return bool(session.name and session.name.strip())


def _matches_name_filter(session: SessionInfo, name_filter: NameFilter) -> bool:
    if name_filter == "all":
        return True
    return has_session_name(session)


def parse_search_query(query: str) -> ParsedSearchQuery:
    trimmed = query.strip()
    if not trimmed:
        return ParsedSearchQuery(mode="tokens")

    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return ParsedSearchQuery(mode="regex", error="Empty regex")
        try:
            return ParsedSearchQuery(mode="regex", regex=re.compile(pattern, re.IGNORECASE))
        except re.error as error:
            return ParsedSearchQuery(mode="regex", error=str(error))

    # Token mode with quote support, e.g. `foo "node cve" bar`.
    tokens: list[SearchToken] = []
    buffer = ""
    in_quote = False
    had_unclosed_quote = False

    def flush(kind: TokenKind) -> None:
        nonlocal buffer
        value = buffer.strip()
        buffer = ""
        if value:
            tokens.append(SearchToken(kind=kind, value=value))

    for char in trimmed:
        if char == '"':
            if in_quote:
                flush("phrase")
                in_quote = False
            else:
                flush("fuzzy")
                in_quote = True
            continue
        if not in_quote and char.isspace():
            flush("fuzzy")
            continue
        buffer += char

    if in_quote:
        had_unclosed_quote = True

    if had_unclosed_quote:
        # Unbalanced quotes fall back to plain whitespace tokenization.
        return ParsedSearchQuery(
            mode="tokens",
            tokens=[
                SearchToken(kind="fuzzy", value=part)
                for part in (piece.strip() for piece in _WHITESPACE_RUN_RE.split(trimmed))
                if part
            ],
        )

    flush("phrase" if in_quote else "fuzzy")
    return ParsedSearchQuery(mode="tokens", tokens=tokens)


def match_session(session: SessionInfo, parsed: ParsedSearchQuery) -> MatchResult:
    text = get_session_search_text(session)

    if parsed.mode == "regex":
        if parsed.regex is None:
            return MatchResult(matches=False)
        found = parsed.regex.search(text)
        if found is None:
            return MatchResult(matches=False)
        return MatchResult(matches=True, score=found.start() * 0.1)

    if len(parsed.tokens) == 0:
        return MatchResult(matches=True)

    total_score = 0.0
    normalized_text: str | None = None

    for token in parsed.tokens:
        if token.kind == "phrase":
            if normalized_text is None:
                normalized_text = _normalize_whitespace_lower(text)
            phrase = _normalize_whitespace_lower(token.value)
            if not phrase:
                continue
            index = normalized_text.find(phrase)
            if index < 0:
                return MatchResult(matches=False)
            total_score += index * 0.1
            continue

        match = fuzzy_match(token.value, text)
        if not match.matches:
            return MatchResult(matches=False)
        total_score += match.score

    return MatchResult(matches=True, score=total_score)


def filter_and_sort_sessions(
    sessions: list[SessionInfo],
    query: str,
    sort_mode: SortMode,
    name_filter: NameFilter = "all",
) -> list[SessionInfo]:
    name_filtered = (
        sessions
        if name_filter == "all"
        else [session for session in sessions if _matches_name_filter(session, name_filter)]
    )
    if not query.strip():
        return name_filtered

    parsed = parse_search_query(query)
    if parsed.error:
        return []

    if sort_mode == "recent":
        # Filter only; the incoming order is already newest-first.
        return [session for session in name_filtered if match_session(session, parsed).matches]

    scored: list[tuple[SessionInfo, float]] = []
    for session in name_filtered:
        result = match_session(session, parsed)
        if result.matches:
            scored.append((session, result.score))

    # Sort by score, tie-broken by most recently modified. Python's sort is
    # stable, so sorting on the negated timestamp inside the key matches the
    # TypeScript comparator exactly.
    scored.sort(key=lambda pair: (pair[1], -pair[0].modified.timestamp()))
    return [session for session, _score in scored]


__all__ = [
    "MatchResult",
    "NameFilter",
    "ParsedSearchQuery",
    "SearchToken",
    "SortMode",
    "filter_and_sort_sessions",
    "get_session_search_text",
    "has_session_name",
    "match_session",
    "parse_search_query",
]
