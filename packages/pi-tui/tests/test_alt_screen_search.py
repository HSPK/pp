"""Python port of the search cases in `packages/tui/test/tui-alt-screen.test.ts`.

The upstream assertion in "searches normalized rendered transcript text across
rows" is reproduced exactly, including its literal column numbers, because it
pins the part of this feature that is easy to get subtly wrong: a match is
located in a whitespace-normalised corpus and then mapped *back* to per-row
column ranges in the rendered output.
"""

from __future__ import annotations

from pi_tui.alt_screen_search import (
    find_alt_screen_search_matches,
    get_alt_screen_search_match_key,
)


def _spans(match):
    return [(s.row, s.start_col, s.end_col) for s in match.segments]


def test_searches_normalized_rendered_transcript_text_across_rows():
    """Verbatim port of the upstream case, expected values included.

    The query spans a row boundary and differs in case; neither is visible in
    the rendered lines, which is the whole reason the corpus exists.
    """
    matches = find_alt_screen_search_matches(["alpha QUICK", "brown fox"], "quick brown")

    assert len(matches) == 1
    assert _spans(matches[0]) == [(0, 6, 11), (1, 0, 5)]


def test_a_match_inside_one_row_is_a_single_segment():
    """Per-character spans are merged, or a plain hit would highlight in pieces."""
    matches = find_alt_screen_search_matches(["hello world"], "world")

    assert _spans(matches[0]) == [(0, 6, 11)]


def test_matching_ignores_terminal_sequences():
    """Rendered rows carry ANSI colour; searching the raw string would miss."""
    matches = find_alt_screen_search_matches(["\x1b[31mred\x1b[39m alert"], "red alert")

    assert len(matches) == 1
    assert _spans(matches[0]) == [(0, 0, 3), (0, 4, 9)]


def test_runs_of_whitespace_collapse_to_one_separator():
    """A wrapped line pads with spaces; the user should not have to reproduce them."""
    matches = find_alt_screen_search_matches(["one     two"], "one two")

    assert len(matches) == 1


def test_a_blank_query_matches_nothing():
    assert find_alt_screen_search_matches(["anything"], "") == []
    assert find_alt_screen_search_matches(["anything"], "   ") == []


def test_every_occurrence_is_returned():
    matches = find_alt_screen_search_matches(["ab", "ab"], "ab")

    assert [_spans(m) for m in matches] == [[(0, 0, 2)], [(1, 0, 2)]]


def test_match_keys_identify_a_hit_across_rows():
    """The key keeps the cursor on the same hit as the transcript re-renders."""
    matches = find_alt_screen_search_matches(["alpha QUICK", "brown fox"], "quick brown")

    assert get_alt_screen_search_match_key(matches[0]) == "0:6:1:5"
