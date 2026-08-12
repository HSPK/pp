"""Python port of `packages/tui/test/regression-overlay-cjk-boundary.test.ts`.

`extract_segments` returns a `(before, before_width, after, after_width)` tuple in
the Python port where TypeScript returns an object with named fields; the
assertions are otherwise identical.
"""

from __future__ import annotations

from pi_tui.tui import composite_tui_line
from pi_tui.utils import extract_segments, slice_by_column, visible_width


def test_excludes_a_wide_grapheme_from_before_when_overlay_starts_inside_it():
    before, before_width, after, after_width = extract_segments("abcd让EFGH", 5, 9, 11, True)

    assert before == "abcd"
    assert before_width == 4
    assert visible_width(before) == before_width
    assert after == "H"
    assert after_width == 1


def test_keeps_ascii_before_segment_behavior_at_the_same_boundary():
    before, before_width, _after, _after_width = extract_segments("abcdG EFGH", 5, 9, 11, True)

    assert before == "abcdG"
    assert before_width == 5
    assert visible_width(before) == before_width


def test_composites_an_overlay_at_the_requested_column_when_it_starts_inside_a_wide_grapheme():
    out = composite_tui_line("abcd让EFGH", "│XX│", 5, 4, 20)
    prefix = slice_by_column(out, 0, 5, True)
    overlay = slice_by_column(out, 5, 4, True)

    assert ("让" in out) is False
    assert visible_width(out) == 20
    assert visible_width(prefix) == 5
    assert visible_width(overlay) == 4
    assert ("│XX│" in overlay) is True


def test_composites_an_overlay_when_it_starts_at_a_wide_grapheme_boundary():
    out = composite_tui_line("abcd让EFGH", "│XX│", 4, 4, 20)
    overlay = slice_by_column(out, 4, 4, True)

    assert ("让" in out) is False
    assert visible_width(out) == 20
    assert visible_width(overlay) == 4
    assert ("│XX│" in overlay) is True
