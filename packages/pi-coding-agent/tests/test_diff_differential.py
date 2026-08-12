"""Differential tests against the real upstream JavaScript implementations.

The fixtures in ``tests/data/`` were produced by running the actual reference
code, not by recording this port's own output:

* ``jsdiff_word_*.json`` -- ``diffWords`` from the ``diff`` npm package at
  version 8.0.4, the exact version pinned by
  ``packages/coding-agent/package.json`` upstream.
* ``render_diff_*.json`` -- the upstream
  ``modes/interactive/components/diff.ts`` ``renderDiff``, executed under Node
  with its type annotations stripped and ``theme`` stubbed to emit
  ``<name>text</name>`` / ``[INV]text[/INV]`` markers.

Regenerating them requires Node plus the ``diff`` tarball; the fixtures are
checked in so the comparison runs offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_coding_agent.modes.interactive.components import diff as diff_module
from pi_coding_agent.utils.word_diff import diff_words

_DATA_DIR = Path(__file__).parent / "data"


class _MarkerTheme:
    """Matches the theme stub used when capturing the reference output."""

    def fg(self, name: str, text: str) -> str:
        return f"<{name}>{text}</{name}>"

    def inverse(self, text: str) -> str:
        return f"[INV]{text}[/INV]"


def _load(name: str) -> list:
    return json.loads((_DATA_DIR / name).read_text(encoding="utf-8"))


def test_diff_words_matches_jsdiff_8_0_4():
    cases = _load("jsdiff_word_cases.json")
    expected = _load("jsdiff_word_expected.json")
    assert len(cases) == len(expected)

    mismatches = []
    for (old, new), want in zip(cases, expected, strict=True):
        got = [[change.value, change.added, change.removed] for change in diff_words(old, new)]
        if got != want:
            mismatches.append((old, new, want, got))

    assert mismatches == [], f"{len(mismatches)} of {len(cases)} cases differ; first: {mismatches[0]}"


def test_render_diff_matches_upstream_typescript(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(diff_module, "theme", _MarkerTheme())
    cases = _load("render_diff_cases.json")
    expected = _load("render_diff_expected.json")
    assert len(cases) == len(expected)

    mismatches = []
    for case, want in zip(cases, expected, strict=True):
        got = diff_module.render_diff(case)
        if got != want:
            mismatches.append((case, want, got))

    assert mismatches == [], f"{len(mismatches)} of {len(cases)} cases differ; first: {mismatches[0]}"


def test_fixtures_are_non_trivial():
    # Guard against a regeneration accident silently emptying the corpus.
    assert len(_load("jsdiff_word_cases.json")) >= 200
    assert len(_load("render_diff_cases.json")) >= 100
