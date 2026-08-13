"""Syntax highlighting, port of `highlightCode` / `getLanguageFromPath`.

Upstream highlights through `cli-highlight` (highlight.js); this port uses
Pygments. The tokenizer is not what has to match -- the *colours* are. Upstream
maps highlight.js scopes onto the active theme's `syntax*` entries
(`theme.ts:1137`), which is what makes highlighted code follow a theme switch
instead of sitting in a library's default palette.

This was previously unported, so file contents rendered without colour at all.
"""

from __future__ import annotations

import re

import pytest
from pi_coding_agent.modes.interactive.theme.theme import (
    get_language_from_path,
    get_markdown_theme,
    highlight_code,
    init_theme,
    theme,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


@pytest.fixture(autouse=True)
def _theme():
    init_theme("dark")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("a/b.py", "python"),
        ("x.rs", "rust"),
        ("Foo.tsx", "typescript"),
        ("s.SH", "bash"),
        ("d.yml", "yaml"),
        ("noext", None),
        ("x.unknownext", None),
    ],
)
def test_language_is_derived_from_the_extension(path, expected):
    assert get_language_from_path(path) == expected


def test_keywords_and_numbers_use_the_theme_syntax_colours():
    """The specific assertion that would fail against a default palette."""
    lines = highlight_code("return 1", "python")

    assert theme.fg("syntaxKeyword", "return") in lines[0]
    assert theme.fg("syntaxNumber", "1") in lines[0]


def test_unknown_language_falls_back_to_the_code_block_colour():
    """Upstream disables auto-detection deliberately.

    highlight.js misreads prose as AppleScript and colours random English words
    as keywords, so no-language input is tinted rather than guessed at.
    """
    lines = highlight_code("just some prose", None)

    assert lines == [theme.fg("mdCodeBlock", "just some prose")]


@pytest.mark.parametrize(
    "code, lang",
    [("a\nb\nc", "python"), ("x", "python"), ("", "python"), ("a\n\nb", "python"), ("a\nb", None)],
)
def test_line_count_and_text_are_preserved(code, lang):
    """The caller slices these lines and counts them for the "N more" hint.

    A tokenizer that swallows a blank line or splits differently would shift
    every line number the user sees.
    """
    lines = highlight_code(code, lang)

    assert [_plain(line) for line in lines] == code.split("\n")


def test_a_style_never_spans_a_newline():
    """A colour span crossing a row boundary leaves the reset on the wrong row."""
    for line in highlight_code("def f():\n    return 1", "python"):
        assert "\n" not in line


def test_an_unusable_language_does_not_raise():
    assert [_plain(x) for x in highlight_code("x = 1", "definitely-not-a-language")] == ["x = 1"]


def test_the_markdown_theme_uses_the_highlighter():
    """`highlight_code` existed but was never handed to `MarkdownTheme`.

    `Markdown` only highlights a fenced block when `theme.highlight_code` is
    set, so leaving it at its `None` default made every code block in the
    transcript render flat, however complete the highlighter itself was.
    """
    markdown_theme = get_markdown_theme()

    assert markdown_theme.highlight_code is not None

    lines = markdown_theme.highlight_code("def f():\n    return 1", "python")
    assert [_plain(line) for line in lines] == ["def f():", "    return 1"]
    assert theme.fg("syntaxKeyword", "def") in lines[0]
