"""Python port of `packages/tui/test/truncated-text.test.ts`.

The TypeScript suite builds styled input with `new Chalk({ level: 3 })` so the
ANSI assertions are deterministic. There is no chalk in the Python port, so the
same sequences are written literally: `chalk.red(s)` is `\\x1b[31m{s}\\x1b[39m`
and `chalk.blue(s)` is `\\x1b[34m{s}\\x1b[39m`. Every rendered line is compared
against the exact bytes the TypeScript implementation produces for the same
input, not just its visible width.
"""

from __future__ import annotations

import re

from pi_tui.components.truncated_text import TruncatedText
from pi_tui.utils import visible_width

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _red(text: str) -> str:
    return f"\x1b[31m{text}\x1b[39m"


def _blue(text: str) -> str:
    return f"\x1b[34m{text}\x1b[39m"


class TestTruncatedTextComponent:
    def test_pads_output_lines_to_exactly_match_width(self) -> None:
        text = TruncatedText("Hello world", 1, 0)
        lines = text.render(50)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 50
        assert lines[0] == " Hello world" + " " * 38

    def test_pads_output_with_vertical_padding_lines_to_width(self) -> None:
        text = TruncatedText("Hello", 0, 2)
        lines = text.render(40)

        # 2 padding lines + 1 content line + 2 padding lines
        assert len(lines) == 5
        for line in lines:
            assert visible_width(line) == 40
        assert lines[0] == " " * 40
        assert lines[2] == "Hello" + " " * 35
        assert lines[4] == " " * 40

    def test_truncates_long_text_and_pads_to_width(self) -> None:
        long_text = "This is a very long piece of text that will definitely exceed the available width"
        text = TruncatedText(long_text, 1, 0)
        lines = text.render(30)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 30
        assert "..." in _ANSI.sub("", lines[0])
        assert lines[0] == " This is a very long piece\x1b[0m...\x1b[0m "

    def test_preserves_ansi_codes_in_output_and_pads_correctly(self) -> None:
        styled_text = f"{_red('Hello')} {_blue('world')}"
        text = TruncatedText(styled_text, 1, 0)
        lines = text.render(40)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 40
        assert "\x1b[" in lines[0]
        assert lines[0] == " \x1b[31mHello\x1b[39m \x1b[34mworld\x1b[39m" + " " * 28

    def test_truncates_styled_text_and_adds_reset_code_before_ellipsis(self) -> None:
        long_styled_text = _red("This is a very long red text that will be truncated")
        text = TruncatedText(long_styled_text, 1, 0)
        lines = text.render(20)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 20
        assert "\x1b[0m..." in lines[0]
        assert lines[0] == " \x1b[31mThis is a very \x1b[0m...\x1b[0m "

    def test_handles_text_that_fits_exactly(self) -> None:
        # With padding_x=1 the available width is 30-2=28; "Hello world" is 11 wide.
        text = TruncatedText("Hello world", 1, 0)
        lines = text.render(30)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 30
        assert "..." not in _ANSI.sub("", lines[0])
        assert lines[0] == " Hello world" + " " * 18

    def test_handles_empty_text(self) -> None:
        text = TruncatedText("", 1, 0)
        lines = text.render(30)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 30
        assert lines[0] == " " * 30

    def test_stops_at_newline_and_only_shows_first_line(self) -> None:
        multiline_text = "First line\nSecond line\nThird line"
        text = TruncatedText(multiline_text, 1, 0)
        lines = text.render(40)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 40

        stripped = _ANSI.sub("", lines[0]).strip()
        assert "First line" in stripped
        assert "Second line" not in stripped
        assert "Third line" not in stripped
        assert lines[0] == " First line" + " " * 29

    def test_truncates_first_line_even_with_newlines_in_text(self) -> None:
        long_multiline_text = "This is a very long first line that needs truncation\nSecond line"
        text = TruncatedText(long_multiline_text, 1, 0)
        lines = text.render(25)

        assert len(lines) == 1
        assert visible_width(lines[0]) == 25

        stripped = _ANSI.sub("", lines[0])
        assert "..." in stripped
        assert "Second line" not in stripped
        assert lines[0] == " This is a very long \x1b[0m...\x1b[0m "
