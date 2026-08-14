"""Tests ported from packages/tui/test/markdown.test.ts.

The upstream test file uses `chalk` (level 3, i.e. full ANSI SGR codes) for a
`defaultMarkdownTheme` fixture; `_chalk_theme()` below reproduces the exact
SGR sequences chalk would emit for that theme (bold=1/22, dim=2/22,
italic=3/23, underline=4/24, strikethrough=9/29, and standard 16-color
codes), following chalk's nested-styler algorithm (`openAll = outerOpen +
innerOpen`, `closeAll = innerClose + outerClose`) so `chalk.bold.cyan(text)`
matches byte-for-byte.

The upstream `VirtualTerminal`/`@xterm/headless` cases in "Pre-styled text"
and "Heading with inline code" read real terminal cell attributes
(`isItalic()`, `isUnderline()`) after driving a full TUI event loop. This
port drives the same TUI against `MiniTerminalModel` (see `pi_tui.testing`),
which tracks the italic and underline SGR attributes per cell.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from pi_tui.components.markdown import DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme
from pi_tui.terminal_image import TerminalCapabilities, reset_capabilities_cache, set_capabilities
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui_main_screen import TuiMainScreen

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line)


# SGR (open, close) pairs used by the default theme / pre-styled-text tests.
_BOLD = (1, 22)
_DIM = (2, 22)
_ITALIC = (3, 23)
_UNDERLINE = (4, 24)
_STRIKETHROUGH = (9, 29)
_BLUE = (34, 39)
_YELLOW = (33, 39)
_GREEN = (32, 39)
_CYAN = (36, 39)
_GRAY = (90, 39)
_MAGENTA = (35, 39)


def _chalk(*pairs: tuple[int, int]):
    """Build a function reproducing chalk's nested nested-styler output.

    `_chalk(BOLD, CYAN)` matches `chalk.bold.cyan`: opens are concatenated in
    call order, closes are concatenated in reverse order, e.g.
    "\\x1b[1m\\x1b[36m" + text + "\\x1b[39m\\x1b[22m".
    """
    opens = "".join(f"\x1b[{o}m" for o, _c in pairs)
    closes = "".join(f"\x1b[{c}m" for _o, c in reversed(pairs))

    def styled(text: str) -> str:
        return f"{opens}{text}{closes}"

    return styled


def default_markdown_theme() -> MarkdownTheme:
    """Python port of test-themes.ts's `defaultMarkdownTheme` (chalk level 3)."""
    return MarkdownTheme(
        heading=_chalk(_BOLD, _CYAN),
        link=_chalk(_BLUE),
        link_url=_chalk(_DIM),
        code=_chalk(_YELLOW),
        code_block=_chalk(_GREEN),
        code_block_border=_chalk(_DIM),
        quote=_chalk(_ITALIC),
        quote_border=_chalk(_DIM),
        hr=_chalk(_DIM),
        list_bullet=_chalk(_CYAN),
        bold=_chalk(_BOLD),
        italic=_chalk(_ITALIC),
        strikethrough=_chalk(_STRIKETHROUGH),
        underline=_chalk(_UNDERLINE),
    )


class TestTransforms:
    def test_caches_transformed_markdown_by_source_and_available_width(self):
        calls = []

        def transform(source: str, available_width: int) -> str:
            calls.append((source, available_width))
            return f"{source} {available_width}"

        markdown = Markdown("source", 2, 0, default_markdown_theme(), None, MarkdownOptions(transform=transform))

        assert [strip_ansi(line).strip() for line in markdown.render(80)] == ["source 76"]
        markdown.render(80)
        assert [strip_ansi(line).strip() for line in markdown.render(60)] == ["source 56"]
        assert calls == [("source", 76), ("source", 56)]

        markdown.set_text("updated")
        assert [strip_ansi(line).strip() for line in markdown.render(60)] == ["updated 56"]
        assert calls[-1] == ("updated", 56)

        markdown.invalidate()
        markdown.render(60)
        assert calls[-1] == ("updated", 56)
        assert len(calls) == 4


class TestLists:
    def test_renders_simple_nested_list(self):
        markdown = Markdown("- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        assert len(lines) > 0
        plain_lines = [strip_ansi(line) for line in lines]
        assert any("- Item 1" in line for line in plain_lines)
        assert any("    - Nested 1.1" in line for line in plain_lines)
        assert any("    - Nested 1.2" in line for line in plain_lines)
        assert any("- Item 2" in line for line in plain_lines)

    def test_renders_deeply_nested_list(self):
        markdown = Markdown("- Level 1\n  - Level 2\n    - Level 3\n      - Level 4", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        assert any("- Level 1" in line for line in plain_lines)
        assert any("    - Level 2" in line for line in plain_lines)
        assert any("        - Level 3" in line for line in plain_lines)
        assert any("            - Level 4" in line for line in plain_lines)

    def test_renders_ordered_nested_list(self):
        markdown = Markdown(
            "1. First\n   1. Nested first\n   2. Nested second\n2. Second", 0, 0, default_markdown_theme()
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        assert any("1. First" in line for line in plain_lines)
        assert any("    1. Nested first" in line for line in plain_lines)
        assert any("    2. Nested second" in line for line in plain_lines)
        assert any("2. Second" in line for line in plain_lines)

    def test_normalizes_ordered_list_markers_by_default(self):
        markdown = Markdown("1. alpha\n1. beta\n1. gamma", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines == ["1. alpha", "2. beta", "3. gamma"]

    def test_preserves_source_list_markers_when_configured(self):
        markdown = Markdown(
            "  4. forth\n  3. third\n\n10) ten\n7) seven\n\n+ plus\n* star\n- minus\n+",
            0,
            0,
            default_markdown_theme(),
            None,
            MarkdownOptions(preserve_ordered_list_markers=True),
        )
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines == [
            "4. forth",
            "3. third",
            "",
            "10) ten",
            "7) seven",
            "",
            "+ plus",
            "* star",
            "- minus",
            "+",
        ]

    def test_renders_mixed_ordered_and_unordered_nested_lists(self):
        markdown = Markdown(
            "1. Ordered item\n   - Unordered nested\n   - Another nested\n2. Second ordered\n   - More nested",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        assert any("1. Ordered item" in line for line in plain_lines)
        assert any("    - Unordered nested" in line for line in plain_lines)
        assert any("2. Second ordered" in line for line in plain_lines)

    def test_renders_blank_lines_between_loose_list_items(self):
        markdown = Markdown(
            "1. Lorem ipsum dolor sit amet.\n\n"
            "   Ut enim ad minim veniam.\n\n"
            "2. Duis aute irure dolor.\n\n"
            "   Excepteur sint occaecat cupidatat.\n\n"
            "3. Beep boop",
            0,
            0,
            default_markdown_theme(),
        )
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines == [
            "1. Lorem ipsum dolor sit amet.",
            "",
            "   Ut enim ad minim veniam.",
            "",
            "2. Duis aute irure dolor.",
            "",
            "   Excepteur sint occaecat cupidatat.",
            "",
            "3. Beep boop",
        ]

    def test_renders_task_list_markers(self):
        markdown = Markdown("- [ ] beep\n- [x] boop", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines == ["- [ ] beep", "- [x] boop"]

    def test_maintains_numbering_when_code_blocks_are_not_indented(self):
        markdown = Markdown(
            "1. First item\n\n```typescript\n// code block\n```\n\n"
            "2. Second item\n\n```typescript\n// another code block\n```\n\n"
            "3. Third item",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line).strip() for line in markdown.render(80)]
        numbered_lines = [line for line in plain_lines if re.match(r"^\d+\.", line)]
        assert len(numbered_lines) == 3, f"Expected 3 numbered items, got: {numbered_lines}"
        assert numbered_lines[0].startswith("1.")
        assert numbered_lines[1].startswith("2.")
        assert numbered_lines[2].startswith("3.")

    def test_indents_wrapped_unordered_list_lines(self):
        markdown = Markdown("- alpha beta gamma delta epsilon", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(20)]
        assert lines == ["- alpha beta gamma", "  delta epsilon"]

    def test_indents_wrapped_ordered_list_lines(self):
        markdown = Markdown("1. alpha beta gamma delta epsilon", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(20)]
        assert lines == ["1. alpha beta gamma", "   delta epsilon"]

    def test_indents_wrapped_ordered_list_lines_with_multi_digit_markers(self):
        markdown = Markdown("10. alpha beta gamma delta epsilon", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(21)]
        assert lines == ["10. alpha beta gamma", "    delta epsilon"]

    def test_indents_wrapped_nested_list_lines(self):
        markdown = Markdown("- parent\n  - alpha beta gamma delta epsilon", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]
        assert lines == ["- parent", "    - alpha beta gamma", "      delta epsilon"]

    def test_indents_wrapped_nested_list_lines_under_ordered_parents(self):
        markdown = Markdown("1. parent\n   - alpha beta gamma delta epsilon", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]
        assert lines == ["1. parent", "    - alpha beta gamma", "      delta epsilon"]

    def test_renders_and_wraps_blockquotes_inside_list_items(self):
        markdown = Markdown("- > alpha beta gamma delta epsilon zeta", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]
        assert lines == ["- │ alpha beta gamma", "  │ delta epsilon zeta"]

    def test_renders_and_wraps_code_blocks_inside_list_items(self):
        markdown = Markdown("- ```ts\n  alpha beta gamma delta epsilon zeta\n  ```", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]
        assert lines == ["- ```ts", "    alpha beta gamma", "  delta epsilon zeta", "  ```"]


class TestTables:
    def test_renders_simple_table(self):
        markdown = Markdown(
            "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |", 0, 0, default_markdown_theme()
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        assert any("Name" in line for line in plain_lines)
        assert any("Age" in line for line in plain_lines)
        assert any("Alice" in line for line in plain_lines)
        assert any("Bob" in line for line in plain_lines)
        assert any("│" in line for line in plain_lines)
        assert any("─" in line for line in plain_lines)

    def test_renders_row_dividers_between_data_rows(self):
        markdown = Markdown(
            "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |", 0, 0, default_markdown_theme()
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        divider_lines = [line for line in plain_lines if "┼" in line]
        assert len(divider_lines) == 2

    def test_keeps_column_width_at_least_the_longest_word(self):
        longest_word = "superlongword"
        markdown = Markdown(
            f"| Column One | Column Two |\n| --- | --- |\n| {longest_word} short | otherword |\n| small | tiny |",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(32)]
        data_line = next(line for line in plain_lines if longest_word in line)
        segments = data_line.split("│")[1:-1]
        first_segment = segments[0]
        first_column_width = len(first_segment) - 2
        assert first_column_width >= len(longest_word)

    def test_renders_table_with_alignment(self):
        markdown = Markdown(
            "| Left | Center | Right |\n| :--- | :---: | ---: |\n| A | B | C |\n| Long text | Middle | End |",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        assert any("Left" in line for line in plain_lines)
        assert any("Center" in line for line in plain_lines)
        assert any("Right" in line for line in plain_lines)
        assert any("Long text" in line for line in plain_lines)

    def test_handles_tables_with_varying_column_widths(self):
        markdown = Markdown(
            "| Short | Very long column header |\n| --- | --- |\n"
            "| A | This is a much longer cell content |\n| B | Short |",
            0,
            0,
            default_markdown_theme(),
        )
        lines = markdown.render(80)
        assert len(lines) > 0
        plain_lines = [strip_ansi(line) for line in lines]
        assert any("Very long column header" in line for line in plain_lines)
        assert any("This is a much longer cell content" in line for line in plain_lines)

    def test_wraps_table_cells_when_table_exceeds_available_width(self):
        markdown = Markdown(
            "| Command | Description | Example |\n| --- | --- | --- |\n"
            "| npm install | Install all dependencies | npm install |\n"
            "| npm run build | Build the project | npm run build |",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(50)]
        for line in plain_lines:
            assert len(line) <= 50, f"Line exceeds width 50: {line!r}"
        all_text = " ".join(plain_lines)
        assert "Command" in all_text
        assert "Description" in all_text
        assert "npm install" in all_text
        assert "Install" in all_text

    def test_wraps_long_cell_content_to_multiple_lines(self):
        markdown = Markdown(
            "| Header |\n| --- |\n| This is a very long cell content that should wrap |",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(25)]
        data_rows = [line for line in plain_lines if line.startswith("│") and "─" not in line]
        assert len(data_rows) > 2
        all_text = " ".join(plain_lines)
        assert "very long" in all_text
        assert "cell content" in all_text
        assert "should wrap" in all_text

    def test_wraps_long_unbroken_tokens_inside_table_cells(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        url = "https://example.com/this/is/a/very/long/url/that/should/wrap"
        markdown = Markdown(f"| Value |\n| --- |\n| prefix {url} |", 0, 0, default_markdown_theme())
        width = 30
        lines = markdown.render(width)
        reset_capabilities_cache()
        plain_lines = [strip_ansi(line).rstrip() for line in lines]

        for line in plain_lines:
            assert len(line) <= width, f"Line exceeds width {width}: {line!r}"

        table_lines = [line for line in plain_lines if line.startswith("│")]
        assert len(table_lines) > 0
        for line in table_lines:
            border_count = line.count("│")
            assert border_count == 2, f"Expected 2 borders, got {border_count}: {line!r}"

        extracted = re.sub(r"[│├┤─\s]", "", "".join(plain_lines))
        assert "prefix" in extracted
        assert url in extracted

    def test_wraps_styled_inline_code_inside_table_cells_without_breaking_borders(self):
        markdown = Markdown("| Code |\n| --- |\n| `averyveryveryverylongidentifier` |", 0, 0, default_markdown_theme())
        width = 20
        lines = markdown.render(width)
        joined_output = "\n".join(lines)
        assert "\x1b[33m" in joined_output

        plain_lines = [strip_ansi(line).rstrip() for line in lines]
        for line in plain_lines:
            assert len(line) <= width

        table_lines = [line for line in plain_lines if line.startswith("│")]
        for line in table_lines:
            border_count = line.count("│")
            assert border_count == 2

    def test_handles_extremely_narrow_width_gracefully(self):
        markdown = Markdown("| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |", 0, 0, default_markdown_theme())
        lines = markdown.render(15)
        plain_lines = [strip_ansi(line).rstrip() for line in lines]
        assert len(lines) > 0
        for line in plain_lines:
            assert len(line) <= 15

    def test_renders_table_correctly_when_it_fits_naturally(self):
        markdown = Markdown("| A | B |\n| --- | --- |\n| 1 | 2 |", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        header_line = next((line for line in plain_lines if "A" in line and "B" in line), None)
        assert header_line is not None
        assert "│" in header_line
        separator_line = next((line for line in plain_lines if "├" in line and "┼" in line), None)
        assert separator_line is not None
        data_line = next((line for line in plain_lines if "1" in line and "2" in line), None)
        assert data_line is not None

    def test_respects_padding_x_when_calculating_table_width(self):
        markdown = Markdown(
            "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |", 2, 0, default_markdown_theme()
        )
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(40)]
        for line in plain_lines:
            assert len(line) <= 40
        table_row = next((line for line in plain_lines if "│" in line), None)
        assert table_row is not None
        assert table_row.startswith("  ")

    def test_does_not_add_trailing_blank_line_when_table_is_last(self):
        markdown = Markdown("| Name |\n| --- |\n| Alice |", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain_lines[-1] != ""


class TestCombinedFeatures:
    def test_renders_lists_and_tables_together(self):
        markdown = Markdown(
            "# Test Document\n\n- Item 1\n  - Nested item\n- Item 2\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        assert any("Test Document" in line for line in plain_lines)
        assert any("- Item 1" in line for line in plain_lines)
        assert any("    - Nested item" in line for line in plain_lines)
        assert any("Col1" in line for line in plain_lines)
        assert any("│" in line for line in plain_lines)


class TestBackslashEscapes:
    def test_normalizes_escaped_punctuation_by_default(self):
        markdown = Markdown('"\\"', 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines == ['""']

    def test_preserves_source_backslash_escapes_when_configured(self):
        markdown = Markdown(
            '"\\"', 0, 0, default_markdown_theme(), None, MarkdownOptions(preserve_backslash_escapes=True)
        )
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines == ['"\\"']


class TestPreStyledText:
    def test_preserves_gray_italic_styling_after_inline_code(self):
        markdown = Markdown(
            "This is thinking with `inline code` and more text after",
            1,
            0,
            default_markdown_theme(),
            DefaultTextStyle(color=_chalk(_GRAY), italic=True),
        )
        lines = markdown.render(80)
        joined_output = "\n".join(lines)
        assert "inline code" in joined_output
        assert "\x1b[90m" in joined_output
        assert "\x1b[3m" in joined_output
        assert "\x1b[33m" in joined_output

    def test_preserves_gray_italic_styling_after_bold_text(self):
        markdown = Markdown(
            "This is thinking with **bold text** and more after",
            1,
            0,
            default_markdown_theme(),
            DefaultTextStyle(color=_chalk(_GRAY), italic=True),
        )
        lines = markdown.render(80)
        joined_output = "\n".join(lines)
        assert "bold text" in joined_output
        assert "\x1b[90m" in joined_output
        assert "\x1b[3m" in joined_output
        assert "\x1b[1m" in joined_output

    @pytest.mark.asyncio
    async def test_does_not_leak_styles_into_following_lines_when_rendered_in_tui(self):
        class MarkdownWithInput:
            def __init__(self, markdown: Markdown) -> None:
                self.markdown = markdown
                self.markdown_line_count = 0

            def render(self, width: int) -> list[str]:
                lines = self.markdown.render(width)
                self.markdown_line_count = len(lines)
                return [*lines, "INPUT"]

            def invalidate(self) -> None:
                self.markdown.invalidate()

        markdown = Markdown(
            "This is thinking with `inline code`",
            1,
            0,
            default_markdown_theme(),
            DefaultTextStyle(color=_chalk(_GRAY), italic=True),
        )

        terminal = FakeTerminal(80, 6)
        model = MiniTerminalModel(80, 6)
        tui = TuiMainScreen(terminal)
        component = MarkdownWithInput(markdown)
        tui.add_child(component)
        tui.start()
        await asyncio.sleep(0.03)
        model.feed("".join(terminal.writes))
        terminal.writes.clear()

        assert component.markdown_line_count > 0
        input_row = component.markdown_line_count
        assert model.cell_italic(input_row, 0) is False
        tui.stop()


class TestSpacingAfterCodeBlocks:
    def test_only_one_blank_line_between_code_block_and_following_paragraph(self):
        markdown = Markdown(
            'hello world\n\n```js\nconst hello = "world";\n```\n\nagain, hello world', 0, 0, default_markdown_theme()
        )
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        closing_index = plain_lines.index("```")
        after_backticks = plain_lines[closing_index + 1 :]
        empty_line_count = next((i for i, line in enumerate(after_backticks) if line != ""), len(after_backticks))
        assert empty_line_count == 1

    def test_normalizes_paragraph_and_code_block_spacing_to_one_blank_line(self):
        cases = [
            "hello this is text\n```\ncode block\n```\nmore text",
            "hello this is text\n\n```\ncode block\n```\n\nmore text",
        ]
        expected_lines = ["hello this is text", "", "```", "  code block", "```", "", "more text"]

        for text in cases:
            markdown = Markdown(text, 0, 0, default_markdown_theme())
            plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
            assert plain_lines == expected_lines

    def test_does_not_add_trailing_blank_line_when_code_block_is_last(self):
        cases = ["```js\nconst hello = 'world';\n```", "hello world\n\n```js\nconst hello = 'world';\n```"]
        for text in cases:
            markdown = Markdown(text, 0, 0, default_markdown_theme())
            plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
            assert plain_lines[-1] != ""


class TestSpacingAfterDividers:
    def test_only_one_blank_line_between_divider_and_following_paragraph(self):
        markdown = Markdown("hello world\n\n---\n\nagain, hello world", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        divider_index = next(i for i, line in enumerate(plain_lines) if "─" in line)
        after_divider = plain_lines[divider_index + 1 :]
        empty_line_count = next((i for i, line in enumerate(after_divider) if line != ""), len(after_divider))
        assert empty_line_count == 1

    def test_does_not_add_trailing_blank_line_when_divider_is_last(self):
        markdown = Markdown("---", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain_lines[-1] != ""


class TestSpacingAfterHeadings:
    def test_only_one_blank_line_between_heading_and_following_paragraph(self):
        markdown = Markdown("# Hello\n\nThis is a paragraph", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        heading_index = next(i for i, line in enumerate(plain_lines) if "Hello" in line)
        after_heading = plain_lines[heading_index + 1 :]
        empty_line_count = next((i for i, line in enumerate(after_heading) if line != ""), len(after_heading))
        assert empty_line_count == 1

    def test_does_not_add_trailing_blank_line_when_heading_is_last(self):
        markdown = Markdown("# Hello", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain_lines[-1] != ""


class TestSpacingAfterBlockquotes:
    def test_only_one_blank_line_between_blockquote_and_following_paragraph(self):
        markdown = Markdown("hello world\n\n> This is a quote\n\nagain, hello world", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        quote_index = next(i for i, line in enumerate(plain_lines) if "This is a quote" in line)
        after_quote = plain_lines[quote_index + 1 :]
        empty_line_count = next((i for i, line in enumerate(after_quote) if line != ""), len(after_quote))
        assert empty_line_count == 1

    def test_does_not_add_trailing_blank_line_when_blockquote_is_last(self):
        markdown = Markdown("> This is a quote", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain_lines[-1] != ""


class TestBlockquotesWithMultilineContent:
    def test_applies_consistent_styling_to_all_lines_in_lazy_continuation_blockquote(self):
        markdown = Markdown(">Foo\nbar", 0, 0, default_markdown_theme(), DefaultTextStyle(color=_chalk(_MAGENTA)))
        lines = markdown.render(80)
        plain_lines = [strip_ansi(line) for line in lines]
        quoted_lines = [line for line in plain_lines if line.startswith("│ ")]
        assert len(quoted_lines) == 2

        foo_line = next(line for line in lines if "Foo" in line)
        bar_line = next(line for line in lines if "bar" in line)
        assert "\x1b[3m" in foo_line
        assert "\x1b[3m" in bar_line
        assert "\x1b[35m" not in foo_line
        assert "\x1b[35m" not in bar_line

    def test_applies_consistent_styling_to_explicit_multiline_blockquote(self):
        markdown = Markdown(">Foo\n>bar", 0, 0, default_markdown_theme(), DefaultTextStyle(color=_chalk(_CYAN)))
        lines = markdown.render(80)
        plain_lines = [strip_ansi(line) for line in lines]
        quoted_lines = [line for line in plain_lines if line.startswith("│ ")]
        assert len(quoted_lines) == 2

        foo_line = next(line for line in lines if "Foo" in line)
        bar_line = next(line for line in lines if "bar" in line)
        assert "\x1b[3m" in foo_line
        assert "\x1b[3m" in bar_line
        assert "\x1b[36m" not in foo_line
        assert "\x1b[36m" not in bar_line

    def test_renders_list_content_inside_blockquotes(self):
        markdown = Markdown("> 1. bla bla\n> - nested bullet", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line) for line in markdown.render(80)]
        quoted_lines = [line for line in plain_lines if line.startswith("│ ")]
        assert any("1. bla bla" in line for line in quoted_lines)
        assert any("- nested bullet" in line for line in quoted_lines)

    def test_wraps_long_blockquote_lines_and_adds_border_to_each_wrapped_line(self):
        long_text = "This is a very long blockquote line that should wrap to multiple lines when rendered"
        markdown = Markdown(f"> {long_text}", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(30)]
        content_lines = [line for line in plain_lines if line]
        assert len(content_lines) > 1
        for line in content_lines:
            assert line.startswith("│ ")
        all_text = " ".join(content_lines)
        assert "very long" in all_text
        assert "blockquote" in all_text
        assert "multiple" in all_text

    def test_properly_indents_wrapped_blockquote_lines_with_styling(self):
        markdown = Markdown(
            "> This is styled text that is long enough to wrap",
            0,
            0,
            default_markdown_theme(),
            DefaultTextStyle(color=_chalk(_YELLOW), italic=True),
        )
        lines = markdown.render(25)
        plain_lines = [strip_ansi(line).rstrip() for line in lines]
        content_lines = [line for line in plain_lines if line]
        for line in content_lines:
            assert line.startswith("│ ")
        all_output = "\n".join(lines)
        assert "\x1b[3m" in all_output
        assert "\x1b[33m" not in all_output

    def test_renders_inline_formatting_inside_blockquotes_and_reapplies_quote_styling_after(self):
        markdown = Markdown("> Quote with **bold** and `code`", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        plain_lines = [strip_ansi(line) for line in lines]
        assert any(line.startswith("│ ") for line in plain_lines)
        all_plain = " ".join(plain_lines)
        assert "Quote with" in all_plain
        assert "bold" in all_plain
        assert "code" in all_plain

        all_output = "\n".join(lines)
        assert "\x1b[1m" in all_output
        assert "\x1b[33m" in all_output
        assert "\x1b[3m" in all_output


class TestHeadingWithInlineCode:
    def test_preserves_heading_styling_after_inline_code(self):
        markdown = Markdown("### Why `sourceInfo` should not be optional", 0, 0, default_markdown_theme())
        joined_output = "\n".join(markdown.render(80))
        assert "\x1b[33m" in joined_output
        after_code_index = joined_output.index("should not be optional")
        assert after_code_index > 0
        preceding_chunk = joined_output[max(0, after_code_index - 40) : after_code_index]
        assert "\x1b[1m" in preceding_chunk
        assert "\x1b[36m" in preceding_chunk

    def test_preserves_heading_styling_after_inline_code_for_h1(self):
        markdown = Markdown("# Title with `code` inside", 0, 0, default_markdown_theme())
        joined_output = "\n".join(markdown.render(80))
        after_code_index = joined_output.index("inside")
        assert after_code_index > 0
        preceding_chunk = joined_output[max(0, after_code_index - 40) : after_code_index]
        assert "\x1b[1m" in preceding_chunk
        assert "\x1b[36m" in preceding_chunk
        assert "\x1b[4m" in preceding_chunk

    def test_preserves_heading_styling_after_bold_text(self):
        markdown = Markdown("## Heading with **bold** and more", 0, 0, default_markdown_theme())
        joined_output = "\n".join(markdown.render(80))
        after_bold_index = joined_output.index("and more")
        assert after_bold_index > 0
        preceding_chunk = joined_output[max(0, after_bold_index - 40) : after_bold_index]
        assert "\x1b[1m" in preceding_chunk
        assert "\x1b[36m" in preceding_chunk

    @pytest.mark.asyncio
    async def test_does_not_leak_h1_underline_into_padding_when_code_is_last_token(self):
        markdown = Markdown("# Important distinction from `open()`", 0, 0, default_markdown_theme())
        terminal = FakeTerminal(80, 4)
        model = MiniTerminalModel(80, 4)
        tui = TuiMainScreen(terminal)
        tui.add_child(markdown)
        tui.start()
        await asyncio.sleep(0.03)
        model.feed("".join(terminal.writes))
        terminal.writes.clear()

        rendered_line = markdown.render(80)[0]
        content_width = len(strip_ansi(rendered_line).rstrip())
        assert content_width > 0

        for col in range(content_width, 80):
            assert model.cell_underline(0, col) is False, f"underline leaked into padding at col {col}"

        tui.stop()


class TestStrikethroughSyntax:
    def test_renders_double_tilde_as_strikethrough(self):
        markdown = Markdown("Use ~~strikethrough~~ here", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        joined_output = "\n".join(lines)
        joined_plain = " ".join(strip_ansi(line) for line in lines)
        assert "\x1b[9m" in joined_output
        assert "strikethrough" in joined_plain
        assert "~~strikethrough~~" not in joined_plain

    def test_keeps_single_tilde_as_plain_text(self):
        markdown = Markdown("Use ~strikethrough~ literally", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        joined_output = "\n".join(lines)
        joined_plain = " ".join(strip_ansi(line) for line in lines)
        assert "~strikethrough~" in joined_plain
        assert "\x1b[9m" not in joined_output


class TestLinks:
    def teardown_method(self):
        reset_capabilities_cache()

    def test_does_not_duplicate_url_for_autolinked_emails(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        markdown = Markdown("Contact user@example.com for help", 0, 0, default_markdown_theme())
        joined_plain = " ".join(strip_ansi(line) for line in markdown.render(80))
        assert "user@example.com" in joined_plain
        assert "mailto:" not in joined_plain

    def test_does_not_duplicate_url_for_bare_urls(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        markdown = Markdown("Visit https://example.com for more", 0, 0, default_markdown_theme())
        joined_plain = " ".join(strip_ansi(line) for line in markdown.render(80))
        assert joined_plain.count("https://example.com") == 1

    def test_shows_url_in_parentheses_when_hyperlinks_not_supported(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        markdown = Markdown("[click here](https://example.com)", 0, 0, default_markdown_theme())
        joined_plain = " ".join(strip_ansi(line) for line in markdown.render(80))
        assert "click here" in joined_plain
        assert "(https://example.com)" in joined_plain

    def test_shows_mailto_url_in_parentheses_when_hyperlinks_not_supported(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, default_markdown_theme())
        joined_plain = " ".join(strip_ansi(line) for line in markdown.render(80))
        assert "Email me" in joined_plain
        assert "(mailto:test@example.com)" in joined_plain

    def test_emits_osc8_hyperlink_sequence_when_terminal_supports_hyperlinks(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=True))
        markdown = Markdown("[click here](https://example.com)", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        joined = "".join(lines)
        assert "\x1b]8;;https://example.com\x1b\\" in joined
        assert "\x1b]8;;\x1b\\" in joined
        visible = "".join(re.sub(r"\x1b[^a-zA-Z]*[a-zA-Z]|\x1b\].*?\x1b\\", "", line) for line in lines)
        assert "click here" in visible
        raw_plain = "".join(
            re.sub(r"\x1b\[[0-9;]*m", "", re.sub(r"\x1b\]8;;[^\x1b]*\x1b\\", "", line)) for line in lines
        )
        assert "(https://example.com)" not in raw_plain

    def test_uses_osc8_for_mailto_links_when_terminal_supports_hyperlinks(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=True))
        markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, default_markdown_theme())
        joined = "".join(markdown.render(80))
        assert "\x1b]8;;mailto:test@example.com\x1b\\" in joined
        assert "\x1b]8;;\x1b\\" in joined

    def test_uses_osc8_for_bare_urls_when_terminal_supports_hyperlinks(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=True))
        markdown = Markdown("Visit https://example.com for more", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        joined = "".join(lines)
        assert "\x1b]8;;https://example.com\x1b\\" in joined
        raw_plain = "".join(
            re.sub(r"\x1b\[[0-9;]*m", "", re.sub(r"\x1b\]8;;[^\x1b]*\x1b\\", "", line)) for line in lines
        )
        assert "(https://example.com)" not in raw_plain


class TestHtmlLikeTagsInText:
    def test_renders_content_with_html_like_tags_as_text(self):
        markdown = Markdown(
            "This is text with <thinking>hidden content</thinking> that should be visible",
            0,
            0,
            default_markdown_theme(),
        )
        joined_plain = " ".join(strip_ansi(line) for line in markdown.render(80))
        assert "hidden content" in joined_plain or "<thinking>" in joined_plain

    def test_renders_html_tags_in_code_blocks_correctly(self):
        markdown = Markdown("```html\n<div>Some HTML</div>\n```", 0, 0, default_markdown_theme())
        joined_plain = "\n".join(strip_ansi(line) for line in markdown.render(80))
        assert "<div>" in joined_plain
        assert "</div>" in joined_plain


class TestStreamingCodeFences:
    def test_stabilizes_partial_closing_fence_rendering(self):
        cases = [
            ("```ts\nconst x = 1;\n``", ["```ts", "  const x = 1;", "```"]),
            (
                "```md\nnot a closing fence:\n``\n```",
                ["```md", "  not a closing fence:", "  ``", "```"],
            ),
            ("```ts\n``", ["```ts", "", "```"]),
            ("````\n```", ["```", "", "```"]),
            ("~~~~~\n~~~~", ["```", "", "```"]),
            (
                "```md\nnot a closing fence:\n``\n```\n\nafter",
                ["```md", "  not a closing fence:", "  ``", "```", "", "after"],
            ),
        ]

        for text, expected in cases:
            markdown = Markdown(text, 0, 0, default_markdown_theme())
            lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
            assert lines == expected, f"input={text!r}"

        partial = Markdown("```ts\nconst x = 1;\n``", 0, 0, default_markdown_theme())
        complete = Markdown("```ts\nconst x = 1;\n```", 0, 0, default_markdown_theme())
        assert len(partial.render(80)) == len(complete.render(80))


class TestParserCoverage:
    """Additional coverage not exercised by the upstream test suite.

    Covers nested-list depth beyond what markdown.test.ts checks, code fences
    with language info strings, links nested inside emphasis, tables that
    wrap, and indented (4-space) code blocks -- none of which the ported
    upstream cases directly assert against the token structure itself.
    """

    def test_three_level_nested_mixed_list_tokenizes_correctly(self):
        from pi_tui.components.markdown import ListToken, tokenize_blocks

        tokens = tokenize_blocks("- a\n  1. b\n     - c\n  2. d\n- e")
        assert len(tokens) == 1
        top = tokens[0]
        assert isinstance(top, ListToken)
        assert not top.ordered
        assert len(top.items) == 2

        nested_ordered = top.items[0].tokens[-1]
        assert isinstance(nested_ordered, ListToken)
        assert nested_ordered.ordered
        assert len(nested_ordered.items) == 2

        nested_bullet = nested_ordered.items[0].tokens[-1]
        assert isinstance(nested_bullet, ListToken)
        assert not nested_bullet.ordered
        assert len(nested_bullet.items) == 1

    def test_three_level_nested_list_renders_with_correct_indentation(self):
        markdown = Markdown("- a\n  1. b\n     - c\n  2. d\n- e", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert "- a" in lines
        assert "    1. b" in lines
        assert "        - c" in lines
        assert "    2. d" in lines
        assert "- e" in lines

    def test_fenced_code_block_preserves_language_info_string(self):
        from pi_tui.components.markdown import CodeToken, tokenize_blocks

        tokens = tokenize_blocks("```python\nprint('hi')\n```")
        assert len(tokens) == 1
        assert isinstance(tokens[0], CodeToken)
        assert tokens[0].lang == "python"
        assert tokens[0].text == "print('hi')"

    def test_fenced_code_block_with_tilde_fence_and_lang(self):
        from pi_tui.components.markdown import CodeToken, tokenize_blocks

        tokens = tokenize_blocks("~~~ruby\nputs 'hi'\n~~~")
        assert len(tokens) == 1
        assert isinstance(tokens[0], CodeToken)
        assert tokens[0].lang == "ruby"
        assert tokens[0].text == "puts 'hi'"

    def test_code_fence_renders_language_tag_in_output(self):
        markdown = Markdown("```python\nx = 1\n```", 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert lines[0] == "```python"

    def test_link_nested_inside_emphasis_tokenizes_correctly(self):
        from pi_tui.components.markdown import EmToken, LinkToken, tokenize_inline

        tokens = tokenize_inline("*[text](http://example.com)*")
        assert len(tokens) == 1
        em = tokens[0]
        assert isinstance(em, EmToken)
        assert len(em.tokens) == 1
        link = em.tokens[0]
        assert isinstance(link, LinkToken)
        assert link.href == "http://example.com"
        assert link.text == "text"

    def test_link_nested_inside_strong_renders_bold_and_link_styling(self):
        markdown = Markdown("**[bold link](http://example.com)**", 0, 0, default_markdown_theme())
        lines = markdown.render(80)
        joined_output = "\n".join(lines)
        joined_plain = " ".join(strip_ansi(line) for line in lines)
        assert "bold link" in joined_plain
        assert "\x1b[1m" in joined_output  # bold
        assert "\x1b[34m" in joined_output  # link (blue)

    def test_bold_nested_inside_link_text_tokenizes_correctly(self):
        from pi_tui.components.markdown import LinkToken, StrongToken, tokenize_inline

        tokens = tokenize_inline("[**bold** link text](http://example.com)")
        assert len(tokens) == 1
        link = tokens[0]
        assert isinstance(link, LinkToken)
        assert isinstance(link.tokens[0], StrongToken)

    def test_table_cell_wraps_long_content_across_multiple_lines(self):
        markdown = Markdown(
            "| A |\n| --- |\n| one two three four five six seven eight nine ten |",
            0,
            0,
            default_markdown_theme(),
        )
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(20)]
        data_rows = [line for line in plain_lines if line.startswith("│") and "─" not in line]
        assert len(data_rows) > 1
        for line in plain_lines:
            assert len(line) <= 20

    def test_indented_code_block_tokenizes_and_dedents(self):
        from pi_tui.components.markdown import CodeToken, tokenize_blocks

        tokens = tokenize_blocks("    line one\n    line two")
        assert len(tokens) == 1
        assert isinstance(tokens[0], CodeToken)
        assert tokens[0].text == "line one\nline two"
        assert tokens[0].lang is None

    def test_indented_code_block_renders_with_code_block_border_and_indent(self):
        markdown = Markdown("Paragraph first.\n\n    indented code here\n", 0, 0, default_markdown_theme())
        plain_lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert "Paragraph first." in plain_lines
        assert "```" in plain_lines
        assert any("indented code here" in line for line in plain_lines)

    def test_escaped_brackets_in_link_text(self):
        from pi_tui.components.markdown import LinkToken, tokenize_inline

        tokens = tokenize_inline(r"[a \[b\] c](http://example.com)")
        assert len(tokens) == 1
        link = tokens[0]
        assert isinstance(link, LinkToken)
        assert link.href == "http://example.com"

    def test_trailing_punctuation_trimmed_from_bare_url_autolink(self):
        from pi_tui.components.markdown import LinkToken, TextToken, tokenize_inline

        tokens = tokenize_inline("See https://example.com/page.")
        assert isinstance(tokens[-1], TextToken)
        assert tokens[-1].text == "."
        link = next(t for t in tokens if isinstance(t, LinkToken))
        assert link.href == "https://example.com/page"

    def test_horizontal_rule_variants_all_tokenize(self):
        from pi_tui.components.markdown import HrToken, tokenize_blocks

        for text in ["---", "***", "___", "- - -", "* * *"]:
            tokens = tokenize_blocks(text)
            assert any(isinstance(t, HrToken) for t in tokens), f"failed for {text!r}"

    def test_wrapping_text_in_paragraph_respects_width(self):
        from pi_tui.utils import visible_width

        markdown = Markdown(
            "This is a fairly long sentence that should wrap across several lines given a narrow width.",
            0,
            0,
            default_markdown_theme(),
        )
        lines = [strip_ansi(line) for line in markdown.render(20)]
        assert len(lines) > 1
        for line in lines:
            assert visible_width(line) <= 20

    def test_codespan_with_multiple_backticks_containing_single_backtick(self):
        from pi_tui.components.markdown import CodespanToken, tokenize_inline

        tokens = tokenize_inline("``code with ` backtick``")
        assert len(tokens) == 1
        assert isinstance(tokens[0], CodespanToken)
        assert tokens[0].text == "code with ` backtick"


class TestLatexMath:
    def test_renders_inline_dollar_and_parenthesis_delimiters(self):
        markdown = Markdown(
            r"A map $\mathbb{C}^3 \to \mathbb{C}^3$, $xy$, $x-y$, $-x$, $\frac{1}{2}$, and \(s \to \infty\).",
            0,
            0,
            default_markdown_theme(),
        )

        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["A map ℂ³ → ℂ³, xy, x-y, -x, 1/2, and s → ∞."]

    def test_renders_display_dollar_delimiters_without_markdown_escape_corruption(self):
        markdown = Markdown(
            "Before\n\n$$\\{3x+2y,\\; x \\in \\{0, \\pm 1\\}\\}$$\n\nafter",
            0,
            0,
            default_markdown_theme(),
        )

        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["Before", "", "{3x+2y, x ∈ {0, ± 1}}", "", "after"]

    def test_renders_display_bracket_delimiters(self):
        markdown = Markdown(
            "Before\n\n\\[\nE \\approx \\frac{0.1\\ \\text{lux}}{100\\ \\text{lm/W}}\n\\]\n\nafter",
            0,
            0,
            default_markdown_theme(),
        )

        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["Before", "", "    0.1 lux", "E ≈ ────────", "    100 lm/W", "", "after"]

    def test_aligns_matrix_rows_with_the_opening_delimiter(self):
        markdown = Markdown(
            "Consider the matrix\n\n\\[\nA=\n\\begin{pmatrix}\n\\pi & 0\\\\\n0 & \\frac{1}{\\pi}\n\\end{pmatrix}.\n\\]",
            0,
            0,
            default_markdown_theme(),
        )

        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["Consider the matrix", "", "A = ⎛ π │ 0   ⎞", "    ⎝ 0 │ 1/π ⎠."]

    def test_renders_lower_limits_beneath_display_operators(self):
        markdown = Markdown(
            "\\[\n\\lim_{x\\to 0}\\frac{\\frac{\\sin x}{x}-1}{\\frac{e^x-1}{x}-1}=0\n\\]",
            0,
            0,
            default_markdown_theme(),
        )

        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["     (sin x)/x-1", "lim  ─────────── = 0", "x→0  (eˣ-1)/x-1"]

    def test_renders_math_inside_lists_and_tables(self):
        markdown = Markdown(
            "- Formula: $F_1 = u^2$\n\n| Value |\n| --- |\n| $\\mathbb{C}^3$ |",
            0,
            0,
            default_markdown_theme(),
        )

        output = "\n".join(strip_ansi(line).rstrip() for line in markdown.render(80))

        assert "- Formula: F₁ = u²" in output
        assert "│ ℂ³" in output

    def test_does_not_treat_currency_shell_variables_or_code_spans_as_math(self):
        source = "Costs $5 and $10 or $8k–$12k; use `$x$`, $HOME, and $" + "{PATH}."
        markdown = Markdown(source, 0, 0, default_markdown_theme())

        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["Costs $5 and $10 or $8k–$12k; use $x$, $HOME, and $" + "{PATH}."]

        shell_variables = "Paths: $HOME/$USER and $XDG_CONFIG_HOME/$APP_CONFIG"
        shell_lines = [
            strip_ansi(line).rstrip() for line in Markdown(shell_variables, 0, 0, default_markdown_theme()).render(80)
        ]
        assert shell_lines == [shell_variables]

    def test_preserves_unsupported_and_incomplete_latex_exactly(self):
        for source in (r"Unknown $x + \unknown{y}$ after", r"Streaming $\mathbb{C}^3"):
            markdown = Markdown(source, 0, 0, default_markdown_theme())
            lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
            assert lines == [source]

    def test_preserves_incomplete_backslash_delimiters_while_streaming(self):
        inline = Markdown(r"Map \(\mathbb{C}^3", 0, 0, default_markdown_theme())
        assert [strip_ansi(line).rstrip() for line in inline.render(80)] == [r"Map \(\mathbb{C}^3"]

        display = Markdown("\\[\nx^2", 0, 0, default_markdown_theme())
        assert [strip_ansi(line).rstrip() for line in display.render(80)] == ["\\[", "x^2"]

    def test_does_not_render_latex_inside_escaped_delimiters_or_code_fences(self):
        source = "\n".join([r"Escaped \$x-y\$.", "", "```text", r"$\mathbb{C}^3$", "```"])
        markdown = Markdown(source, 0, 0, default_markdown_theme())
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == ["Escaped $x-y$.", "", "```text", "  $\\mathbb{C}^3$", "```"]

    def test_allows_latex_rendering_to_be_disabled(self):
        markdown = Markdown(
            r"Map $\mathbb{C}^3 \to \mathbb{C}^3$",
            0,
            0,
            default_markdown_theme(),
            None,
            MarkdownOptions(render_latex=False),
        )

        assert [strip_ansi(line).rstrip() for line in markdown.render(80)] == [r"Map $\mathbb{C}^3 \to \mathbb{C}^3$"]

    def test_switches_from_raw_to_rendered_math_when_a_streamed_delimiter_closes(self):
        markdown = Markdown(r"Map $\mathbb{C}^3", 0, 0, default_markdown_theme())
        assert [strip_ansi(line).rstrip() for line in markdown.render(80)] == [r"Map $\mathbb{C}^3"]

        markdown.set_text(r"Map $\mathbb{C}^3$")

        assert [strip_ansi(line).rstrip() for line in markdown.render(80)] == ["Map ℂ³"]
