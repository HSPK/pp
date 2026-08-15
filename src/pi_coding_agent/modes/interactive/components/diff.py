"""Colored diff rendering with intra-line change highlighting.

Ported from ``packages/coding-agent/src/modes/interactive/components/diff.ts``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ....utils.word_diff import diff_words
from ..theme.theme import theme

_DIFF_LINE_RE = re.compile(r"^([+\-\s])(\s*\d*)\s(.*)$")
_LEADING_WS_RE = re.compile(r"^(\s*)")


@dataclass
class ParsedDiffLine:
    prefix: str
    line_num: str
    content: str


@dataclass
class RenderDiffOptions:
    """``file_path`` is unused; kept for API compatibility with TypeScript."""

    file_path: str | None = None


def parse_diff_line(line: str) -> ParsedDiffLine | None:
    """Parse ``"+123 content"`` / ``"-123 content"`` / ``" 123 content"``."""
    match = _DIFF_LINE_RE.match(line)
    if not match:
        return None
    return ParsedDiffLine(prefix=match.group(1), line_num=match.group(2), content=match.group(3))


def _replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def render_intra_line_diff(old_content: str, new_content: str) -> tuple[str, str]:
    """Word-diff two lines, inverting the changed parts.

    Leading whitespace is excluded from the inverse so indentation is not
    highlighted. Returns ``(removed_line, added_line)``.
    """
    word_diff = diff_words(old_content, new_content)

    removed_line = ""
    added_line = ""
    is_first_removed = True
    is_first_added = True

    for part in word_diff:
        if part.removed:
            value = part.value
            if is_first_removed:
                leading_ws = _LEADING_WS_RE.match(value).group(1)  # type: ignore[union-attr]
                value = value[len(leading_ws) :]
                removed_line += leading_ws
                is_first_removed = False
            if value:
                removed_line += theme.inverse(value)
        elif part.added:
            value = part.value
            if is_first_added:
                leading_ws = _LEADING_WS_RE.match(value).group(1)  # type: ignore[union-attr]
                value = value[len(leading_ws) :]
                added_line += leading_ws
                is_first_added = False
            if value:
                added_line += theme.inverse(value)
        else:
            removed_line += part.value
            added_line += part.value

    return removed_line, added_line


def render_diff(diff_text: str, _options: RenderDiffOptions | None = None) -> str:
    """Colorize a diff: context dim, removals red, additions green, with
    inverse highlighting on the changed tokens of a single-line modification."""
    lines = diff_text.split("\n")
    result: list[str] = []

    index = 0
    while index < len(lines):
        parsed = parse_diff_line(lines[index])

        if parsed is None:
            result.append(theme.fg("toolDiffContext", lines[index]))
            index += 1
            continue

        if parsed.prefix == "-":
            removed_lines: list[ParsedDiffLine] = []
            while index < len(lines):
                current = parse_diff_line(lines[index])
                if current is None or current.prefix != "-":
                    break
                removed_lines.append(current)
                index += 1

            added_lines: list[ParsedDiffLine] = []
            while index < len(lines):
                current = parse_diff_line(lines[index])
                if current is None or current.prefix != "+":
                    break
                added_lines.append(current)
                index += 1

            # Intra-line diffing only makes sense for a 1:1 replacement.
            if len(removed_lines) == 1 and len(added_lines) == 1:
                removed = removed_lines[0]
                added = added_lines[0]
                removed_line, added_line = render_intra_line_diff(
                    _replace_tabs(removed.content), _replace_tabs(added.content)
                )
                result.append(theme.fg("toolDiffRemoved", f"-{removed.line_num} {removed_line}"))
                result.append(theme.fg("toolDiffAdded", f"+{added.line_num} {added_line}"))
            else:
                for removed in removed_lines:
                    result.append(theme.fg("toolDiffRemoved", f"-{removed.line_num} {_replace_tabs(removed.content)}"))
                for added in added_lines:
                    result.append(theme.fg("toolDiffAdded", f"+{added.line_num} {_replace_tabs(added.content)}"))
        elif parsed.prefix == "+":
            result.append(theme.fg("toolDiffAdded", f"+{parsed.line_num} {_replace_tabs(parsed.content)}"))
            index += 1
        else:
            result.append(theme.fg("toolDiffContext", f" {parsed.line_num} {_replace_tabs(parsed.content)}"))
            index += 1

    return "\n".join(result)


__all__ = ["ParsedDiffLine", "RenderDiffOptions", "parse_diff_line", "render_diff", "render_intra_line_diff"]
