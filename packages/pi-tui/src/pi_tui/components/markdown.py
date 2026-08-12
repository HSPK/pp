"""Markdown rendering component, ported from `packages/tui/src/components/markdown.ts`.

`markdown.ts` parses Markdown with the `marked` npm package plus a hand-rolled
LaTeX extension that renders `$...$`/`\\(...\\)`/`$$...$$` math as Unicode text.
This port does **not** add a Markdown-parsing dependency, but it does port the
LaTeX extension: `tokenize_inline_latex`/`tokenize_block_latex` below mirror
`tokenizeInlineLatex`/`tokenizeBlockLatex`, and rendering goes through
`pi_tui.latex.render_latex` exactly as in TypeScript.

Instead, `_tokenize_blocks`/`_tokenize_inline` below implement a small,
hand-written block/inline Markdown tokenizer covering exactly the constructs
the renderer needs. It is intentionally *not* a full CommonMark
implementation: notably, it does not support setext headings (`Title\\n===`),
HTML blocks/inline HTML tokens (HTML-like text such as `<thinking>` is simply
left as plain paragraph text, matching what the TS renderer's tests actually
assert), link reference definitions (`[text][ref]`), or angle-bracket
autolinks (`<https://...>`) -- only bare URL/email autolinking and explicit
`[text](url)` links are implemented, since that is what `markdown.ts`'s tests
exercise. Indented (4-space) code blocks are supported on a best-effort basis;
`markdown.test.ts` does not exercise them directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pi_tui.component import Component
from pi_tui.latex import RenderLatexOptions, render_latex
from pi_tui.terminal_image import get_capabilities, hyperlink, is_image_line
from pi_tui.utils import apply_background_to_line, visible_width, wrap_text_with_ansi

# ---------------------------------------------------------------------------
# Inline tokens
# ---------------------------------------------------------------------------


@dataclass
class TextToken:
    text: str
    type: Literal["text"] = "text"


@dataclass
class EscapeToken:
    """A backslash-escaped punctuation character, e.g. `\\*`."""

    text: str  # unescaped character
    raw: str  # original "\" + character
    type: Literal["escape"] = "escape"


@dataclass
class StrongToken:
    tokens: list[InlineToken]
    type: Literal["strong"] = "strong"


@dataclass
class EmToken:
    tokens: list[InlineToken]
    type: Literal["em"] = "em"


@dataclass
class DelToken:
    tokens: list[InlineToken]
    type: Literal["del"] = "del"


@dataclass
class CodespanToken:
    text: str
    type: Literal["codespan"] = "codespan"


@dataclass
class LinkToken:
    href: str
    text: str
    tokens: list[InlineToken]
    type: Literal["link"] = "link"


@dataclass
class LatexToken:
    """Inline math delimited by `$...$`, `$$...$$`, `\\(...\\)` or `\\[...\\]`."""

    text: str
    raw: str
    pending: bool = False
    type: Literal["latex"] = "latex"


InlineToken = TextToken | EscapeToken | StrongToken | EmToken | DelToken | CodespanToken | LinkToken | LatexToken


# ---------------------------------------------------------------------------
# Block tokens
# ---------------------------------------------------------------------------


@dataclass
class SpaceToken:
    """A run of one or more blank lines, rendered as a single blank line."""

    type: Literal["space"] = "space"


@dataclass
class HeadingToken:
    depth: int
    tokens: list[InlineToken]
    type: Literal["heading"] = "heading"


@dataclass
class ParagraphToken:
    tokens: list[InlineToken]
    type: Literal["paragraph"] = "paragraph"


@dataclass
class CodeToken:
    text: str
    raw: str
    lang: str | None = None
    type: Literal["code"] = "code"


@dataclass
class HrToken:
    type: Literal["hr"] = "hr"


@dataclass
class ListItemToken:
    raw: str
    tokens: list[BlockToken]
    task: bool = False
    checked: bool | None = None


@dataclass
class ListToken:
    ordered: bool
    start: int
    loose: bool
    items: list[ListItemToken]
    type: Literal["list"] = "list"


@dataclass
class BlockquoteToken:
    tokens: list[BlockToken]
    type: Literal["blockquote"] = "blockquote"


@dataclass
class TableCell:
    tokens: list[InlineToken]


@dataclass
class TableToken:
    header: list[TableCell]
    rows: list[list[TableCell]]
    raw: str
    type: Literal["table"] = "table"


@dataclass
class LatexBlockToken:
    """Display math delimited by `$$...$$` or `\\[...\\]` on its own block."""

    text: str
    raw: str
    pending: bool = False
    type: Literal["latexBlock"] = "latexBlock"


BlockToken = (
    SpaceToken
    | HeadingToken
    | ParagraphToken
    | CodeToken
    | HrToken
    | ListToken
    | BlockquoteToken
    | TableToken
    | LatexBlockToken
)


# ---------------------------------------------------------------------------
# Inline tokenizer
# ---------------------------------------------------------------------------

_ESCAPE_RE = re.compile(r"""^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])""")
_CODESPAN_RE = re.compile(r"^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)")
_STRONG_STAR_RE = re.compile(r"^\*\*(?=[^\s*])((?:\\.|[^\\])*?(?:\\.|[^\s*\\]))\*\*(?!\*)")
_STRONG_UNDERSCORE_RE = re.compile(r"^__(?=[^\s_])((?:\\.|[^\\])*?(?:\\.|[^\s_\\]))__(?!_)")
_EM_STAR_RE = re.compile(r"^\*(?=[^\s*])((?:\\.|[^\\])*?(?:\\.|[^\s*\\]))\*(?!\*)")
_EM_UNDERSCORE_RE = re.compile(r"^_(?=[^\s_])((?:\\.|[^\\])*?(?:\\.|[^\s_\\]))_(?!_)")
# Strict strikethrough: ported verbatim from markdown.ts's STRICT_STRIKETHROUGH_REGEX
# (a single tilde, e.g. "~text~", is intentionally left as plain text).
_DEL_RE = re.compile(r"^(~~)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)")
_IMAGE_RE = re.compile(r"""^!\[((?:\\.|[^\\\[\]])*)\]\(\s*<?([^\s>()]*)>?(?:\s+["'][^"']*["'])?\s*\)""")
_LINK_RE = re.compile(r"""^\[((?:\\.|[^\\\[\]])*)\]\(\s*<?([^\s>()]*)>?(?:\s+["'][^"']*["'])?\s*\)""")
_BARE_URL_RE = re.compile(r"^(https?://[^\s<]+)")
_BARE_EMAIL_RE = re.compile(
    r"^([A-Za-z0-9._%+\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+)"
)
_TRAILING_URL_PUNCT = ".,;:!?)]}'\""
_CODESPAN_WHITESPACE_RE = re.compile(r"[ \t]*\n[ \t]*")


def _trim_trailing_url_punct(url: str) -> str:
    while url and url[-1] in _TRAILING_URL_PUNCT:
        url = url[:-1]
    return url


# --- LaTeX math tokenizers (ported from markdown.ts) ------------------------

_PENDING_DOLLAR_MATH_RE = re.compile(r"\\[A-Za-z]+|[_^=+*/<>()\[\]|±≤≥≠≈∈→⇒∞∫∑√-]")
_DOLLAR_SPACE_RE = re.compile(r"^\$\s")
_ALL_CAPS_IDENTIFIER_RE = re.compile(r"^[A-Z_][A-Z0-9_]*(?:[^A-Za-z0-9_\s])?$")
_IDENTIFIER_START_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_LATEX_BLOCK_DOLLAR_RE = re.compile(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)")
_LATEX_BLOCK_BRACKET_RE = re.compile(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)")
_LATEX_BLOCK_PENDING_BRACKET_RE = re.compile(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*)$")
_LATEX_BLOCK_PENDING_DOLLAR_RE = re.compile(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*)$")
_LATEX_BLOCK_START_RE = re.compile(r"^ {0,3}(?:\$\$|\\\[)")


def _is_escaped(source: str, index: int) -> bool:
    backslashes = 0
    position = index - 1
    while position >= 0 and source[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _find_closing_delimiter(source: str, closing: str, start: int) -> int:
    index = source.find(closing, start)
    while index >= 0 and _is_escaped(source, index):
        index = source.find(closing, index + len(closing))
    return index


def _looks_like_pending_dollar_math(source: str) -> bool:
    return _PENDING_DOLLAR_MATH_RE.search(source) is not None


def tokenize_inline_latex(source: str) -> LatexToken | None:
    """Port of `tokenizeInlineLatex` in `markdown.ts`."""
    if source.startswith("$$"):
        opening, closing = "$$", "$$"
    elif source.startswith("\\("):
        opening, closing = "\\(", "\\)"
    elif source.startswith("\\["):
        opening, closing = "\\[", "\\]"
    elif source.startswith("$") and not _DOLLAR_SPACE_RE.match(source):
        opening, closing = "$", "$"
    else:
        return None

    closing_index = _find_closing_delimiter(source, closing, len(opening))
    if closing_index >= 0 and opening == "$":
        inner = source[len(opening) : closing_index]
        after = source[closing_index + 1 :]
        if (
            (inner and inner[-1].isspace())
            or (after[:1].isdigit())
            or (_ALL_CAPS_IDENTIFIER_RE.match(inner) is not None and _IDENTIFIER_START_RE.match(after) is not None)
            or "`" in inner
        ):
            return None

    if closing_index < 0:
        pending_source = source[len(opening) :]
        if opening.startswith("\\") or _looks_like_pending_dollar_math(pending_source):
            return LatexToken(text=pending_source, raw=source, pending=True)
        return None

    text = source[len(opening) : closing_index]
    if not text or "\n" in text:
        return None

    return LatexToken(text=text, raw=source[: closing_index + len(closing)])


def tokenize_block_latex(source: str) -> LatexBlockToken | None:
    """Port of `tokenizeBlockLatex` in `markdown.ts`."""
    dollar_match = _LATEX_BLOCK_DOLLAR_RE.match(source)
    if dollar_match and dollar_match.group(1):
        return LatexBlockToken(text=dollar_match.group(1).strip(), raw=dollar_match.group(0))

    bracket_match = _LATEX_BLOCK_BRACKET_RE.match(source)
    if bracket_match and bracket_match.group(1):
        return LatexBlockToken(text=bracket_match.group(1).strip(), raw=bracket_match.group(0))

    pending_bracket = _LATEX_BLOCK_PENDING_BRACKET_RE.match(source)
    if pending_bracket:
        return LatexBlockToken(text=pending_bracket.group(1), raw=pending_bracket.group(0), pending=True)

    pending_dollar = _LATEX_BLOCK_PENDING_DOLLAR_RE.match(source)
    if pending_dollar and pending_dollar.group(1) and _looks_like_pending_dollar_math(pending_dollar.group(1)):
        return LatexBlockToken(text=pending_dollar.group(1), raw=pending_dollar.group(0), pending=True)

    return None


def _underscore_run_is_delimiter(text: str, pos: int, matched: str) -> bool:
    """Whether an `_`/`__` run at `pos` may delimit emphasis (CommonMark).

    Underscores inside a word never open or close emphasis, so identifiers like
    `$XDG_CONFIG_HOME` survive intact. Asterisks have no such restriction.
    """
    before = text[pos - 1] if pos > 0 else ""
    after_index = pos + len(matched)
    after = text[after_index] if after_index < len(text) else ""
    return not (before.isalnum() or after.isalnum())


def tokenize_inline(text: str) -> list[InlineToken]:
    """Tokenize a run of inline Markdown text into `InlineToken`s."""
    tokens: list[InlineToken] = []
    buffer: list[str] = []
    pos = 0
    n = len(text)

    def flush_text() -> None:
        if buffer:
            tokens.append(TextToken(text="".join(buffer)))
            buffer.clear()

    while pos < n:
        remaining = text[pos:]

        # Backslash immediately before a newline is a hard line break; the
        # newline is already preserved verbatim by paragraph/list-item joins,
        # so we only need to drop the backslash itself.
        if remaining.startswith("\\\n"):
            buffer.append("\n")
            pos += 2
            continue

        if remaining[0] == "$" or remaining.startswith(("\\(", "\\[")):
            latex = tokenize_inline_latex(remaining)
            if latex is not None:
                flush_text()
                tokens.append(latex)
                pos += len(latex.raw)
                continue

        m = _ESCAPE_RE.match(remaining)
        if m:
            flush_text()
            tokens.append(EscapeToken(text=m.group(1), raw=m.group(0)))
            pos += len(m.group(0))
            continue

        m = _CODESPAN_RE.match(remaining)
        if m:
            flush_text()
            content = m.group(2).strip()
            content = _CODESPAN_WHITESPACE_RE.sub(" ", content)
            tokens.append(CodespanToken(text=content))
            pos += len(m.group(0))
            continue

        m = _DEL_RE.match(remaining)
        if m:
            flush_text()
            tokens.append(DelToken(tokens=tokenize_inline(m.group(2))))
            pos += len(m.group(0))
            continue

        m = _STRONG_STAR_RE.match(remaining)
        if m is None:
            underscore = _STRONG_UNDERSCORE_RE.match(remaining)
            if underscore is not None and _underscore_run_is_delimiter(text, pos, underscore.group(0)):
                m = underscore
        if m:
            flush_text()
            tokens.append(StrongToken(tokens=tokenize_inline(m.group(1))))
            pos += len(m.group(0))
            continue

        m = _EM_STAR_RE.match(remaining)
        if m is None:
            underscore = _EM_UNDERSCORE_RE.match(remaining)
            if underscore is not None and _underscore_run_is_delimiter(text, pos, underscore.group(0)):
                m = underscore
        if m:
            flush_text()
            tokens.append(EmToken(tokens=tokenize_inline(m.group(1))))
            pos += len(m.group(0))
            continue

        m = _IMAGE_RE.match(remaining)
        if m:
            # Images are out of scope (see components/image.py). `marked`
            # tokenizes `![alt](url)` as an image token, and markdown.ts's
            # inline renderer does not special-case images, so it falls
            # through to rendering the alt text as plain text -- match that.
            flush_text()
            tokens.append(TextToken(text=m.group(1)))
            pos += len(m.group(0))
            continue

        m = _LINK_RE.match(remaining)
        if m:
            flush_text()
            link_text, href = m.group(1), m.group(2)
            tokens.append(LinkToken(href=href, text=link_text, tokens=tokenize_inline(link_text)))
            pos += len(m.group(0))
            continue

        # Bare URL/email autolinking only triggers at a word boundary (start
        # of text or after a non-alphanumeric character), mirroring GFM.
        at_boundary = pos == 0 or not text[pos - 1].isalnum()
        if at_boundary:
            m = _BARE_URL_RE.match(remaining)
            if m:
                url = _trim_trailing_url_punct(m.group(1))
                flush_text()
                tokens.append(LinkToken(href=url, text=url, tokens=[TextToken(text=url)]))
                pos += len(url)
                continue

            m = _BARE_EMAIL_RE.match(remaining)
            if m:
                email = m.group(1)
                flush_text()
                tokens.append(LinkToken(href=f"mailto:{email}", text=email, tokens=[TextToken(text=email)]))
                pos += len(email)
                continue

        buffer.append(text[pos])
        pos += 1

    flush_text()
    return tokens


# ---------------------------------------------------------------------------
# Block tokenizer
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?$")
_HEADING_TRAILING_HASHES_RE = re.compile(r"(?:^|[ \t])#+[ \t]*$")
_FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")
_HR_RE = re.compile(r"^ {0,3}((?:-[ \t]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})$")
_BULLET_LIST_ITEM_RE = re.compile(r"^( {0,3})([-+*])([ \t]+|$)")
_ORDERED_LIST_ITEM_RE = re.compile(r"^( {0,3})(\d{1,9})([.)])([ \t]+|$)")
_BLOCKQUOTE_LINE_RE = re.compile(r"^ {0,3}>[ \t]?")
_TABLE_DELIM_ROW_RE = re.compile(r"^ {0,3}\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_TASK_MARKER_RE = re.compile(r"^\[([ xX])\][ \t]+")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _match_heading(line: str) -> HeadingToken | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    depth = len(m.group(1))
    text = m.group(2) or ""
    text = _HEADING_TRAILING_HASHES_RE.sub("", text).strip()
    return HeadingToken(depth=depth, tokens=tokenize_inline(text))


def _is_closing_fence(line: str, fence_char: str, min_len: int) -> bool:
    leading = len(line) - len(line.lstrip(" "))
    if leading > 3:
        return False
    stripped = line.strip()
    if len(stripped) < min_len:
        return False
    return all(ch == fence_char for ch in stripped)


def _try_fence(lines: list[str], i: int) -> tuple[CodeToken, int] | None:
    m = _FENCE_OPEN_RE.match(lines[i])
    if not m:
        return None
    _indent, fence, info = m.group(1), m.group(2), m.group(3)
    fence_char = fence[0]
    fence_len = len(fence)
    if fence_char == "`" and "`" in info:
        return None

    lang = info.strip() or None

    closing_index = None
    j = i + 1
    while j < len(lines):
        if _is_closing_fence(lines[j], fence_char, fence_len):
            closing_index = j
            break
        j += 1

    if closing_index is not None:
        content_lines = lines[i + 1 : closing_index]
        raw_lines = lines[i : closing_index + 1]
        end = closing_index + 1
    else:
        content_lines = lines[i + 1 :]
        raw_lines = lines[i:]
        end = len(lines)

    return CodeToken(text="\n".join(content_lines), raw="\n".join(raw_lines), lang=lang), end


@dataclass
class _ListItemMarkerInfo:
    indent: int
    ordered: bool
    start_num: int | None
    content_start: int


def _list_item_marker_info(line: str) -> _ListItemMarkerInfo | None:
    m = _ORDERED_LIST_ITEM_RE.match(line)
    if m:
        indent_str, num_str, _punct, ws = m.group(1), m.group(2), m.group(3), m.group(4)
        marker_len = len(indent_str) + len(num_str) + 1
        content_start = marker_len + len(ws) if ws else marker_len + 1
        return _ListItemMarkerInfo(
            indent=len(indent_str), ordered=True, start_num=int(num_str), content_start=min(content_start, len(line))
        )

    m = _BULLET_LIST_ITEM_RE.match(line)
    if m:
        indent_str, _bullet, ws = m.group(1), m.group(2), m.group(3)
        marker_len = len(indent_str) + 1
        content_start = marker_len + len(ws) if ws else marker_len + 1
        return _ListItemMarkerInfo(
            indent=len(indent_str), ordered=False, start_num=None, content_start=min(content_start, len(line))
        )

    return None


def _starts_new_block(line: str) -> bool:
    if line.strip() == "":
        return True
    if _match_heading(line) is not None:
        return True
    if _FENCE_OPEN_RE.match(line) is not None:
        return True
    if _HR_RE.match(line) is not None:
        return True
    if _BLOCKQUOTE_LINE_RE.match(line) is not None:
        return True
    if _LATEX_BLOCK_START_RE.match(line) is not None:
        return True
    return _list_item_marker_info(line) is not None


def _split_table_row(line: str) -> list[str]:
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|") and not trimmed.endswith("\\|"):
        trimmed = trimmed[:-1]
    cells = re.split(r"(?<!\\)\|", trimmed)
    return [c.strip() for c in cells]


def _try_table(lines: list[str], i: int) -> tuple[TableToken, int] | None:
    if "|" not in lines[i]:
        return None
    if i + 1 >= len(lines) or not _TABLE_DELIM_ROW_RE.match(lines[i + 1]):
        return None

    header_cells = _split_table_row(lines[i])
    if not header_cells:
        return None

    j = i + 2
    row_texts: list[list[str]] = []
    while j < len(lines) and lines[j].strip() != "" and "|" in lines[j] and not _BLOCKQUOTE_LINE_RE.match(lines[j]):
        row_texts.append(_split_table_row(lines[j]))
        j += 1

    header = [TableCell(tokens=tokenize_inline(c)) for c in header_cells]
    rows = [[TableCell(tokens=tokenize_inline(c)) for c in row] for row in row_texts]
    return TableToken(header=header, rows=rows, raw="\n".join(lines[i:j])), j


def _is_indented_code_line(line: str) -> bool:
    return line.startswith("    ") or line.startswith("\t")


def _parse_indented_code(lines: list[str], i: int) -> tuple[CodeToken, int]:
    j = i
    content_lines: list[str] = []
    while j < len(lines) and (lines[j].strip() == "" or _is_indented_code_line(lines[j])):
        content_lines.append(lines[j])
        j += 1
    while content_lines and content_lines[-1].strip() == "":
        content_lines.pop()
        j -= 1

    dedented = [re.sub(r"^(?: {4}|\t)", "", line) if line.strip() else "" for line in content_lines]
    return CodeToken(text="\n".join(dedented), raw="\n".join(lines[i:j]), lang=None), j


def _extract_task_marker(content: str) -> tuple[bool, bool | None, str]:
    first_line, _, rest = content.partition("\n")
    m = _TASK_MARKER_RE.match(first_line)
    if not m:
        return False, None, content
    checked = m.group(1).lower() == "x"
    new_first_line = first_line[m.end() :]
    new_content = new_first_line if not rest else f"{new_first_line}\n{rest}"
    return True, checked, new_content


def _parse_blockquote(lines: list[str], i: int) -> tuple[BlockquoteToken, int]:
    quote_lines: list[str] = []
    j = i
    while j < len(lines):
        line = lines[j]
        m = _BLOCKQUOTE_LINE_RE.match(line)
        if m:
            quote_lines.append(line[m.end() :])
            j += 1
            continue
        if line.strip() != "" and quote_lines and not _starts_new_block(line):
            quote_lines.append(line)
            j += 1
            continue
        break

    content = "\n".join(quote_lines)
    return BlockquoteToken(tokens=_parse_block_lines(content.split("\n"))), j


def _parse_list(lines: list[str], i: int) -> tuple[ListToken, int]:
    first_info = _list_item_marker_info(lines[i])
    assert first_info is not None
    first_indent = first_info.indent
    ordered = first_info.ordered
    start_num = first_info.start_num if first_info.start_num is not None else 1

    items: list[ListItemToken] = []
    loose = False
    j = i

    while j < len(lines):
        info = _list_item_marker_info(lines[j])
        if info is None or info.indent != first_indent or info.ordered != ordered:
            break

        marker_width = info.content_start
        item_lines = [lines[j][marker_width:]]
        item_raw_lines = [lines[j]]
        k = j + 1
        stop_list = False

        while k < len(lines):
            line = lines[k]

            if line.strip() == "":
                k2 = k
                while k2 < len(lines) and lines[k2].strip() == "":
                    k2 += 1
                if k2 >= len(lines):
                    break
                next_line = lines[k2]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent >= marker_width:
                    item_lines.extend([""] * (k2 - k))
                    item_raw_lines.extend(lines[k:k2])
                    k = k2
                    continue
                next_info = _list_item_marker_info(next_line)
                if next_info is not None and next_info.indent == first_indent and next_info.ordered == ordered:
                    loose = True
                else:
                    stop_list = True
                break

            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent >= marker_width:
                item_lines.append(line[marker_width:])
                item_raw_lines.append(line)
                k += 1
                continue

            if item_lines and item_lines[-1] != "" and not _starts_new_block(line):
                item_lines.append(line)
                item_raw_lines.append(line)
                k += 1
                continue

            break

        item_content = "\n".join(item_lines)
        raw = "\n".join(item_raw_lines)
        task, checked, item_content = _extract_task_marker(item_content)
        item_tokens = _parse_block_lines(item_content.split("\n"))
        items.append(ListItemToken(raw=raw, tokens=item_tokens, task=task, checked=checked))
        j = k
        if stop_list:
            break

    return ListToken(ordered=ordered, start=start_num, loose=loose, items=items), j


def _parse_paragraph(lines: list[str], i: int) -> tuple[ParagraphToken, int]:
    para_lines = [lines[i]]
    j = i + 1
    while j < len(lines) and lines[j].strip() != "" and not _starts_new_block(lines[j]):
        para_lines.append(lines[j])
        j += 1
    return ParagraphToken(tokens=tokenize_inline("\n".join(para_lines))), j


def _parse_block_lines(lines: list[str]) -> list[BlockToken]:
    tokens: list[BlockToken] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if line.strip() == "":
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            tokens.append(SpaceToken())
            continue

        heading = _match_heading(line)
        if heading is not None:
            tokens.append(heading)
            i += 1
            continue

        fence = _try_fence(lines, i)
        if fence is not None:
            token, i = fence
            tokens.append(token)
            continue

        if _LATEX_BLOCK_START_RE.match(line):
            latex_block = tokenize_block_latex("\n".join(lines[i:]))
            if latex_block is not None:
                consumed = latex_block.raw.count("\n")
                if not latex_block.raw.endswith("\n"):
                    consumed += 1
                tokens.append(latex_block)
                i += max(1, consumed)
                continue

        if _HR_RE.match(line) is not None:
            tokens.append(HrToken())
            i += 1
            continue

        table = _try_table(lines, i)
        if table is not None:
            token, i = table
            tokens.append(token)
            continue

        if _BLOCKQUOTE_LINE_RE.match(line):
            token, i = _parse_blockquote(lines, i)
            tokens.append(token)
            continue

        if _list_item_marker_info(line) is not None:
            token, i = _parse_list(lines, i)
            tokens.append(token)
            continue

        if _is_indented_code_line(line) and (not tokens or isinstance(tokens[-1], SpaceToken)):
            token, i = _parse_indented_code(lines, i)
            tokens.append(token)
            continue

        token, i = _parse_paragraph(lines, i)
        tokens.append(token)

    return tokens


def tokenize_blocks(source: str) -> list[BlockToken]:
    """Tokenize a Markdown document into top-level `BlockToken`s."""
    return _parse_block_lines(_normalize_newlines(source).split("\n"))


_FENCE_MARKER_RE = re.compile(r"^(`{3,}|~{3,})")


def trim_partial_closing_fences(tokens: list[BlockToken]) -> None:
    """Trim a streamed, not-quite-closed closing fence from the last code block.

    Ported from `trimPartialClosingFences` in markdown.ts: when Markdown
    arrives token-by-token (e.g. from an LLM), a code fence's closing marker
    may momentarily appear as a shorter, invalid fence (e.g. "``" before the
    final backtick arrives). Trimming that partial attempt from the trailing
    edge of the *last* token in the document avoids visual flicker as the
    real closing fence completes. See
    https://github.com/earendil-works/pi/issues/5825.
    """
    if not tokens:
        return

    token = tokens[-1]
    if isinstance(token, ListToken):
        if token.items:
            trim_partial_closing_fences(token.items[-1].tokens)
        return
    if isinstance(token, BlockquoteToken):
        trim_partial_closing_fences(token.tokens)
        return
    if not isinstance(token, CodeToken):
        return

    marker_match = _FENCE_MARKER_RE.match(token.raw)
    if not marker_match:
        return
    marker = marker_match.group(1)

    raw_lines = token.raw.split("\n")
    last_line = raw_lines[-1] if raw_lines else ""
    if not last_line or len(last_line) >= len(marker) or last_line != marker[0] * len(last_line):
        return

    token.text = token.text[: -len(last_line)]
    token.text = re.sub(r"\n$", "", token.text)


# ---------------------------------------------------------------------------
# Theme / options
# ---------------------------------------------------------------------------


@dataclass
class DefaultTextStyle:
    """Default text styling for markdown content.

    Applied to all text unless overridden by markdown formatting.
    """

    color: Callable[[str], str] | None = None
    bg_color: Callable[[str], str] | None = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


@dataclass
class MarkdownTheme:
    """Theme functions for markdown elements.

    Each function takes text and returns styled text with ANSI codes.
    """

    heading: Callable[[str], str]
    link: Callable[[str], str]
    link_url: Callable[[str], str]
    code: Callable[[str], str]
    code_block: Callable[[str], str]
    code_block_border: Callable[[str], str]
    quote: Callable[[str], str]
    quote_border: Callable[[str], str]
    hr: Callable[[str], str]
    list_bullet: Callable[[str], str]
    bold: Callable[[str], str]
    italic: Callable[[str], str]
    strikethrough: Callable[[str], str]
    underline: Callable[[str], str]
    highlight_code: Callable[[str, str | None], list[str]] | None = None
    # Prefix applied to each rendered code block line (default: "  ").
    code_block_indent: str | None = None


@dataclass
class MarkdownOptions:
    # Preserve source list markers instead of normalizing them.
    preserve_ordered_list_markers: bool = False
    # Preserve source backslash escapes instead of normalizing escaped punctuation.
    preserve_backslash_escapes: bool = False
    # Transform source Markdown before parsing, with the exact width available for content.
    transform: Callable[[str, int], str] | None = None
    # Render supported LaTeX math expressions as Unicode text. Not implemented
    # by this port (see module docstring); kept for interface parity with the
    # TS options type.
    render_latex: bool = True


@dataclass
class _InlineStyleContext:
    apply_text: Callable[[str], str]
    style_prefix: str


_SENTINEL = "\x00"


def _get_style_prefix(style_fn: Callable[[str], str]) -> str:
    styled = style_fn(_SENTINEL)
    sentinel_index = styled.find(_SENTINEL)
    return styled[:sentinel_index] if sentinel_index >= 0 else ""


# ---------------------------------------------------------------------------
# Markdown component
# ---------------------------------------------------------------------------


class Markdown(Component):
    def __init__(
        self,
        text: str,
        padding_x: int,
        padding_y: int,
        theme: MarkdownTheme,
        default_text_style: DefaultTextStyle | None = None,
        options: MarkdownOptions | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._theme = theme
        self._default_text_style = default_text_style
        self._options = options if options is not None else MarkdownOptions()

        self._default_style_prefix: str | None = None

        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self.invalidate()

    def invalidate(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_text == self._text and self._cached_width == width:
            return self._cached_lines

        content_width = max(1, width - self._padding_x * 2)
        text = self._options.transform(self._text, content_width) if self._options.transform else self._text

        if not text or text.strip() == "":
            result: list[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result

        normalized_text = text.replace("\t", "   ")

        tokens = tokenize_blocks(normalized_text)
        trim_partial_closing_fences(tokens)

        rendered_lines: list[str] = []
        for i, token in enumerate(tokens):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            next_type = next_token.type if next_token is not None else None
            rendered_lines.extend(self._render_token(token, content_width, next_type))

        wrapped_lines: list[str] = []
        for line in rendered_lines:
            if is_image_line(line):
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrap_text_with_ansi(line, content_width))

        left_margin = " " * self._padding_x
        right_margin = " " * self._padding_x
        bg_fn = self._default_text_style.bg_color if self._default_text_style else None
        content_lines: list[str] = []

        for line in wrapped_lines:
            if is_image_line(line):
                content_lines.append(line)
                continue

            line_with_margins = left_margin + line + right_margin

            if bg_fn:
                content_lines.append(apply_background_to_line(line_with_margins, width, bg_fn))
            else:
                visible_len = visible_width(line_with_margins)
                padding_needed = max(0, width - visible_len)
                content_lines.append(line_with_margins + " " * padding_needed)

        empty_line = " " * width
        empty_lines: list[str] = []
        for _ in range(self._padding_y):
            line = apply_background_to_line(empty_line, width, bg_fn) if bg_fn else empty_line
            empty_lines.append(line)

        result = empty_lines + content_lines + empty_lines

        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = result

        return result if len(result) > 0 else [""]

    # -- Styling helpers -----------------------------------------------------

    def _apply_default_style(self, text: str) -> str:
        """Apply default text style to a string.

        This is the base styling applied to all text content. NOTE:
        background color is NOT applied here - it's applied at the padding
        stage to ensure it extends to the full line width.
        """
        if not self._default_text_style:
            return text

        styled = text
        style = self._default_text_style

        if style.color:
            styled = style.color(styled)
        if style.bold:
            styled = self._theme.bold(styled)
        if style.italic:
            styled = self._theme.italic(styled)
        if style.strikethrough:
            styled = self._theme.strikethrough(styled)
        if style.underline:
            styled = self._theme.underline(styled)

        return styled

    def _get_default_style_prefix(self) -> str:
        if not self._default_text_style:
            return ""

        if self._default_style_prefix is not None:
            return self._default_style_prefix

        style = self._default_text_style
        styled = _SENTINEL

        if style.color:
            styled = style.color(styled)
        if style.bold:
            styled = self._theme.bold(styled)
        if style.italic:
            styled = self._theme.italic(styled)
        if style.strikethrough:
            styled = self._theme.strikethrough(styled)
        if style.underline:
            styled = self._theme.underline(styled)

        sentinel_index = styled.find(_SENTINEL)
        self._default_style_prefix = styled[:sentinel_index] if sentinel_index >= 0 else ""
        return self._default_style_prefix

    def _get_default_inline_style_context(self) -> _InlineStyleContext:
        return _InlineStyleContext(
            apply_text=self._apply_default_style,
            style_prefix=self._get_default_style_prefix(),
        )

    # -- Block-level rendering ------------------------------------------------

    def _render_token(
        self,
        token: BlockToken,
        width: int,
        next_token_type: str | None = None,
        style_context: _InlineStyleContext | None = None,
    ) -> list[str]:
        lines: list[str] = []

        if isinstance(token, HeadingToken):
            heading_level = token.depth
            heading_prefix = f"{'#' * heading_level} "

            # Build a heading-specific style context so inline tokens (codespan, bold, etc.)
            # restore heading styling after their own ANSI resets instead of falling back to
            # the default text style.
            if heading_level == 1:
                heading_style_fn: Callable[[str], str] = lambda text: self._theme.heading(  # noqa: E731
                    self._theme.bold(self._theme.underline(text))
                )
            else:
                heading_style_fn = lambda text: self._theme.heading(self._theme.bold(text))  # noqa: E731

            heading_style_context = _InlineStyleContext(
                apply_text=heading_style_fn,
                style_prefix=_get_style_prefix(heading_style_fn),
            )

            heading_text = self._render_inline_tokens(token.tokens, heading_style_context)
            styled_heading = heading_style_fn(heading_prefix) + heading_text if heading_level >= 3 else heading_text
            lines.append(styled_heading)
            if next_token_type and next_token_type != "space":
                lines.append("")

        elif isinstance(token, ParagraphToken):
            lines.append(self._render_inline_tokens(token.tokens, style_context))
            if next_token_type and next_token_type not in ("list", "space"):
                lines.append("")

        elif isinstance(token, LatexBlockToken):
            rendered = None
            if not token.pending and self._options.render_latex is not False:
                rendered = render_latex(token.text, RenderLatexOptions(display=True))
            for latex_line in (rendered if rendered is not None else token.raw.strip()).split("\n"):
                lines.append(self._apply_default_style(latex_line))
            if next_token_type and next_token_type != "space":
                lines.append("")

        elif isinstance(token, CodeToken):
            indent = self._theme.code_block_indent if self._theme.code_block_indent is not None else "  "
            lines.append(self._theme.code_block_border(f"```{token.lang or ''}"))
            if self._theme.highlight_code:
                for hl_line in self._theme.highlight_code(token.text, token.lang):
                    lines.append(f"{indent}{hl_line}")
            else:
                for code_line in token.text.split("\n"):
                    lines.append(f"{indent}{self._theme.code_block(code_line)}")
            lines.append(self._theme.code_block_border("```"))
            if next_token_type and next_token_type != "space":
                lines.append("")

        elif isinstance(token, ListToken):
            lines.extend(self._render_list(token, 0, width, style_context))

        elif isinstance(token, TableToken):
            lines.extend(self._render_table(token, width, next_token_type, style_context))

        elif isinstance(token, BlockquoteToken):

            def quote_style(text: str) -> str:
                return self._theme.quote(self._theme.italic(text))

            quote_style_prefix = _get_style_prefix(quote_style)

            def apply_quote_style(line: str) -> str:
                if not quote_style_prefix:
                    return quote_style(line)
                line_with_reapplied_style = line.replace("\x1b[0m", f"\x1b[0m{quote_style_prefix}")
                return quote_style(line_with_reapplied_style)

            quote_content_width = max(1, width - 2)

            # Blockquotes contain block-level tokens (paragraph, list, code, etc.), so render
            # children with _render_token() instead of _render_inline_tokens(). Default
            # message style should not apply inside blockquotes.
            quote_inline_style_context = _InlineStyleContext(
                apply_text=lambda text: text, style_prefix=quote_style_prefix
            )
            quote_tokens = token.tokens
            rendered_quote_lines: list[str] = []
            for i, quote_token in enumerate(quote_tokens):
                next_quote_token = quote_tokens[i + 1] if i + 1 < len(quote_tokens) else None
                rendered_quote_lines.extend(
                    self._render_token(
                        quote_token,
                        quote_content_width,
                        next_quote_token.type if next_quote_token is not None else None,
                        quote_inline_style_context,
                    )
                )

            # Avoid rendering an extra empty quote line before the outer blockquote spacing.
            while rendered_quote_lines and rendered_quote_lines[-1] == "":
                rendered_quote_lines.pop()

            for quote_line in rendered_quote_lines:
                styled_line = apply_quote_style(quote_line)
                for wrapped_line in wrap_text_with_ansi(styled_line, quote_content_width):
                    lines.append(self._theme.quote_border("│ ") + wrapped_line)
            if next_token_type and next_token_type != "space":
                lines.append("")

        elif isinstance(token, HrToken):
            lines.append(self._theme.hr("─" * min(width, 80)))
            if next_token_type and next_token_type != "space":
                lines.append("")

        elif isinstance(token, SpaceToken):
            lines.append("")

        return lines

    def _render_inline_tokens(self, tokens: list[InlineToken], style_context: _InlineStyleContext | None = None) -> str:
        result = ""
        resolved_style_context = (
            style_context if style_context is not None else self._get_default_inline_style_context()
        )
        apply_text = resolved_style_context.apply_text
        style_prefix = resolved_style_context.style_prefix

        def apply_text_with_newlines(text: str) -> str:
            return "\n".join(apply_text(segment) for segment in text.split("\n"))

        for token in tokens:
            if isinstance(token, LatexToken):
                rendered = None
                if not token.pending and self._options.render_latex is not False:
                    rendered = render_latex(token.text)
                result += apply_text_with_newlines(rendered if rendered is not None else token.raw)
            elif isinstance(token, EscapeToken):
                result += apply_text_with_newlines(
                    token.raw if self._options.preserve_backslash_escapes else token.text
                )
            elif isinstance(token, TextToken):
                result += apply_text_with_newlines(token.text)
            elif isinstance(token, StrongToken):
                bold_content = self._render_inline_tokens(token.tokens, resolved_style_context)
                result += self._theme.bold(bold_content) + style_prefix
            elif isinstance(token, EmToken):
                italic_content = self._render_inline_tokens(token.tokens, resolved_style_context)
                result += self._theme.italic(italic_content) + style_prefix
            elif isinstance(token, CodespanToken):
                result += self._theme.code(token.text) + style_prefix
            elif isinstance(token, LinkToken):
                link_text = self._render_inline_tokens(token.tokens, resolved_style_context)
                styled_link = self._theme.link(self._theme.underline(link_text))
                if get_capabilities().hyperlinks:
                    # OSC 8: render as a clickable hyperlink. The URL is not printed inline,
                    # so we always show only the link text regardless of whether it matches href.
                    result += hyperlink(styled_link, token.href) + style_prefix
                else:
                    # Fallback: print URL in parentheses when text differs from href.
                    # For mailto: links strip the prefix (autolinked emails use text="foo@bar.com"
                    # but href="mailto:foo@bar.com").
                    href_for_comparison = token.href[7:] if token.href.startswith("mailto:") else token.href
                    if token.text == token.href or token.text == href_for_comparison:
                        result += styled_link + style_prefix
                    else:
                        result += styled_link + self._theme.link_url(f" ({token.href})") + style_prefix
            elif isinstance(token, DelToken):
                del_content = self._render_inline_tokens(token.tokens, resolved_style_context)
                result += self._theme.strikethrough(del_content) + style_prefix

        while style_prefix and result.endswith(style_prefix):
            result = result[: -len(style_prefix)]

        return result

    # -- Lists -----------------------------------------------------------------

    def _get_ordered_list_marker(self, item: ListItemToken) -> str | None:
        m = re.match(r"^(?: {0,3})(\d{1,9}[.)])[ \t]+", item.raw)
        return f"{m.group(1)} " if m else None

    def _get_unordered_list_marker(self, item: ListItemToken) -> str | None:
        m = re.match(r"^(?: {0,3})([-+*])(?:[ \t]+|$)", item.raw)
        return f"{m.group(1)} " if m else None

    def _render_list(
        self, token: ListToken, depth: int, width: int, style_context: _InlineStyleContext | None = None
    ) -> list[str]:
        lines: list[str] = []
        indent = "    " * depth
        start_number = token.start

        for i, item in enumerate(token.items):
            is_last_item = i == len(token.items) - 1
            if token.ordered:
                bullet = (
                    (self._get_ordered_list_marker(item) or f"{start_number + i}. ")
                    if self._options.preserve_ordered_list_markers
                    else f"{start_number + i}. "
                )
            else:
                bullet = (
                    (self._get_unordered_list_marker(item) or "- ")
                    if self._options.preserve_ordered_list_markers
                    else "- "
                )
            task_marker = f"[{'x' if item.checked else ' '}] " if item.task else ""
            marker = bullet + task_marker
            first_prefix = indent + self._theme.list_bullet(marker)
            continuation_prefix = indent + " " * visible_width(marker)
            item_width = max(1, width - visible_width(first_prefix))
            rendered_any_line = False

            for item_token in item.tokens:
                if isinstance(item_token, ListToken):
                    lines.extend(self._render_list(item_token, depth + 1, width, style_context))
                    rendered_any_line = True
                    continue

                item_lines = self._render_token(item_token, item_width, None, style_context)
                for line in item_lines:
                    for wrapped_line in wrap_text_with_ansi(line, item_width):
                        line_prefix = continuation_prefix if rendered_any_line else first_prefix
                        lines.append(line_prefix + wrapped_line)
                        rendered_any_line = True

            if not rendered_any_line:
                lines.append(first_prefix)

            if token.loose and not is_last_item:
                lines.append("")

        return lines

    # -- Tables -----------------------------------------------------------------

    def _get_longest_word_width(self, text: str, max_width: int | None = None) -> int:
        words = [w for w in re.split(r"\s+", text) if w]
        longest = max((visible_width(word) for word in words), default=0)
        return min(longest, max_width) if max_width is not None else longest

    def _wrap_cell_text(self, text: str, max_width: int) -> list[str]:
        return wrap_text_with_ansi(text, max(1, max_width))

    def _render_table(
        self,
        token: TableToken,
        available_width: int,
        next_token_type: str | None = None,
        style_context: _InlineStyleContext | None = None,
    ) -> list[str]:
        lines: list[str] = []
        num_cols = len(token.header)

        if num_cols == 0:
            return lines

        # Calculate border overhead: "│ " + (n-1) * " │ " + " │" = 3n + 1
        border_overhead = 3 * num_cols + 1
        available_for_cells = available_width - border_overhead
        if available_for_cells < num_cols:
            # Too narrow to render a stable table. Fall back to raw markdown.
            fallback_lines = wrap_text_with_ansi(token.raw, available_width) if token.raw else []
            if next_token_type and next_token_type != "space":
                fallback_lines.append("")
            return fallback_lines

        max_unbroken_word_width = 30

        natural_widths = [0] * num_cols
        min_word_widths = [1] * num_cols
        for i in range(num_cols):
            header_text = self._render_inline_tokens(token.header[i].tokens, style_context)
            natural_widths[i] = visible_width(header_text)
            min_word_widths[i] = max(1, self._get_longest_word_width(header_text, max_unbroken_word_width))
        for row in token.rows:
            for i, cell in enumerate(row):
                cell_text = self._render_inline_tokens(cell.tokens, style_context)
                natural_widths[i] = max(natural_widths[i], visible_width(cell_text))
                min_word_widths[i] = max(
                    min_word_widths[i], self._get_longest_word_width(cell_text, max_unbroken_word_width)
                )

        min_column_widths = list(min_word_widths)
        min_cells_width = sum(min_column_widths)

        if min_cells_width > available_for_cells:
            min_column_widths = [1] * num_cols
            remaining = available_for_cells - num_cols

            if remaining > 0:
                total_weight = sum(max(0, w - 1) for w in min_word_widths)
                growth = [
                    (int((max(0, w - 1) / total_weight) * remaining) if total_weight > 0 else 0)
                    for w in min_word_widths
                ]

                for i in range(num_cols):
                    min_column_widths[i] += growth[i]

                allocated = sum(growth)
                leftover = remaining - allocated
                i = 0
                while leftover > 0 and i < num_cols:
                    min_column_widths[i] += 1
                    leftover -= 1
                    i += 1

            min_cells_width = sum(min_column_widths)

        total_natural_width = sum(natural_widths) + border_overhead
        column_widths: list[int]

        if total_natural_width <= available_width:
            column_widths = [max(natural_widths[i], min_column_widths[i]) for i in range(num_cols)]
        else:
            total_grow_potential = sum(max(0, natural_widths[i] - min_column_widths[i]) for i in range(num_cols))
            extra_width = max(0, available_for_cells - min_cells_width)
            column_widths = []
            for i in range(num_cols):
                min_width_i = min_column_widths[i]
                natural_width = natural_widths[i]
                min_width_delta = max(0, natural_width - min_width_i)
                grow = int((min_width_delta / total_grow_potential) * extra_width) if total_grow_potential > 0 else 0
                column_widths.append(min_width_i + grow)

            allocated = sum(column_widths)
            remaining = available_for_cells - allocated
            while remaining > 0:
                grew = False
                for i in range(num_cols):
                    if remaining <= 0:
                        break
                    if column_widths[i] < natural_widths[i]:
                        column_widths[i] += 1
                        remaining -= 1
                        grew = True
                if not grew:
                    break

        top_border_cells = ["─" * w for w in column_widths]
        lines.append(f"┌─{'─┬─'.join(top_border_cells)}─┐")

        header_cell_lines = [
            self._wrap_cell_text(self._render_inline_tokens(cell.tokens, style_context), column_widths[i])
            for i, cell in enumerate(token.header)
        ]
        header_line_count = max((len(c) for c in header_cell_lines), default=0)

        for line_idx in range(header_line_count):
            row_parts = []
            for col_idx, cell_lines in enumerate(header_cell_lines):
                text = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                padded = text + " " * max(0, column_widths[col_idx] - visible_width(text))
                row_parts.append(self._theme.bold(padded))
            lines.append(f"│ {' │ '.join(row_parts)} │")

        separator_cells = ["─" * w for w in column_widths]
        separator_line = f"├─{'─┼─'.join(separator_cells)}─┤"
        lines.append(separator_line)

        for row_index, row in enumerate(token.rows):
            row_cell_lines = [
                self._wrap_cell_text(self._render_inline_tokens(cell.tokens, style_context), column_widths[i])
                for i, cell in enumerate(row)
            ]
            row_line_count = max((len(c) for c in row_cell_lines), default=0)

            for line_idx in range(row_line_count):
                row_parts = []
                for col_idx, cell_lines in enumerate(row_cell_lines):
                    text = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                    row_parts.append(text + " " * max(0, column_widths[col_idx] - visible_width(text)))
                lines.append(f"│ {' │ '.join(row_parts)} │")

            if row_index < len(token.rows) - 1:
                lines.append(separator_line)

        bottom_border_cells = ["─" * w for w in column_widths]
        lines.append(f"└─{'─┴─'.join(bottom_border_cells)}─┘")

        if next_token_type and next_token_type != "space":
            lines.append("")
        return lines
