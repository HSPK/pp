"""Shared truncation utilities for tool outputs.

Python port of `packages/coding-agent/src/core/tools/truncate.ts`.

Truncation is based on two independent limits - whichever is hit first wins:
- Line limit (default: 2000 lines)
- Byte limit (default: 50KB)

Never returns partial lines (except the tail-truncation edge case).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500  # Max chars per grep match line


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None
    """Which limit was hit: "lines", "bytes", or None if not truncated."""
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    """Whether the last line was partially truncated (tail truncation edge case)."""
    first_line_exceeds_limit: bool
    """Whether the first line exceeded the byte limit (for head truncation)."""
    max_lines: int
    max_bytes: int


def _byte_length(text: str) -> int:
    return len(text.encode("utf-8"))


def format_size(num_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f}MB"


def _split_lines_for_counting(content: str) -> list[str]:
    if len(content) == 0:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def truncate_head(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes).

    Suitable for file reads where you want to see the beginning.

    Never returns partial lines. If the first line exceeds the byte limit,
    returns empty content with ``first_line_exceeds_limit=True``.
    """
    total_bytes = _byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = _byte_length(lines[0])
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by = "lines"

    for i in range(min(len(lines), max_lines)):
        line = lines[i]
        line_bytes = _byte_length(line) + (1 if i > 0 else 0)  # +1 for newline

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        output_lines_arr.append(line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    final_output_bytes = _byte_length(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """Truncate a string to fit within a byte limit (from the end), handling multi-byte UTF-8 boundaries."""
    buf = text.encode("utf-8")
    if len(buf) <= max_bytes:
        return text

    start = len(buf) - max_bytes
    # Find a valid UTF-8 boundary (start of a character): continuation bytes have the form 0b10xxxxxx.
    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
        start += 1

    return buf[start:].decode("utf-8", errors="replace")


def truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes).

    Suitable for bash output where you want to see the end (errors, final results).

    May return a partial first line if the last line of the original content
    exceeds the byte limit.
    """
    total_bytes = _byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by = "lines"
    last_line_partial = False

    i = len(lines) - 1
    while i >= 0 and len(output_lines_arr) < max_lines:
        line = lines[i]
        line_bytes = _byte_length(line) + (1 if len(output_lines_arr) > 0 else 0)  # +1 for newline

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: if we haven't added ANY lines yet and this line exceeds maxBytes,
            # take the end of the line (partial)
            if len(output_lines_arr) == 0:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = _byte_length(truncated_line)
                last_line_partial = True
            break

        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes
        i -= 1

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    final_output_bytes = _byte_length(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max characters, adding a ``[truncated]`` suffix.

    Used for grep match lines. Returns ``(text, was_truncated)``.
    """
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True
