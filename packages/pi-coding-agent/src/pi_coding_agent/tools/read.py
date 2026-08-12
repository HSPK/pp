"""Read file contents, including images.

Python port of `packages/coding-agent/src/core/tools/read.ts`.

Images go through `utils.image_process.process_image`, matching upstream: an
unsupported format (e.g. BMP) is converted to PNG and oversized images are
resized, with the resulting hints appended to the text note. When the image
cannot be converted or resized the tool returns text only, carrying the
`[Image omitted: ...]` message.

Upstream also appends a "current model does not support images" note taken
from the execution context's model. This port's `AgentTool.execute` has no
context argument (see `pi_agent.agent_loop`), so that note has no source here.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import ImageContent, TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.path_utils import resolve_read_path, resolve_to_cwd
from pi_coding_agent.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)
from pi_coding_agent.utils.image_process import process_image
from pi_coding_agent.utils.mime import detect_supported_image_mime_type_from_file
from pi_coding_agent.utils.paths import format_path_relative_to_cwd_or_absolute


@dataclass
class ReadToolDetails:
    truncation: TruncationResult | None = None


def create_read_tool(cwd: str, *, auto_resize_images: bool = True) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        path = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        absolute_path = resolve_read_path(path, cwd)

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(errno.ENOENT, f"ENOENT: no such file or directory, open '{absolute_path}'")
        if not os.access(absolute_path, os.R_OK):
            raise PermissionError(errno.EACCES, f"EACCES: permission denied, open '{absolute_path}'")

        mime_type = detect_supported_image_mime_type_from_file(absolute_path)
        content: list[TextContent | ImageContent]
        details: ReadToolDetails | None = None

        if mime_type:
            with open(absolute_path, "rb") as f:
                data = f.read()
            processed = process_image(data, mime_type, auto_resize_images=auto_resize_images)
            if not processed.ok:
                content = [TextContent(text=f"Read image file [{mime_type}]\n{processed.message}")]
            else:
                text_note = f"Read image file [{processed.mime_type}]"
                if processed.hints:
                    text_note += "\n" + "\n".join(processed.hints)
                content = [
                    TextContent(text=text_note),
                    ImageContent(data=processed.data, mime_type=processed.mime_type),
                ]
        else:
            with open(absolute_path, "rb") as f:
                buffer = f.read()
            text_content = buffer.decode("utf-8", errors="replace")
            all_lines = text_content.split("\n")
            total_file_lines = len(all_lines)

            start_line = max(0, offset - 1) if offset else 0
            start_line_display = start_line + 1
            if start_line >= len(all_lines):
                raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

            user_limited_lines: int | None = None
            if limit is not None:
                end_line = min(start_line + limit, len(all_lines))
                selected_content = "\n".join(all_lines[start_line:end_line])
                user_limited_lines = end_line - start_line
            else:
                selected_content = "\n".join(all_lines[start_line:])

            truncation = truncate_head(selected_content)

            if truncation.first_line_exceeds_limit:
                first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
                output_text = (
                    f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} "
                    f"limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
                )
                details = ReadToolDetails(truncation=truncation)
            elif truncation.truncated:
                end_line_display = start_line_display + truncation.output_lines - 1
                next_offset = end_line_display + 1
                output_text = truncation.content
                if truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                        f"Use offset={next_offset} to continue.]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                    )
                details = ReadToolDetails(truncation=truncation)
            elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
                remaining = len(all_lines) - (start_line + user_limited_lines)
                next_offset = start_line + user_limited_lines + 1
                output_text = (
                    f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                )
            else:
                output_text = truncation.content

            content = [TextContent(text=output_text)]

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        return AgentToolResult(content=content, details=details)

    return AgentTool(
        name="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            "Images are sent as attachments. For text files, output is truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "Use offset/limit for large files. When you need the full file, continue with offset until complete."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        execute=execute,
    )


# --------------------------------------------------------------------------
# Rendering
#
# Port of `read.ts`'s `formatReadCall` / `formatReadResult` and the compact
# classifications. These were previously unported, so the read tool fell back
# to the generic renderer: no `read <path>` title, no syntax-highlighted body,
# no truncation notices, and no collapsed form.
# --------------------------------------------------------------------------

COMPACT_RESOURCE_FILE_NAMES = frozenset({"AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"})

DEFAULT_MAX_BYTES_DISPLAY = 256 * 1024
DEFAULT_MAX_LINES_DISPLAY = 2000


def _render_args(args: Any) -> dict[str, Any]:
    return args if isinstance(args, dict) else {}


def _format_read_line_range(args: Any, theme: Any) -> str:
    a = _render_args(args)
    offset, limit = a.get("offset"), a.get("limit")
    if offset is None and limit is None:
        return ""
    start_line = offset if offset is not None else 1
    end_line = start_line + limit - 1 if limit is not None else None
    suffix = f"-{end_line}" if end_line else ""
    return theme.fg("warning", f":{start_line}{suffix}")


def _pi_docs_classification(absolute_path: str) -> tuple[str, str] | None:
    """`(kind, label)` when the path is inside this package's own docs."""
    from pi_coding_agent.core.config import get_readme_path

    package_root = os.path.dirname(get_readme_path())
    relative_path = os.path.relpath(os.path.realpath(absolute_path), os.path.realpath(package_root))
    if relative_path in ("", os.pardir) or relative_path.startswith(f"{os.pardir}{os.sep}"):
        return None
    if os.path.isabs(relative_path):
        return None
    label = relative_path.replace(os.sep, "/")
    if label == "README.md" or label.startswith("docs/") or label.startswith("examples/"):
        return ("docs", label)
    return None


def _compact_read_classification(args: Any, cwd: str) -> tuple[str, str] | None:
    from pi_coding_agent.tools.render_utils import str_arg

    a = _render_args(args)
    raw_path = str_arg(a.get("file_path") if a.get("file_path") is not None else a.get("path"))
    if not raw_path:
        return None

    absolute_path = resolve_to_cwd(raw_path, cwd)
    file_name = os.path.basename(absolute_path)
    if file_name == "SKILL.md":
        return ("skill", os.path.basename(os.path.dirname(absolute_path)) or file_name)

    docs = _pi_docs_classification(absolute_path)
    if docs is not None:
        return docs

    if file_name in COMPACT_RESOURCE_FILE_NAMES:
        return ("resource", format_path_relative_to_cwd_or_absolute(absolute_path, cwd))
    return None


def _format_compact_read_call(classification: tuple[str, str], args: Any, theme: Any) -> str:
    from pi_coding_agent.modes.interactive.components.keybinding_hints import key_text

    kind, label = classification
    expand_hint = theme.fg("dim", f" ({key_text('app.tools.expand')} to expand)")
    if kind == "skill":
        return (
            theme.fg("customMessageLabel", "\x1b[1m[skill]\x1b[22m ")
            + theme.fg("customMessageText", label)
            + _format_read_line_range(args, theme)
            + expand_hint
        )
    return (
        theme.fg("toolTitle", theme.bold(f"read {kind}"))
        + " "
        + theme.fg("accent", label)
        + _format_read_line_range(args, theme)
        + expand_hint
    )


def format_read_call(args: Any, theme: Any, cwd: str) -> str:
    """Port of `formatReadCall`."""
    from pi_coding_agent.tools.render_utils import render_tool_path, str_arg

    a = _render_args(args)
    raw = str_arg(a.get("file_path") if a.get("file_path") is not None else a.get("path"))
    path_display = render_tool_path(raw, theme, cwd)
    return f"{theme.fg('toolTitle', theme.bold('read'))} {path_display}{_format_read_line_range(args, theme)}"


def _trim_trailing_empty_lines(lines: list[str]) -> list[str]:
    end = len(lines)
    while end > 0 and lines[end - 1] == "":
        end -= 1
    return lines[:end]


def format_read_result(
    args: Any, result: Any, options: Any, theme: Any, show_images: bool, _cwd: str, is_error: bool
) -> str:
    """Port of `formatReadResult`.

    Collapsed and not an error renders as the empty string: upstream shows only
    the call line until the user expands, so returning body text here would
    make every read twice as tall as it is in TypeScript.
    """
    from pi_coding_agent.modes.interactive.components.keybinding_hints import key_hint
    from pi_coding_agent.modes.interactive.theme.theme import get_language_from_path, highlight_code
    from pi_coding_agent.tools.render_utils import get_text_output, replace_tabs, str_arg

    expanded = bool(getattr(options, "expanded", False))
    if not expanded and not is_error:
        return ""

    a = _render_args(args)
    raw_path = str_arg(a.get("file_path") if a.get("file_path") is not None else a.get("path"))
    output = get_text_output(result, show_images)
    lang = get_language_from_path(raw_path) if (not is_error and raw_path) else None
    rendered_lines = highlight_code(replace_tabs(output), lang) if lang else output.split("\n")
    lines = _trim_trailing_empty_lines(rendered_lines)
    max_lines = len(lines) if expanded else 10
    display_lines = lines[:max_lines]
    remaining = len(lines) - max_lines

    body = "\n".join(line if lang else theme.fg("toolOutput", replace_tabs(line)) for line in display_lines)
    text = f"\n{body}"
    if remaining > 0:
        text += (
            theme.fg("muted", f"\n... ({remaining} more lines,")
            + " "
            + key_hint("app.tools.expand", "to expand")
            + theme.fg("muted", ")")
        )

    truncation = getattr(getattr(result, "details", None), "truncation", None)
    if truncation is not None and getattr(truncation, "truncated", False):
        max_bytes = getattr(truncation, "max_bytes", None) or DEFAULT_MAX_BYTES_DISPLAY
        if getattr(truncation, "first_line_exceeds_limit", False):
            text += "\n" + theme.fg("warning", f"[First line exceeds {format_size(max_bytes)} limit]")
        elif getattr(truncation, "truncated_by", None) == "lines":
            max_l = getattr(truncation, "max_lines", None) or DEFAULT_MAX_LINES_DISPLAY
            text += "\n" + theme.fg(
                "warning",
                f"[Truncated: showing {truncation.output_lines} of {truncation.total_lines} lines ({max_l} line limit)]",
            )
        else:
            text += "\n" + theme.fg(
                "warning",
                f"[Truncated: {truncation.output_lines} lines shown ({format_size(max_bytes)} limit)]",
            )
    return text
