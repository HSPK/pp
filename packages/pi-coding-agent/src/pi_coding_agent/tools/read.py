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

from pi_coding_agent.tools.path_utils import resolve_read_path
from pi_coding_agent.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)
from pi_coding_agent.utils.image_process import process_image
from pi_coding_agent.utils.mime import detect_supported_image_mime_type_from_file


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
