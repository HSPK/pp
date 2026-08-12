"""Write content to a file, creating parent directories as needed.

Python port of `packages/coding-agent/src/core/tools/write.ts`.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.file_mutation_queue import with_file_mutation_queue
from pi_coding_agent.tools.path_utils import resolve_to_cwd


@dataclass
class WriteOperations:
    """Pluggable filesystem operations for the write tool.

    Port of `WriteOperations` in `write.ts`: override these to delegate file
    writing to a remote system (for example SSH).
    """

    write_file: Callable[[str, str], Awaitable[None]]
    """Write content to a file."""
    mkdir: Callable[[str], Awaitable[None]]
    """Create a directory recursively."""


async def _default_write_file(absolute_path: str, content: str) -> None:
    with open(absolute_path, "w", encoding="utf-8") as fh:
        fh.write(content)


async def _default_mkdir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def default_write_operations() -> WriteOperations:
    return WriteOperations(write_file=_default_write_file, mkdir=_default_mkdir)


def create_write_tool(cwd: str, operations: WriteOperations | None = None) -> AgentTool:
    ops = operations if operations is not None else default_write_operations()

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        path = params["path"]
        content = params["content"]
        absolute_path = resolve_to_cwd(path, cwd)
        directory = os.path.dirname(absolute_path)

        async def mutate() -> AgentToolResult:
            def throw_if_aborted() -> None:
                if signal is not None and signal.aborted:
                    raise RuntimeError("Operation aborted")

            throw_if_aborted()
            if directory:
                await ops.mkdir(directory)
            throw_if_aborted()

            await ops.write_file(absolute_path, content)
            throw_if_aborted()

            return AgentToolResult(content=[TextContent(text=f"Successfully wrote {len(content)} bytes to {path}")])

        return await with_file_mutation_queue(absolute_path, mutate)

    return AgentTool(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        execute=execute,
    )


# -- rendering (port of `write.ts`'s formatWriteCall / formatWriteResult) ---

WRITE_COLLAPSED_MAX_LINES = 10


def format_write_call(args: Any, options: Any, theme: Any, cwd: str) -> str:
    """Port of `formatWriteCall`.

    The content preview lives on the *call*, not the result: what the model is
    about to write is the interesting part, and the result carries only an
    error if one occurred.
    """
    from pi_coding_agent.modes.interactive.components.keybinding_hints import key_hint
    from pi_coding_agent.modes.interactive.theme.theme import get_language_from_path, highlight_code
    from pi_coding_agent.tools.render_utils import (
        normalize_display_text,
        render_tool_path,
        replace_tabs,
        str_arg,
    )

    a = args if isinstance(args, dict) else {}
    raw_path = str_arg(a.get("file_path") if a.get("file_path") is not None else a.get("path"))
    file_content = str_arg(a.get("content"))
    text = f"{theme.fg('toolTitle', theme.bold('write'))} {render_tool_path(raw_path, theme, cwd)}"

    if file_content is None:
        return text + f"\n\n{theme.fg('error', '[invalid content arg - expected string]')}"
    if not file_content:
        return text

    lang = get_language_from_path(raw_path) if raw_path else None
    normalized = normalize_display_text(file_content)
    rendered_lines = highlight_code(replace_tabs(normalized), lang) if lang else normalized.split("\n")
    while rendered_lines and rendered_lines[-1] == "":
        rendered_lines.pop()
    total_lines = len(rendered_lines)
    max_lines = total_lines if getattr(options, "expanded", False) else WRITE_COLLAPSED_MAX_LINES
    display_lines = rendered_lines[:max_lines]
    remaining = total_lines - max_lines

    body = "\n".join(line if lang else theme.fg("toolOutput", replace_tabs(line)) for line in display_lines)
    text += f"\n\n{body}"
    if remaining > 0:
        text += (
            theme.fg("muted", f"\n... ({remaining} more lines, {total_lines} total,")
            + " "
            + key_hint("app.tools.expand", "to expand")
            + theme.fg("muted", ")")
        )
    return text


def format_write_result(result: Any, theme: Any, is_error: bool) -> str | None:
    """Port of `formatWriteResult`: nothing unless it failed."""
    if not is_error:
        return None
    output = "\n".join(
        getattr(c, "text", "") or "" for c in getattr(result, "content", []) if getattr(c, "type", None) == "text"
    )
    if not output:
        return None
    return f"\n{theme.fg('error', output)}"
