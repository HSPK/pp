"""List directory contents.

Python port of `packages/coding-agent/src/core/tools/ls.ts`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.path_utils import resolve_to_cwd
from pi_coding_agent.tools.truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head

_DEFAULT_LIMIT = 500


@dataclass
class LsToolDetails:
    truncation: TruncationResult | None = None
    entry_limit_reached: int | None = None


def create_ls_tool(cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        path = params.get("path")
        limit = params.get("limit")
        dir_path = resolve_to_cwd(path or ".", cwd)
        effective_limit = limit if limit is not None else _DEFAULT_LIMIT

        if not os.path.exists(dir_path):
            raise RuntimeError(f"Path not found: {dir_path}")
        if not os.path.isdir(dir_path):
            raise RuntimeError(f"Not a directory: {dir_path}")

        try:
            entries = os.listdir(dir_path)
        except OSError as e:
            raise RuntimeError(f"Cannot read directory: {e.strerror or e}") from e

        entries.sort(key=str.lower)

        results: list[str] = []
        entry_limit_reached = False
        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break
            full_path = os.path.join(dir_path, entry)
            try:
                suffix = "/" if os.path.isdir(full_path) else ""
            except OSError:
                continue
            results.append(entry + suffix)

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        if not results:
            return AgentToolResult(content=[TextContent(text="(empty directory)")])

        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=2**31)
        output = truncation.content
        details = LsToolDetails()
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
            details.entry_limit_reached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.truncation is not None or details.entry_limit_reached is not None
        return AgentToolResult(content=[TextContent(text=output)], details=details if has_details else None)

    return AgentTool(
        name="ls",
        description=(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
            f"Includes dotfiles. Output is truncated to {_DEFAULT_LIMIT} entries or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
            },
        },
        execute=execute,
    )
