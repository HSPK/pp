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
