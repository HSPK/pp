"""Built-in coding-agent tools.

Python port of `packages/coding-agent/src/core/tools/index.ts`. Exports the
tool factory functions and convenience groupings (`create_coding_tools`,
`create_read_only_tools`, `create_all_tools`). The `*ToolDefinition`/
`ToolDef` split from the TypeScript (used for the interactive TUI's
render-aware tool definitions) has no equivalent here since this port's
`AgentTool` has no rendering fields; only the `create_*_tool` factories that
produce a runnable `AgentTool` are ported.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pi_agent.types import AgentTool

from pi_coding_agent.tools.bash import BashToolDetails, create_bash_tool
from pi_coding_agent.tools.edit import EditToolDetails, create_edit_tool
from pi_coding_agent.tools.file_mutation_queue import with_file_mutation_queue
from pi_coding_agent.tools.find import FindToolDetails, create_find_tool
from pi_coding_agent.tools.grep import GrepToolDetails, create_grep_tool
from pi_coding_agent.tools.ls import LsToolDetails, create_ls_tool
from pi_coding_agent.tools.read import ReadToolDetails, create_read_tool
from pi_coding_agent.tools.write import create_write_tool

ToolName = str
ALL_TOOL_NAMES = frozenset({"read", "bash", "edit", "write", "grep", "find", "ls"})

__all__ = [
    "ALL_TOOL_NAMES",
    "BashToolDetails",
    "EditToolDetails",
    "FindToolDetails",
    "GrepToolDetails",
    "LsToolDetails",
    "ReadToolDetails",
    "ToolName",
    "create_all_tools",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_only_tools",
    "create_read_tool",
    "create_tool",
    "create_write_tool",
    "with_file_mutation_queue",
]


def create_tool(
    tool_name: str,
    cwd: str,
    *,
    session_environment: Callable[[], dict[str, str] | Awaitable[dict[str, str]]] | None = None,
    bash_command_prefix: str | None = None,
    bash_shell_path: str | None = None,
) -> AgentTool:
    """Create a single built-in tool by name.

    `session_environment`, `bash_command_prefix` and `bash_shell_path` are
    forwarded to the bash tool only. The latter two are the `shellCommandPrefix`
    and `shellPath` settings, which TypeScript passes into
    `createAllToolDefinitions(cwd, {bash: {commandPrefix, shellPath}})`.
    """
    if tool_name == "read":
        return create_read_tool(cwd)
    if tool_name == "bash":
        return create_bash_tool(
            cwd,
            bash_command_prefix,
            session_environment=session_environment,
            shell_path=bash_shell_path,
        )
    if tool_name == "edit":
        return create_edit_tool(cwd)
    if tool_name == "write":
        return create_write_tool(cwd)
    if tool_name == "grep":
        return create_grep_tool(cwd)
    if tool_name == "find":
        return create_find_tool(cwd)
    if tool_name == "ls":
        return create_ls_tool(cwd)
    raise ValueError(f"Unknown tool name: {tool_name}")


def create_coding_tools(cwd: str) -> list[AgentTool]:
    """Create the tool set for full read/write coding: read, bash, edit, write."""
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_read_only_tools(cwd: str) -> list[AgentTool]:
    """Create the tool set for read-only exploration: read, grep, find, ls."""
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_all_tools(cwd: str) -> dict[str, AgentTool]:
    """Create every built-in tool, keyed by name."""
    return {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "grep": create_grep_tool(cwd),
        "find": create_find_tool(cwd),
        "ls": create_ls_tool(cwd),
    }
