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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

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


@dataclass
class BuiltInToolDefinition:
    """The renderer half of a built-in tool.

    Port of the `ToolDefinition` objects `createAllToolDefinitions` returns
    (`core/tools/index.ts:156`). `ToolExecutionComponent` looks these up by
    tool name and calls `render_call`/`render_result`; without them every
    built-in tool falls back to the generic renderer, which is why read output
    had no title line, no highlighting and no collapsed form.

    Kept separate from `AgentTool` because `pi_agent.types.AgentTool` is the
    execution contract and carries no display concerns.
    """

    render_call: Callable[..., Any] | None = None
    render_result: Callable[..., Any] | None = None
    render_shell: str | None = None


def create_all_tool_definitions(cwd: str) -> dict[str, BuiltInToolDefinition]:
    """Renderer definitions for the built-in tools, keyed by name.

    Port of `createAllToolDefinitions`. Only tools whose renderers are ported
    appear here; a missing entry leaves that tool on the generic renderer,
    which is the same fallback upstream uses for tools without renderers.
    """
    from pi_tui.components.text import Text

    from pi_coding_agent.tools.read import format_read_call, format_read_result

    def _text_component(context: Any) -> Any:
        existing = getattr(context, "last_component", None)
        return existing if isinstance(existing, Text) else Text("", 0, 0)

    def read_render_call(args: Any, theme: Any, context: Any) -> Any:
        component = _text_component(context)
        component.set_text(format_read_call(args, theme, getattr(context, "cwd", cwd)))
        return component

    def read_render_result(result: Any, options: Any, theme: Any, context: Any) -> Any:
        component = _text_component(context)
        component.set_text(
            format_read_result(
                getattr(context, "args", None),
                result,
                options,
                theme,
                getattr(context, "show_images", False),
                getattr(context, "cwd", cwd),
                getattr(context, "is_error", False),
            )
        )
        return component

    from pi_coding_agent.tools.bash import format_bash_call, format_bash_result_lines

    def bash_render_call(args: Any, theme: Any, context: Any) -> Any:
        component = _text_component(context)
        component.set_text(format_bash_call(args, theme))
        return component

    def bash_render_result(result: Any, options: Any, theme: Any, context: Any) -> Any:
        component = _text_component(context)
        state = getattr(context, "state", {}) or {}
        component.set_text(
            "\n".join(
                format_bash_result_lines(
                    result,
                    options,
                    theme,
                    getattr(context, "show_images", False),
                    state.get("started_at"),
                    state.get("ended_at"),
                )
            )
        )
        return component

    from pi_coding_agent.tools.edit import format_edit_call, format_edit_result

    def edit_render_call(args: Any, theme: Any, context: Any) -> Any:
        component = _text_component(context)
        component.set_text(format_edit_call(args, theme, getattr(context, "cwd", cwd)))
        return component

    def edit_render_result(result: Any, options: Any, theme: Any, context: Any) -> Any:
        del options
        text = format_edit_result(getattr(context, "args", None), result, theme, getattr(context, "is_error", False))
        component = _text_component(context)
        # `formatEditResult` returns undefined for "nothing to add" (the error
        # text merely repeats the preview, or there is no diff). Upstream's
        # renderer signature always yields a component, so render empty rather
        # than handing the container a None child.
        component.set_text(text if text is not None else "")
        return component

    from pi_coding_agent.tools.find import format_find_call, format_find_result
    from pi_coding_agent.tools.grep import format_grep_call, format_grep_result
    from pi_coding_agent.tools.ls import format_ls_call, format_ls_result
    from pi_coding_agent.tools.write import format_write_call, format_write_result

    def _simple_call(formatter: Any, pass_cwd: bool) -> Any:
        def render(args: Any, theme: Any, context: Any) -> Any:
            component = _text_component(context)
            if pass_cwd:
                component.set_text(formatter(args, theme, getattr(context, "cwd", cwd)))
            else:
                component.set_text(formatter(args, theme))
            return component

        return render

    def _listing_result(formatter: Any) -> Any:
        def render(result: Any, options: Any, theme: Any, context: Any) -> Any:
            component = _text_component(context)
            component.set_text(formatter(result, options, theme, getattr(context, "show_images", False)))
            return component

        return render

    def write_render_call(args: Any, theme: Any, context: Any) -> Any:
        component = _text_component(context)
        # `write` is the one call renderer that needs the expansion state: the
        # content preview lives on the call, not the result.
        options = SimpleNamespace(expanded=getattr(context, "expanded", False))
        component.set_text(format_write_call(args, options, theme, getattr(context, "cwd", cwd)))
        return component

    def write_render_result(result: Any, options: Any, theme: Any, context: Any) -> Any:
        del options
        component = _text_component(context)
        text = format_write_result(result, theme, getattr(context, "is_error", False))
        component.set_text(text if text is not None else "")
        return component

    return {
        "read": BuiltInToolDefinition(render_call=read_render_call, render_result=read_render_result),
        "bash": BuiltInToolDefinition(render_call=bash_render_call, render_result=bash_render_result),
        "edit": BuiltInToolDefinition(render_call=edit_render_call, render_result=edit_render_result),
        "write": BuiltInToolDefinition(render_call=write_render_call, render_result=write_render_result),
        "ls": BuiltInToolDefinition(
            render_call=_simple_call(format_ls_call, True), render_result=_listing_result(format_ls_result)
        ),
        "grep": BuiltInToolDefinition(
            render_call=_simple_call(format_grep_call, False), render_result=_listing_result(format_grep_result)
        ),
        "find": BuiltInToolDefinition(
            render_call=_simple_call(format_find_call, False), render_result=_listing_result(format_find_result)
        ),
    }
