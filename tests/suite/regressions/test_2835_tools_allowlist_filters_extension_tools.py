"""Python port of `packages/coding-agent/test/suite/regressions/2835-tools-allowlist-filters-extension-tools.test.ts`.

Like the TypeScript test, the extension tool is registered from a
`session_start` handler and picked up by `session.bind_extensions()`, so the
allowlist has to filter a tool that did not exist when the session was
constructed.
"""

from __future__ import annotations

from pathlib import Path

from harness import Harness, create_harness
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import ToolDefinition


def _factory(pi: ExtensionAPI) -> None:
    async def execute(tool_call_id: str, params, signal, on_update, ctx) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    async def on_session_start(event, ctx) -> None:
        pi.register_tool(
            ToolDefinition(
                name="dynamic_tool",
                label="Dynamic Tool",
                description="Tool registered from session_start",
                prompt_snippet="Run dynamic test behavior",
                execute=execute,
            )
        )

    pi.on("session_start", on_session_start)


async def _create_session(tmp_path: Path, allowed_tool_names: list[str]) -> Harness:
    harness = await create_harness(
        tmp_path,
        extension_factories=[_factory],
        allowed_tool_names=allowed_tool_names,
    )
    await harness.session.bind_extensions()
    return harness


async def test_allows_only_explicitly_listed_builtin_and_extension_tools(tmp_path: Path) -> None:
    harness = await _create_session(tmp_path, ["read", "dynamic_tool"])
    try:
        session = harness.session
        assert sorted(tool.name for tool in session.get_all_tools()) == ["dynamic_tool", "read"]
        assert sorted(session.get_active_tool_names()) == ["dynamic_tool", "read"]
        assert "- read: Read file contents" in session.system_prompt
        assert "- dynamic_tool: Run dynamic test behavior" in session.system_prompt
        assert "- bash:" not in session.system_prompt
        assert "- edit:" not in session.system_prompt
    finally:
        harness.cleanup()


async def test_disables_all_tools_when_the_allowlist_is_empty(tmp_path: Path) -> None:
    harness = await _create_session(tmp_path, [])
    try:
        session = harness.session
        assert session.get_all_tools() == []
        assert session.get_active_tool_names() == []
        assert "Available tools:\n(none)" in session.system_prompt
        assert "dynamic_tool" not in session.system_prompt
    finally:
        harness.cleanup()
