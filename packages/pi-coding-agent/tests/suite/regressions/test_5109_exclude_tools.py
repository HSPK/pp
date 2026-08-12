"""Python port of `packages/coding-agent/test/suite/regressions/5109-exclude-tools.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import create_harness
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import ToolDefinition


def _register_tools(pi: ExtensionAPI) -> None:
    async def execute(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    pi.register_tool(
        ToolDefinition(
            name="ask_question",
            label="Ask Question",
            description="Ask a question",
            prompt_snippet="Ask a question",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="dynamic_tool",
            label="Dynamic Tool",
            description="Dynamic test tool",
            prompt_snippet="Run dynamic test behavior",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )


def _extension_factories() -> list:
    def factory(pi: ExtensionAPI) -> None:
        async def on_session_start(_event, _ctx) -> None:
            _register_tools(pi)

        pi.on("session_start", on_session_start)

    return [factory]


def _tool_names(tools) -> list[str]:
    return sorted(tool.name for tool in tools)


async def test_filters_builtin_and_extension_tools_from_available_and_active_tools(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        excluded_tool_names=["read", "ask_question"],
        extension_factories=_extension_factories(),
    )
    try:
        await harness.session.bind_extensions()

        all_tool_names = _tool_names(harness.session.get_all_tools())
        assert "read" not in all_tool_names
        assert "ask_question" not in all_tool_names
        assert "bash" in all_tool_names
        assert "dynamic_tool" in all_tool_names
        assert sorted(harness.session.get_active_tool_names()) == ["bash", "dynamic_tool", "edit", "write"]
        assert "- read:" not in harness.session.system_prompt
        assert "ask_question" not in harness.session.system_prompt
        assert "- dynamic_tool: Run dynamic test behavior" in harness.session.system_prompt
    finally:
        harness.cleanup()


async def test_excluded_tools_override_the_allowlist(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        allowed_tool_names=["read", "bash", "ask_question"],
        excluded_tool_names=["read", "ask_question"],
        initial_active_tool_names=["read", "bash", "ask_question"],
        extension_factories=_extension_factories(),
    )
    try:
        await harness.session.bind_extensions()

        assert _tool_names(harness.session.get_all_tools()) == ["bash"]
        assert harness.session.get_active_tool_names() == ["bash"]
        assert "- bash:" in harness.session.system_prompt
        assert "- read:" not in harness.session.system_prompt
        assert "ask_question" not in harness.session.system_prompt
    finally:
        harness.cleanup()
