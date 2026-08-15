"""Python port of `packages/coding-agent/test/suite/regressions/6162-extension-active-tools-next-turn.test.ts`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import create_harness
from pi_agent.types import AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import BeforeAgentStartEventResult, ToolDefinition


def _tool_names(context: Any) -> list[str]:
    return sorted(tool.name for tool in (context.tools or []))


async def test_applies_set_active_tools_before_the_next_provider_request_in_the_same_run(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def switch(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
            pi.set_active_tools(["after_switch"])
            return AgentToolResult(content=[TextContent(text="switched")], details={})

        async def after(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="after")], details={})

        pi.register_tool(
            ToolDefinition(
                name="switch_tools",
                label="Switch Tools",
                description="Switch the active extension tool set",
                prompt_snippet="Switch to the next extension tool",
                parameters={"type": "object", "properties": {}},
                execute=switch,
            )
        )
        pi.register_tool(
            ToolDefinition(
                name="after_switch",
                label="After Switch",
                description="Tool that should be available after switching",
                prompt_snippet="Run after the active tool set changes",
                parameters={"type": "object", "properties": {}},
                execute=after,
            )
        )

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.session.set_active_tools_by_name(["switch_tools"])

        provider_tool_names: list[list[str]] = []

        def first(context, _options, _state, _model):
            provider_tool_names.append(_tool_names(context))
            return faux_assistant_message([faux_tool_call("switch_tools", {})], stop_reason="toolUse")

        def second(context, _options, _state, _model):
            provider_tool_names.append(_tool_names(context))
            return faux_assistant_message("done")

        harness.set_responses([first, second])

        assert harness.session.get_active_tool_names() == ["switch_tools"]

        await harness.session.prompt("start")

        assert harness.session.get_active_tool_names() == ["after_switch"]
        assert provider_tool_names == [["switch_tools"], ["after_switch"]]
    finally:
        harness.cleanup()


async def test_records_additive_active_tool_changes_on_the_current_tool_result(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def load_more(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
            pi.set_active_tools([*pi.get_active_tools(), "after_load"])
            return AgentToolResult(content=[TextContent(text="loaded")], details={})

        async def after(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="after")], details={})

        pi.register_tool(
            ToolDefinition(
                name="load_more_tools",
                label="Load More Tools",
                description="Load more tools",
                parameters={"type": "object", "properties": {}},
                execute=load_more,
            )
        )
        pi.register_tool(
            ToolDefinition(
                name="after_load",
                label="After Load",
                description="Tool available after loading",
                parameters={"type": "object", "properties": {}},
                execute=after,
            )
        )

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.session.set_active_tools_by_name(["load_more_tools"])

        added_tool_names: list[list[str]] = []

        def first(_context, _options, _state, _model):
            return faux_assistant_message([faux_tool_call("load_more_tools", {})], stop_reason="toolUse")

        def second(context, _options, _state, _model):
            added_tool_names.append(
                [
                    name
                    for message in context.messages
                    if getattr(message, "role", None) == "toolResult"
                    for name in (message.added_tool_names or [])
                ]
            )
            return faux_assistant_message("done")

        harness.set_responses([first, second])

        await harness.session.prompt("start")

        assert harness.session.get_active_tool_names() == ["load_more_tools", "after_load"]
        assert added_tool_names == [["after_load"]]
    finally:
        harness.cleanup()


async def test_preserves_before_agent_start_system_prompt_overrides_when_tools_change_mid_run(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def on_before_agent_start(event, _ctx) -> BeforeAgentStartEventResult:
            return BeforeAgentStartEventResult(system_prompt=f"{event.system_prompt}\n\nkeep this run override")

        pi.on("before_agent_start", on_before_agent_start)

        async def switch(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
            pi.set_active_tools(["after_switch"])
            return AgentToolResult(content=[TextContent(text="switched")], details={})

        async def after(_tool_call_id, _params, _signal=None, _on_update=None, _ctx=None) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="after")], details={})

        pi.register_tool(
            ToolDefinition(
                name="switch_tools",
                label="Switch Tools",
                description="Switch the active extension tool set",
                prompt_snippet="Switch to the next extension tool",
                parameters={"type": "object", "properties": {}},
                execute=switch,
            )
        )
        pi.register_tool(
            ToolDefinition(
                name="after_switch",
                label="After Switch",
                description="Tool that should be available after switching",
                prompt_snippet="Run after the active tool set changes",
                parameters={"type": "object", "properties": {}},
                execute=after,
            )
        )

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.session.set_active_tools_by_name(["switch_tools"])

        provider_system_prompts: list[str] = []
        provider_tool_names: list[list[str]] = []

        def first(context, _options, _state, _model):
            provider_system_prompts.append(context.system_prompt or "")
            provider_tool_names.append(_tool_names(context))
            return faux_assistant_message([faux_tool_call("switch_tools", {})], stop_reason="toolUse")

        def second(context, _options, _state, _model):
            provider_system_prompts.append(context.system_prompt or "")
            provider_tool_names.append(_tool_names(context))
            return faux_assistant_message("done")

        harness.set_responses([first, second])

        await harness.session.prompt("start")

        assert provider_tool_names == [["switch_tools"], ["after_switch"]]
        assert len(provider_system_prompts) == 2
        assert "keep this run override" in provider_system_prompts[0]
        assert "keep this run override" in provider_system_prompts[1]
    finally:
        harness.cleanup()
