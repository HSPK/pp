"""Python port of `packages/coding-agent/test/agent-session-dynamic-tools.test.ts`.

Covers the tool registry an `AgentSession` builds: the `PI_*` session
environment the bash tool exposes to spawned commands, the provenance
(`sourceInfo`) reported for builtin / SDK / extension tools, and the rule that
a tool with no `promptSnippet` is active but invisible to the system prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.auth.types import Credential
from pi_ai.types import TextContent

from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.extensions.loader import (
    ExtensionAPI,
    NamedInlineExtension,
    load_extension_factories,
)
from pi_coding_agent.core.extensions.types import ToolDefinition
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.tools.bash import BashSpawnContext, create_bash_tool


async def _create_session(tmp_path: Path, **overrides: Any):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    auth_storage = AuthStorage.create(str(agent_dir / "auth.json"))
    await auth_storage.set("anthropic", Credential(type="api_key", key="test-key"))
    model_runtime = await ModelRuntime.create(credentials=auth_storage, agent_dir=str(agent_dir))
    resource_loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            no_skills=True,
            no_prompt_templates=True,
            no_context_files=True,
        )
    )
    resource_loader.reload()
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            model=model_runtime.get_model("anthropic", "claude-sonnet-4-5"),
            model_runtime=model_runtime,
            settings_manager=SettingsManager.create(str(tmp_path), str(agent_dir)),
            session_manager=SessionManager.in_memory(),
            resource_loader=resource_loader,
            **overrides,
        )
    )
    return result.session


def _find_tool(session: Any, name: str):
    return next((tool for tool in session.get_all_tools() if tool.name == name), None)


async def test_exposes_session_state_to_bash_and_supports_opting_out(tmp_path: Path) -> None:
    session = await _create_session(tmp_path, thinking_level="high")
    try:
        model = session.model
        assert model is not None
        assert (
            "You can inspect PI_* environment variables for current model and session details." in session.system_prompt
        )

        # TypeScript reads the resolved env off a `spawnHook`; this port ports the
        # hook too, but running the real tool and printing the variables observes
        # exactly the same env the child would get, without a stand-in.
        bash_tool = session.agent.state.tools[[tool.name for tool in session.agent.state.tools].index("bash")]
        result = await bash_tool.execute(
            "bash-env",
            {
                "command": 'printf \'%s|%s|%s|%s|%s\' "$PI_SESSION_ID" "$PI_SESSION_FILE" '
                '"$PI_PROVIDER" "$PI_MODEL" "$PI_REASONING_LEVEL"'
            },
        )
        session_env = dict(
            zip(
                ["PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"],
                result.content[0].text.split("|"),
                strict=True,
            )
        )
        assert session_env["PI_SESSION_ID"] == session.session_id
        assert session_env["PI_SESSION_FILE"] == (session.session_file or "")
        assert session_env["PI_PROVIDER"] == model.provider
        assert session_env["PI_MODEL"] == model.id
        assert session_env["PI_REASONING_LEVEL"] == session.thinking_level

        opted_out = create_bash_tool(str(tmp_path), expose_session_environment=False)
        opted_out_result = await opted_out.execute(
            "bash-no-env",
            {
                "command": 'printf \'%s|%s|%s|%s|%s\' "$PI_SESSION_ID" "$PI_SESSION_FILE" '
                '"$PI_PROVIDER" "$PI_MODEL" "$PI_REASONING_LEVEL"'
            },
        )
        assert opted_out_result.content[0].text == "||||"

        # Same opt-out, observed through the ported `BashSpawnHook` the way
        # TypeScript's test does.
        observed: list[BashSpawnContext] = []
        hooked = create_bash_tool(
            str(tmp_path),
            session_environment=lambda: {"PI_SESSION_ID": session.session_id},
            expose_session_environment=False,
            spawn_hook=lambda context: (observed.append(context), context)[1],
        )
        await hooked.execute("bash-hook", {"command": "printf ok"})
        assert len(observed) == 1
        for name in ("PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"):
            assert name not in observed[0].env
    finally:
        session.dispose()


def _dynamic_tool_extension(pi: ExtensionAPI) -> None:
    async def execute(_tool_call_id, _params, _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    pi.register_tool(
        ToolDefinition(
            name="dynamic_tool",
            label="Dynamic Tool",
            description="Tool registered from an extension",
            prompt_snippet="Run dynamic test behavior",
            prompt_guidelines=["Use dynamic_tool when the user asks for dynamic behavior tests."],
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )


async def test_registers_extension_tools_with_their_prompt_contributions(tmp_path: Path) -> None:
    loaded = await load_extension_factories(
        [NamedInlineExtension(name="dynamic", factory=_dynamic_tool_extension)], str(tmp_path)
    )
    assert loaded.errors == []
    session = await _create_session(tmp_path, extensions=loaded.extensions)
    try:
        all_tools = session.get_all_tools()
        dynamic_tool = _find_tool(session, "dynamic_tool")
        read_tool = _find_tool(session, "read")

        assert "dynamic_tool" in [tool.name for tool in all_tools]
        assert dynamic_tool is not None
        assert dynamic_tool.prompt_guidelines == ["Use dynamic_tool when the user asks for dynamic behavior tests."]
        # TypeScript's `<inline:1>` numbering comes from bare factories; a named
        # inline extension is `<inline:dynamic>` in this port (see
        # `load_extension_factories`). Source/scope/origin match exactly.
        assert dynamic_tool.source_info.path == "<inline:dynamic>"
        assert dynamic_tool.source_info.source == "inline"
        assert dynamic_tool.source_info.scope == "temporary"
        assert dynamic_tool.source_info.origin == "top-level"

        assert read_tool is not None
        assert read_tool.source_info.path == "<builtin:read>"
        assert read_tool.source_info.source == "builtin"
        assert read_tool.source_info.scope == "temporary"
        assert read_tool.source_info.origin == "top-level"

        assert "dynamic_tool" in session.get_active_tool_names()
        assert "- dynamic_tool: Run dynamic test behavior" in session.system_prompt
        assert "- Use dynamic_tool when the user asks for dynamic behavior tests." in session.system_prompt
    finally:
        session.dispose()


async def test_returns_source_metadata_for_sdk_custom_tools(tmp_path: Path) -> None:
    async def execute(_tool_call_id, _params, _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    sdk_tool = AgentTool(
        name="sdk_tool",
        label="SDK Tool",
        description="Tool registered through create_agent_session",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    session = await _create_session(tmp_path, custom_tools={"sdk_tool": sdk_tool})
    try:
        tool = _find_tool(session, "sdk_tool")
        assert tool is not None
        assert tool.source_info.path == "<sdk:sdk_tool>"
        assert tool.source_info.source == "sdk"
        assert tool.source_info.scope == "temporary"
        assert tool.source_info.origin == "top-level"
        assert "sdk_tool" in session.get_active_tool_names()
    finally:
        session.dispose()


def _hidden_tool_extension(pi: ExtensionAPI) -> None:
    async def execute(_tool_call_id, _params, _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    pi.register_tool(
        ToolDefinition(
            name="hidden_tool",
            label="Hidden Tool",
            description="Description should not appear in available tools",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )


async def test_keeps_custom_tools_active_but_omits_them_without_a_prompt_snippet(
    tmp_path: Path,
) -> None:
    loaded = await load_extension_factories(
        [NamedInlineExtension(name="hidden", factory=_hidden_tool_extension)], str(tmp_path)
    )
    assert loaded.errors == []
    session = await _create_session(tmp_path, extensions=loaded.extensions)
    try:
        assert "hidden_tool" in [tool.name for tool in session.get_all_tools()]
        assert "hidden_tool" in session.get_active_tool_names()
        assert "hidden_tool" not in session.system_prompt
        assert "Description should not appear in available tools" not in session.system_prompt
    finally:
        session.dispose()


@pytest.mark.parametrize("expose", [True, False])
async def test_bash_never_inherits_a_stale_pi_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expose: bool
) -> None:
    """`resolveSpawnContext` deletes the five `PI_*` names before repopulating
    them, so a host process that already exports one cannot leak it."""
    monkeypatch.setenv("PI_MODEL", "host-leaked-model")
    tool = create_bash_tool(str(tmp_path), expose_session_environment=expose)
    result = await tool.execute("bash-leak", {"command": 'printf "[%s]" "$PI_MODEL"'})
    assert result.content[0].text == "[]"


def _session_start_tool_extension(pi: ExtensionAPI) -> None:
    async def execute(_tool_call_id, _params, _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    def on_session_start(_event: Any, _ctx: Any) -> None:
        pi.register_tool(
            ToolDefinition(
                name="dynamic_tool",
                label="Dynamic Tool",
                description="Tool registered from session_start",
                prompt_snippet="Run dynamic test behavior",
                prompt_guidelines=["Use dynamic_tool when the user asks for dynamic behavior tests."],
                parameters={"type": "object", "properties": {}},
                execute=execute,
            )
        )

    pi.on("session_start", on_session_start)


async def test_refreshes_tool_registry_when_tools_are_registered_after_initialization(tmp_path: Path) -> None:
    loaded = await load_extension_factories(
        [NamedInlineExtension(name="late", factory=_session_start_tool_extension)], str(tmp_path)
    )
    assert loaded.errors == []
    session = await _create_session(tmp_path, extensions=loaded.extensions)
    try:
        assert "dynamic_tool" not in [tool.name for tool in session.get_all_tools()]

        await session.bind_extensions()

        dynamic_tool = _find_tool(session, "dynamic_tool")
        read_tool = _find_tool(session, "read")

        assert "dynamic_tool" in [tool.name for tool in session.get_all_tools()]
        assert dynamic_tool is not None
        assert dynamic_tool.prompt_guidelines == ["Use dynamic_tool when the user asks for dynamic behavior tests."]
        assert dynamic_tool.source_info.path == "<inline:late>"
        assert dynamic_tool.source_info.source == "inline"
        assert dynamic_tool.source_info.scope == "temporary"
        assert dynamic_tool.source_info.origin == "top-level"

        assert read_tool is not None
        assert read_tool.source_info.path == "<builtin:read>"
        assert read_tool.source_info.source == "builtin"

        assert "dynamic_tool" in session.get_active_tool_names()
        assert "- dynamic_tool: Run dynamic test behavior" in session.system_prompt
        assert "- Use dynamic_tool when the user asks for dynamic behavior tests." in session.system_prompt
    finally:
        session.dispose()


def _messy_prompt_contributions_extension(pi: ExtensionAPI) -> None:
    async def execute(_tool_call_id, _params, _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    pi.register_tool(
        ToolDefinition(
            name="messy_tool",
            label="Messy Tool",
            description="Tool with unnormalized prompt contributions",
            prompt_snippet="  First line\r\n\tsecond   line\n\n  third  ",
            prompt_guidelines=["  Be careful  ", "", "   ", "Be careful", "Then stop."],
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )


async def test_normalizes_extension_tool_prompt_contributions(tmp_path: Path) -> None:
    """`_normalizePromptSnippet` / `_normalizePromptGuidelines`.

    The snippet is collapsed to one whitespace-normalized line because the
    system prompt lists tools one per line; the guidelines are trimmed, emptied
    entries dropped, and duplicates removed, preserving first-seen order.
    """
    loaded = await load_extension_factories(
        [NamedInlineExtension(name="messy", factory=_messy_prompt_contributions_extension)], str(tmp_path)
    )
    assert loaded.errors == []
    session = await _create_session(tmp_path, extensions=loaded.extensions)
    try:
        assert "- messy_tool: First line second line third" in session.system_prompt
        assert "\n" not in "First line second line third"

        guideline_lines = [
            line for line in session.system_prompt.splitlines() if line.startswith("- ") and "careful" in line
        ]
        assert guideline_lines == ["- Be careful"]
        assert "- Then stop." in session.system_prompt
    finally:
        session.dispose()
