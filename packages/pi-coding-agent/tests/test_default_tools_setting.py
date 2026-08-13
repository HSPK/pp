"""The `defaultTools` setting seeds the initial built-in tool selection.

Port of upstream 4d9aa837c plus its follow-up 541045ae0. The follow-up matters:
the first revision also narrowed `allowedToolNames`, which disabled extension
tools the user had never listed. Only the *initial built-in selection* is
seeded; the allowlist is left alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_provider
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.loader import ExtensionRuntimeActions, load_extension_factories
from pi_coding_agent.core.extensions.types import ToolDefinition
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

_NOOP_ACTIONS = ExtensionRuntimeActions()


async def _create_session(tmp_path: Path, default_tools: list[str], **options: Any):
    """A session built the way the CLI builds one, so `sdk.py` seeding runs."""
    faux = faux_provider()
    model_runtime = await ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[faux.provider])
    await model_runtime.login(faux.provider.id, "faux-key")
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path / "project"),
            agent_dir=str(tmp_path / "agent"),
            model=faux.models[0],
            model_runtime=model_runtime,
            settings_manager=SettingsManager.in_memory({"defaultTools": default_tools}),
            session_manager=SessionManager.in_memory(str(tmp_path / "project")),
            **options,
        )
    )
    return result.session


async def _ok(*_args: Any, **_kwargs: Any) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text="ok")], details={})


def _sdk_tool(name: str) -> AgentTool:
    return AgentTool(name=name, description=f"{name} for the defaultTools test", execute=_ok)


def _extension_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        label=name,
        description=f"{name} for the defaultTools test",
        prompt_snippet=f"Run {name}",
        execute=_ok,
    )


def _settings_manager(tmp_path: Path, settings: dict) -> SettingsManager:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text(json.dumps(settings))
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    return SettingsManager.create(str(cwd), str(agent_dir))


def test_returns_none_when_unset(tmp_path: Path):
    assert _settings_manager(tmp_path, {}).get_default_tools() is None


def test_returns_the_configured_list(tmp_path: Path):
    manager = _settings_manager(tmp_path, {"defaultTools": ["read", "bash"]})

    assert manager.get_default_tools() == ["read", "bash"]


def test_an_empty_list_reads_as_unset(tmp_path: Path):
    """Matches `tools ? [...tools] : undefined` upstream.

    An empty allowlist would otherwise mean "no tools at all", which is what
    `--no-tools` is for.
    """
    assert _settings_manager(tmp_path, {"defaultTools": []}).get_default_tools() is None


def test_the_caller_cannot_mutate_the_stored_setting(tmp_path: Path):
    """`create_agent_session` filters this list in place for `--exclude-tools`."""
    manager = _settings_manager(tmp_path, {"defaultTools": ["read", "bash"]})

    manager.get_default_tools().remove("bash")

    assert manager.get_default_tools() == ["read", "bash"]


# --------------------------------------------------------------------------
# Behaviour, ported from `packages/coding-agent/test/default-tools-setting.test.ts`.
#
# The upstream file's fourth case ("applies through service-based session
# creation") has no counterpart: `AgentSessionServices` is a documented
# omission of this port (see `core/agent_session_runtime.py`).
# --------------------------------------------------------------------------


async def test_uses_the_configured_list_as_the_initial_built_in_selection(tmp_path: Path):
    session = await _create_session(tmp_path, ["grep", "find"])
    try:
        assert sorted(tool.name for tool in session.get_all_tools()) == [
            "bash",
            "edit",
            "find",
            "grep",
            "ls",
            "read",
            "write",
        ]
        assert session.get_active_tool_names() == ["grep", "find"]
        # Only active tools are described to the model.
        assert "- grep:" in session.system_prompt
        assert "- read:" not in session.system_prompt
    finally:
        session.dispose()


async def test_keeps_extension_and_sdk_custom_tools_enabled(tmp_path: Path):
    """`defaultTools` seeds the built-in selection only.

    An earlier revision also narrowed the allowlist, which silently disabled
    tools the user never listed because they are not built-ins at all.
    """

    def factory(pi: Any) -> None:
        pi.register_tool(_extension_tool("static_tool"))

        async def on_session_start(_event: Any, _ctx: Any) -> None:
            pi.register_tool(_extension_tool("dynamic_tool"))

        pi.on("session_start", on_session_start)

    extensions = await load_extension_factories([factory], str(tmp_path / "project"), _NOOP_ACTIONS)
    assert not extensions.errors
    session = await _create_session(
        tmp_path,
        ["grep"],
        custom_tools={"sdk_tool": _sdk_tool("sdk_tool")},
        extensions=extensions.extensions,
    )
    try:
        await session.bind_extensions()

        assert sorted(session.get_active_tool_names()) == ["dynamic_tool", "grep", "sdk_tool", "static_tool"]
        all_names = {tool.name for tool in session.get_all_tools()}
        assert {"read", "dynamic_tool", "sdk_tool", "static_tool"} <= all_names
    finally:
        session.dispose()


async def test_preserves_explicit_tool_option_precedence(tmp_path: Path):
    allowlisted = await _create_session(tmp_path / "allow", ["grep"], tools=["read"])
    try:
        assert allowlisted.get_active_tool_names() == ["read"]
    finally:
        allowlisted.dispose()

    excluded = await _create_session(tmp_path / "exclude", ["read", "grep"], exclude_tools=["read"])
    try:
        assert excluded.get_active_tool_names() == ["grep"]
    finally:
        excluded.dispose()

    tool_less = await _create_session(tmp_path / "none", ["read"], no_tools="all")
    try:
        assert tool_less.get_all_tools() == []
        assert tool_less.get_active_tool_names() == []
    finally:
        tool_less.dispose()
