"""Python port of `packages/coding-agent/test/suite/regressions/3592-no-builtin-tools-keeps-extension-tools.test.ts`.

`noTools: "builtin"` must disable the built-in default tools while leaving
extension-registered tools active; `noTools: "all"` must disable everything.

Two shape differences from the TypeScript test, neither weakening an
assertion:

- The first two cases build the session through the harness rather than
  `createAgentSession`, because `create_agent_session` has no `extensions`
  parameter here (TypeScript threads extensions in via the resource loader).
  `no_tools: "builtin"` is exactly `initial_active_tool_names=[]` and
  `no_tools: "all"` is exactly `allowed_tool_names=[]` in `sdk.py`, which the
  third case exercises through the real `create_agent_session`.
- The third case calls `create_agent_session` directly: TypeScript's
  `createAgentSessionServices`/`createAgentSessionFromServices` pair is not
  ported (see `core/agent_session_runtime.py`'s "Dropped:
  `AgentSessionServices`" note), and `create_agent_session` is the single
  entry point that `no_tools` has to flow through here.
"""

from __future__ import annotations

from pathlib import Path

from harness import Harness, create_harness
from pi_agent.types import AgentToolResult
from pi_ai.providers.faux import faux_provider
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import ToolDefinition
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager


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


async def _create_harness_session(tmp_path: Path, no_tools: str) -> Harness:
    if no_tools == "all":
        harness = await create_harness(tmp_path, extension_factories=[_factory], allowed_tool_names=[])
    else:
        harness = await create_harness(tmp_path, extension_factories=[_factory], initial_active_tool_names=[])
    await harness.session.bind_extensions()
    return harness


async def test_keeps_extension_tools_active_when_builtin_defaults_are_disabled(tmp_path: Path) -> None:
    harness = await _create_harness_session(tmp_path, "builtin")
    try:
        session = harness.session
        assert sorted(tool.name for tool in session.get_all_tools()) == [
            "bash",
            "dynamic_tool",
            "edit",
            "find",
            "grep",
            "ls",
            "read",
            "write",
        ]
        assert session.get_active_tool_names() == ["dynamic_tool"]
        assert "- dynamic_tool: Run dynamic test behavior" in session.system_prompt
        assert "- read:" not in session.system_prompt
        assert "- bash:" not in session.system_prompt
    finally:
        harness.cleanup()


async def test_still_disables_all_tools_when_no_tools_is_all(tmp_path: Path) -> None:
    harness = await _create_harness_session(tmp_path, "all")
    try:
        session = harness.session
        assert session.get_all_tools() == []
        assert session.get_active_tool_names() == []
        assert "Available tools:\n(none)" in session.system_prompt
    finally:
        harness.cleanup()


async def test_propagates_no_tools_through_sdk_session_creation(tmp_path: Path) -> None:
    faux = faux_provider()
    model_runtime = await ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[faux.provider])
    await model_runtime.login(faux.provider.id, "faux-key")
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            model_runtime=model_runtime,
            model=faux.get_model(),
            session_manager=SessionManager.in_memory(str(tmp_path)),
            settings_manager=SettingsManager.in_memory(),
            no_tools="builtin",
        )
    )
    session = result.session
    try:
        assert session.get_active_tool_names() == []
        assert "Available tools:\n(none)" in session.system_prompt
        assert "- read:" not in session.system_prompt
    finally:
        session.dispose()
