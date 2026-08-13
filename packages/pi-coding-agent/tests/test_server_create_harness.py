"""Python port of `packages/coding-agent/test/server/create-harness.test.ts`.

The TypeScript cases `"sets the optional session file in the default bash tool
environment"` and `"keeps bash PI model variables synchronized with Harness
state"` capture the child process environment via a `NodeExecutionEnv`
subclass (`CapturingExecutionEnv`) that records `options?.env` -- the dict
TypeScript's `resolveSpawnContext` builds -- immediately before the real
subprocess is spawned. This port's `create_coding_agent_harness` takes a `cwd`
rather than an injected `ExecutionEnv`, so there is no object to subclass the
way `CapturingExecutionEnv` does. But `tools/bash.py` ports the same
`BashSpawnContext`/`BashSpawnHook` seam TypeScript's `resolveSpawnContext`
exposes (`create_bash_tool(..., spawn_hook=...)`): the hook sees the fully
resolved environment right before the child process is spawned and can record
it without altering execution, which is the same capture point TS's subclass
sits at. `_install_capturing_bash_tool` below wraps `create_bash_tool` at the
one place `create_tool` looks it up (module-global lookup inside
`pi_coding_agent.tools`), so `create_coding_agent_harness`'s real "bash" tool
construction gets a `spawn_hook` without any source change. Both cases are
ported below (`test_sets_the_optional_session_file_...` and
`test_keeps_bash_pi_model_variables_synchronized_...`) using that seam: they
assert the captured environment equals the full ambient environment overlaid
with exactly the expected `PI_*` keys (a superset of, and strictly stronger
than, TypeScript's `toEqual` on the override delta, since TypeScript's
`execution.env` starts empty and only receives the five `prepare`-assigned
keys, whereas this port's resolved environment is the ambient environment plus
those same five keys merged in one step -- so full-dict equality against
"ambient + exactly these five keys" is the only way to catch an extra or
missing key here). They also still run the real bash tool end to end and read
the `PI_*` variables back out of the actual subprocess output, pinning the same
environment-construction behavior against a real child process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_agent.harness.agent_harness import AgentHarness
from pi_agent.harness.session import InMemorySessionStorage, Session
from pi_agent.harness.session.types import SessionMetadata
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import Model, TextContent
from pi_ai.utils.retry import RetryPolicy
from pi_coding_agent import tools as _tools_module
from pi_coding_agent.core.config import get_bin_dir
from pi_coding_agent.core.skills import Skill
from pi_coding_agent.core.source_info import SourceInfo
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions, ContextFile
from pi_coding_agent.server import (
    DEFAULT_HARNESS_TOOL_NAMES,
    CodingAgentHarnessTool,
    CreateCodingAgentHarnessOptions,
    build_coding_agent_harness_system_prompt,
    create_coding_agent_harness,
    create_coding_agent_harness_tools,
)
from pi_coding_agent.utils.shell import get_shell_env


def _install_capturing_bash_tool(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Record the resolved `BashSpawnContext.env` for every bash tool built afterwards.

    This is the port's equivalent of TS's `CapturingExecutionEnv`: it observes
    the environment a real bash execution is about to use without altering
    that execution. Returns the list `captured_envs` is appended to (one dict
    per `bash.execute` call).
    """
    original_create_bash_tool = _tools_module.create_bash_tool
    captured_envs: list[dict[str, str]] = []

    def capturing_create_bash_tool(cwd: str, *args: Any, **kwargs: Any) -> AgentTool:
        def spawn_hook(context: Any) -> Any:
            captured_envs.append(dict(context.env))
            return context

        kwargs["spawn_hook"] = spawn_hook
        return original_create_bash_tool(cwd, *args, **kwargs)

    monkeypatch.setattr(_tools_module, "create_bash_tool", capturing_create_bash_tool)
    return captured_envs


_MODEL = Model(id="gemini-2.5-flash", provider="google")


def _make_session(session_id: str) -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id=session_id, created_at=1)))


def _create_prompt_tool(
    name: str,
    prompt_snippet: str | None = None,
    prompt_guidelines: list[str] | None = None,
    description: str | None = None,
) -> CodingAgentHarnessTool:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details=None)

    return CodingAgentHarnessTool(
        tool=AgentTool(
            name=name,
            label=name,
            description=description or f"{name} description",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        ),
        prompt_snippet=prompt_snippet,
        prompt_guidelines=list(prompt_guidelines or []),
    )


_DEFAULT_PROMPT_TOOLS = [
    _create_prompt_tool("read", "Read file contents", ["Use read to examine files instead of cat or sed."]),
    _create_prompt_tool(
        "bash",
        "Execute bash commands (ls, grep, find, etc.)",
        ["You can inspect PI_* environment variables for current model and session details."],
    ),
    _create_prompt_tool("edit", "Edit files", ["Edit carefully."]),
    _create_prompt_tool("write", "Create or overwrite files", ["Use write only for new files or complete rewrites."]),
]


async def test_adds_coding_agent_policy_to_explicit_harness_options() -> None:
    harness, suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=_make_session("harness-session"),
            model=_MODEL,
            cwd="/workspace",
            harness_options={
                "thinking_level": "high",
                "stream_options": {"max_tokens": 123},
                "retry": RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10),
                "steering_mode": "all",
                "follow_up_mode": "all",
            },
        )
    )
    try:
        assert suspended == []
        assert await harness.get_active_tools() == list(DEFAULT_HARNESS_TOOL_NAMES)
        assert [tool.name for tool in await harness.get_tools()] == list(DEFAULT_HARNESS_TOOL_NAMES)
        assert await harness.get_stream_options() == {"max_tokens": 123}
        retry = await harness.get_retry_policy()
        assert (retry.enabled, retry.max_retries, retry.base_delay_ms) == (True, 2, 10)
        assert await harness.get_steering_mode() == "all"
        assert await harness.get_follow_up_mode() == "all"
    finally:
        await harness.close()


def test_preserves_coding_agent_prompt_snippets_and_guideline_order() -> None:
    prompt = build_coding_agent_harness_system_prompt(
        "/workspace", _DEFAULT_PROMPT_TOOLS, ["read", "bash", "edit", "write"]
    )

    assert "- read: Read file contents" in prompt
    assert "- bash: Execute bash commands (ls, grep, find, etc.)" in prompt
    assert "Use read to examine files instead of cat or sed." in prompt
    assert "You can inspect PI_* environment variables for current model and session details." in prompt
    assert prompt.index("Use read to examine files") < prompt.index("You can inspect PI_* environment variables")


async def test_preserves_caller_supplied_tools_and_activation() -> None:
    custom_tool = _create_prompt_tool("inspect", description="Inspect the configured service")

    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=_make_session("custom-harness-session"),
            model=_MODEL,
            cwd="/workspace",
            tools=[custom_tool],
            active_tool_names=[],
            system_prompt="Server-owned prompt",
        )
    )
    try:
        assert [tool.name for tool in await harness.get_tools()] == ["inspect"]
        assert await harness.get_active_tools() == []
    finally:
        await harness.close()


async def test_builds_each_default_system_prompt_from_current_harness_tool_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    original_create = AgentHarness.create

    async def capturing_create(options: Any) -> Any:
        captured["system_prompt"] = options.system_prompt
        return await original_create(options)

    monkeypatch.setattr(AgentHarness, "create", staticmethod(capturing_create))
    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=_make_session("dynamic-prompt-session"),
            model=_MODEL,
            cwd="/workspace",
        )
    )
    monkeypatch.undo()

    async def resolve_system_prompt() -> str:
        system_prompt = captured["system_prompt"]
        if isinstance(system_prompt, str):
            return system_prompt
        return await system_prompt()

    try:
        initial_prompt = await resolve_system_prompt()
        assert "- read: Read file contents" in initial_prompt
        assert "- bash: Execute bash commands (ls, grep, find, etc.)" in initial_prompt
        assert "- edit: Make precise file edits with exact text replacement" in initial_prompt
        assert "- write: Create or overwrite files" in initial_prompt

        await harness.set_active_tools(["write"])
        write_prompt = await resolve_system_prompt()
        assert "- write: Create or overwrite files" in write_prompt
        assert "- read:" not in write_prompt
        assert "- bash:" not in write_prompt

        read = next(tool for tool in await harness.get_tools() if tool.name == "read")
        await harness.set_tools([read])
        read_prompt = await resolve_system_prompt()
        assert "- read: Read file contents" in read_prompt
        assert "- write:" not in read_prompt

        inspect_tool = _create_prompt_tool(
            "inspect",
            "  Inspect\nthe   configured service  ",
            ["Use inspect for service diagnostics."],
        )
        await harness.set_tools([inspect_tool])
        inspect_prompt = await resolve_system_prompt()
        assert "- inspect: Inspect the configured service" in inspect_prompt
        assert "Use inspect for service diagnostics." in inspect_prompt
    finally:
        await harness.close()


def test_omits_active_custom_tools_without_prompt_metadata_from_the_textual_tools_section() -> None:
    prompt = build_coding_agent_harness_system_prompt("/workspace", [_create_prompt_tool("hidden")], ["hidden"])

    assert "Available tools:\n(none)" in prompt
    assert "- hidden:" not in prompt
    assert "hidden description" not in prompt


@pytest.mark.parametrize(
    ("name", "built_in_snippet", "built_in_guideline"),
    [
        (
            "bash",
            "Execute bash commands (ls, grep, find, etc.)",
            "You can inspect PI_* environment variables for current model and session details.",
        ),
        ("read", "Read file contents", "Use read to examine files instead of cat or sed."),
        (
            "edit",
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
            "Use edit for precise changes (edits[].oldText must match exactly)",
        ),
        ("write", "Create or overwrite files", "Use write only for new files or complete rewrites."),
    ],
)
def test_does_not_infer_prompt_metadata_for_a_caller_supplied_replacement(
    name: str, built_in_snippet: str, built_in_guideline: str
) -> None:
    prompt = build_coding_agent_harness_system_prompt("/workspace", [_create_prompt_tool(name)], [name])

    assert "Available tools:\n(none)" in prompt
    assert built_in_snippet not in prompt
    assert built_in_guideline not in prompt


def test_builds_the_default_prompt_from_active_tools_and_resolved_prompt_resources() -> None:
    prompt = build_coding_agent_harness_system_prompt(
        "/workspace",
        _DEFAULT_PROMPT_TOOLS,
        ["write", "read"],
        BuildSystemPromptOptions(
            cwd="/workspace",
            context_files=[ContextFile(path="/workspace/AGENTS.md", content="Follow project policy.")],
            skills=[
                Skill(
                    name="review",
                    description="Review server changes",
                    file_path="/skills/review/SKILL.md",
                    base_dir="/skills/review",
                    source_info=SourceInfo(
                        path="/skills/review/SKILL.md",
                        source="test",
                        scope="temporary",
                        origin="top-level",
                    ),
                    disable_model_invocation=False,
                )
            ],
        ),
    )

    assert "- write: Create or overwrite files" in prompt
    assert "- read: Read file contents" in prompt
    assert "- bash:" not in prompt
    assert "You can inspect PI_* environment variables" not in prompt
    assert '<project_instructions path="/workspace/AGENTS.md">' in prompt
    assert "<name>review</name>" in prompt
    assert prompt.index("Use write only for new files or complete rewrites.") < prompt.index(
        "Use read to examine files instead of cat or sed."
    )


def test_default_harness_tools_use_the_built_in_prompt_contributions() -> None:
    tools = create_coding_agent_harness_tools("/workspace")

    assert [tool.name for tool in tools] == list(DEFAULT_HARNESS_TOOL_NAMES)
    read = next(tool for tool in tools if tool.name == "read")
    assert read.prompt_snippet == "Read file contents"
    assert read.prompt_guidelines == ["Use read to examine files instead of cat or sed."]


async def test_sets_the_optional_session_file_in_the_default_bash_tool_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # TS's `CapturingExecutionEnv` is constructed with
    # `shellEnv: { PI_SESSION_FILE: "/stale/parent.jsonl", PI_CODING_AGENT: "true" }`: an
    # ambient shell environment the bash tool inherits and then partially overwrites. This
    # port's bash tool spawns against the real process environment (`get_shell_env`), so the
    # ambient state is set the same way, via the actual environment.
    monkeypatch.setenv("PI_SESSION_FILE", "/stale/parent.jsonl")
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    captured_envs = _install_capturing_bash_tool(monkeypatch)

    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=_make_session("session-file-harness"),
            model=_MODEL,
            cwd=str(tmp_path),
            harness_options={"thinking_level": "high"},
            session_file="/sessions/current.jsonl",
        )
    )
    try:
        bash = next(tool for tool in await harness.get_tools() if tool.name == "bash").tool
        assert bash.execute is not None
        result = await bash.execute(
            "bash-call",
            {
                "command": (
                    "printf '%s' \"$PI_SESSION_ID|$PI_SESSION_FILE|$PI_PROVIDER|"
                    '$PI_MODEL|$PI_REASONING_LEVEL|$PI_CODING_AGENT"'
                )
            },
        )

        # Port of `expect(env.executionOverrides).toEqual({...})`: the resolved
        # spawn environment is exactly the ambient environment (PI_CODING_AGENT
        # untouched, since it is not one of the five keys the harness sets) with
        # the five PI_* keys overlaid from the harness's session, session_file
        # option, model and thinking level. Full-dict equality (rather than
        # checking only the five keys) also catches an extra/leaked key, which a
        # subset check would miss.
        assert len(captured_envs) == 1
        expected_env = get_shell_env(get_bin_dir())
        expected_env.update(
            {
                "PI_SESSION_ID": "session-file-harness",
                "PI_SESSION_FILE": "/sessions/current.jsonl",
                "PI_PROVIDER": "google",
                "PI_MODEL": "gemini-2.5-flash",
                "PI_REASONING_LEVEL": "high",
            }
        )
        assert captured_envs[0] == expected_env

        # The subprocess output additionally proves the environment was actually
        # applied to a real child process, not merely computed.
        assert result.content == [
            TextContent(text="session-file-harness|/sessions/current.jsonl|google|gemini-2.5-flash|high|true")
        ]
    finally:
        await harness.close()


async def test_keeps_bash_pi_model_variables_synchronized_with_harness_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PI_SESSION_FILE", "/stale/parent.jsonl")
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    captured_envs = _install_capturing_bash_tool(monkeypatch)

    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=_make_session("dynamic-bash-session"),
            model=_MODEL,
            cwd=str(tmp_path),
            harness_options={"thinking_level": "high"},
        )
    )
    try:
        await harness.set_model(Model(id="claude-sonnet-4-5", provider="anthropic"))
        await harness.set_thinking_level("low")
        bash = next(tool for tool in await harness.get_tools() if tool.name == "bash").tool
        assert bash.execute is not None

        result = await bash.execute(
            "bash-call",
            {
                "command": (
                    "printf '%s:%s' \"${PI_SESSION_FILE+x}\" "
                    '"$PI_SESSION_ID|$PI_PROVIDER|$PI_MODEL|$PI_REASONING_LEVEL|$PI_CODING_AGENT"'
                )
            },
        )

        # Port of `expect(env.executionOverrides).toEqual({...})`: no `session_file`
        # option was given, so PI_SESSION_FILE must still be *set* to "" (not simply
        # absent), matching TS's `sessionFile ?? ""`. The model/provider/thinking-level
        # values reflect the *post-construction* `set_model`/`set_thinking_level`
        # calls, proving the bash tool reads harness state live at execution time
        # rather than snapshotting it at tool creation.
        assert len(captured_envs) == 1
        expected_env = get_shell_env(get_bin_dir())
        expected_env.update(
            {
                "PI_SESSION_ID": "dynamic-bash-session",
                "PI_SESSION_FILE": "",
                "PI_PROVIDER": "anthropic",
                "PI_MODEL": "claude-sonnet-4-5",
                "PI_REASONING_LEVEL": "low",
            }
        )
        assert captured_envs[0] == expected_env
        assert "PI_SESSION_FILE" in captured_envs[0]
        assert captured_envs[0]["PI_SESSION_FILE"] == ""

        # `${PI_SESSION_FILE+x}` expands to "x" only when the variable is set (even to
        # ""), matching TS's `Object.hasOwn(env.executionOverrides ?? {}, "PI_SESSION_FILE")`
        # check -- but proven here against the real subprocess's environment, not just
        # the dict handed to the spawn call.
        assert result.content == [TextContent(text="x:dynamic-bash-session|anthropic|claude-sonnet-4-5|low|true")]
    finally:
        await harness.close()
