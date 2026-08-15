"""Build a coding-agent-flavoured `AgentHarness`.

Python port of `packages/coding-agent/src/server/create-harness.ts`.

The generic `AgentHarness` in `pi_agent` knows nothing about coding tools or
the coding system prompt. This wires the two together: it supplies the default
read/bash/edit/write tool set, and a system prompt that regenerates itself from
whichever tools are active when it is asked for -- so activating or
deactivating a tool mid-session updates the prompt without the caller
rebuilding anything.

Two shapes differ from TypeScript. Upstream's `AgentHarnessTool` carries a
`toolContext` that the harness threads into `execute`; this port's tool
factories already close over their `cwd`, so `CodingAgentHarnessTool` only adds
the prompt metadata. And upstream reads each tool's contribution from a
per-tool `*SystemPromptContribution` export, which this port keeps in one table
in `core.agent_session`; that table is the source here too.

The default bash tool's `PI_*` session environment (`PI_SESSION_ID`,
`PI_SESSION_FILE`, `PI_PROVIDER`, `PI_MODEL`, `PI_REASONING_LEVEL`) mirrors
`create-harness.ts`'s `createBashTool(..., { prepare })`: it reads the live
`AgentHarness` model/thinking level at command-execution time (not at tool
construction time), so `harness.set_model()`/`set_thinking_level()` calls
mid-session are reflected in the next command. `PI_SESSION_FILE` is always set,
even to `""`, matching `sessionFile ?? ""` upstream.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pi_agent.harness.agent_harness import AgentHarness, AgentHarnessOptions, HarnessTool, SuspendedOperation

from pi_coding_agent.core.agent_session import TOOL_PROMPT_CONTRIBUTIONS
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions, build_system_prompt
from pi_coding_agent.tools import create_tool

DEFAULT_HARNESS_TOOL_NAMES = ("read", "bash", "edit", "write")

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class CodingAgentHarnessTool(HarnessTool):
    """A harness tool plus what it contributes to the system prompt."""

    prompt_snippet: str | None = None
    prompt_guidelines: list[str] = field(default_factory=list)


def create_coding_agent_harness_tools(
    cwd: str,
    names: Sequence[str] = DEFAULT_HARNESS_TOOL_NAMES,
    *,
    bash_session_environment: Callable[[], Awaitable[dict[str, str]]] | None = None,
) -> list[CodingAgentHarnessTool]:
    """The default coding tool set, each tool carrying its prompt contribution.

    ``bash_session_environment``, if given, is forwarded only to the `bash`
    tool (mirroring `create_tool`, which ignores `session_environment` for
    every other tool name).
    """
    tools: list[CodingAgentHarnessTool] = []
    for name in names:
        snippet, guidelines = TOOL_PROMPT_CONTRIBUTIONS.get(name, (None, []))
        tools.append(
            CodingAgentHarnessTool(
                tool=create_tool(
                    name,
                    cwd,
                    session_environment=bash_session_environment if name == "bash" else None,
                ),
                prompt_snippet=snippet,
                prompt_guidelines=list(guidelines),
            )
        )
    return tools


def build_coding_agent_harness_system_prompt(
    cwd: str,
    tools: Sequence[CodingAgentHarnessTool],
    active_tool_names: Sequence[str],
    system_prompt_options: BuildSystemPromptOptions | None = None,
) -> str:
    """Build the system prompt from the tools that are currently active.

    Inactive tools contribute nothing, and a name in `active_tool_names` that
    matches no tool is skipped, so a stale name cannot break prompt generation.
    """
    by_name = {tool.name: tool for tool in tools}
    active_tools = [by_name[name] for name in active_tool_names if name in by_name]

    tool_snippets: dict[str, str] = {}
    for tool in active_tools:
        if not tool.prompt_snippet:
            continue
        # Snippets are written multi-line for readability but must render on one line.
        tool_snippets[tool.name] = _WHITESPACE_PATTERN.sub(" ", tool.prompt_snippet).strip()

    prompt_guidelines: list[str] = []
    for tool in active_tools:
        prompt_guidelines.extend(tool.prompt_guidelines)

    base = system_prompt_options or BuildSystemPromptOptions(cwd=cwd)
    return build_system_prompt(
        BuildSystemPromptOptions(
            cwd=cwd,
            custom_prompt=base.custom_prompt,
            append_system_prompt=base.append_system_prompt,
            context_files=base.context_files,
            skills=base.skills,
            selected_tools=[tool.name for tool in active_tools],
            tool_snippets=tool_snippets,
            prompt_guidelines=prompt_guidelines,
        )
    )


@dataclass(kw_only=True)
class CreateCodingAgentHarnessOptions:
    """Options for :func:`create_coding_agent_harness`.

    `cwd` replaces upstream's `env: ExecutionEnv`, since this port's tools take
    a working directory rather than an injected execution environment.
    """

    session: Any
    models: Any = None
    model: Any = None
    cwd: str = "."
    tools: list[CodingAgentHarnessTool] | None = None
    active_tool_names: list[str] | None = None
    system_prompt: str | Callable[[], str | Awaitable[str]] | None = None
    system_prompt_options: BuildSystemPromptOptions | None = None
    session_file: str | None = None
    """Path to the JSONL session file exposed to default bash commands as
    `PI_SESSION_FILE`. Port of `CreateCodingAgentHarnessOptions.sessionFile`."""
    harness_options: dict[str, Any] = field(default_factory=dict)
    """Extra `AgentHarnessOptions` fields passed through unchanged."""


async def create_coding_agent_harness(
    options: CreateCodingAgentHarnessOptions,
) -> tuple[AgentHarness, list[SuspendedOperation]]:
    """Create an `AgentHarness` with the coding tool set and system prompt."""
    harness_ref: dict[str, AgentHarness] = {}

    def get_harness() -> AgentHarness:
        harness = harness_ref.get("harness")
        if harness is None:
            raise RuntimeError("Coding-agent harness callback ran before harness initialization")
        return harness

    if options.tools is not None:
        tools = options.tools
    else:
        metadata = await options.session.get_metadata()

        async def bash_session_environment() -> dict[str, str]:
            harness = get_harness()
            model, thinking_level = await asyncio.gather(harness.get_model(), harness.get_thinking_level())
            return {
                "PI_SESSION_ID": metadata.id,
                "PI_SESSION_FILE": options.session_file if options.session_file is not None else "",
                "PI_PROVIDER": model.provider,
                "PI_MODEL": model.id,
                "PI_REASONING_LEVEL": thinking_level,
            }

        tools = create_coding_agent_harness_tools(options.cwd, bash_session_environment=bash_session_environment)

    active_tool_names = (
        list(options.active_tool_names) if options.active_tool_names is not None else [tool.name for tool in tools]
    )

    async def default_system_prompt() -> str:
        harness = get_harness()
        return build_coding_agent_harness_system_prompt(
            options.cwd,
            await harness.get_tools(),
            await harness.get_active_tools(),
            options.system_prompt_options,
        )

    harness, suspended = await AgentHarness.create(
        AgentHarnessOptions(
            session=options.session,
            models=options.models,
            model=options.model,
            tools=tools,
            active_tool_names=active_tool_names,
            system_prompt=options.system_prompt if options.system_prompt is not None else default_system_prompt,
            **options.harness_options,
        )
    )
    harness_ref["harness"] = harness
    return harness, suspended


__all__ = [
    "DEFAULT_HARNESS_TOOL_NAMES",
    "CodingAgentHarnessTool",
    "CreateCodingAgentHarnessOptions",
    "build_coding_agent_harness_system_prompt",
    "create_coding_agent_harness",
    "create_coding_agent_harness_tools",
]
