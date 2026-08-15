"""Tests for `pi_coding_agent.server.create_harness`.

Supplementary to `tests/test_server_create_harness.py` (the port of
`packages/coding-agent/test/server/create-harness.test.ts`): these pin the
prompt-assembly rules that the factory depends on, plus the harness wiring
itself.
"""

from __future__ import annotations

import pytest
from pi_agent.harness.session import InMemorySessionStorage, Session
from pi_agent.harness.session.types import SessionMetadata
from pi_ai.types import Model

from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions
from pi_coding_agent.server import (
    DEFAULT_HARNESS_TOOL_NAMES,
    CodingAgentHarnessTool,
    CreateCodingAgentHarnessOptions,
    build_coding_agent_harness_system_prompt,
    create_coding_agent_harness,
    create_coding_agent_harness_tools,
)


def make_session() -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id="session", created_at=1)))


def test_default_tools_carry_their_prompt_contributions():
    tools = create_coding_agent_harness_tools("/workspace")

    assert [tool.name for tool in tools] == list(DEFAULT_HARNESS_TOOL_NAMES)
    assert all(tool.prompt_snippet for tool in tools)
    read = next(tool for tool in tools if tool.name == "read")
    assert read.prompt_guidelines == ["Use read to examine files instead of cat or sed."]


def test_system_prompt_only_includes_active_tools():
    tools = create_coding_agent_harness_tools("/workspace")

    prompt = build_coding_agent_harness_system_prompt("/workspace", tools, ["read"])

    assert "Read file contents" in prompt
    assert "Create or overwrite files" not in prompt


def test_system_prompt_ignores_active_names_with_no_matching_tool():
    tools = create_coding_agent_harness_tools("/workspace")

    prompt = build_coding_agent_harness_system_prompt("/workspace", tools, ["read", "does-not-exist"])

    assert "Read file contents" in prompt


def test_system_prompt_collapses_multiline_snippets_to_one_line():
    tool = CodingAgentHarnessTool(
        tool=create_coding_agent_harness_tools("/workspace", ["read"])[0].tool,
        prompt_snippet="first line\n   second line\n\nthird",
    )

    prompt = build_coding_agent_harness_system_prompt("/workspace", [tool], ["read"])

    assert "first line second line third" in prompt


def test_system_prompt_passes_through_supplied_options():
    tools = create_coding_agent_harness_tools("/workspace")
    options = BuildSystemPromptOptions(cwd="/ignored", append_system_prompt="EXTRA INSTRUCTIONS")

    prompt = build_coding_agent_harness_system_prompt("/workspace", tools, ["read"], options)

    assert "EXTRA INSTRUCTIONS" in prompt
    # `cwd` always comes from the explicit argument, never the options object.
    assert "/ignored" not in prompt


@pytest.mark.asyncio
async def test_create_harness_installs_the_default_tools():
    harness, suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(session=make_session(), model=Model(id="m", provider="test"), cwd="/workspace")
    )

    assert suspended == []
    assert await harness.get_active_tools() == list(DEFAULT_HARNESS_TOOL_NAMES)
    assert [tool.name for tool in await harness.get_tools()] == list(DEFAULT_HARNESS_TOOL_NAMES)


@pytest.mark.asyncio
async def test_create_harness_honours_explicit_tools_and_active_names():
    tools = create_coding_agent_harness_tools("/workspace", ["read", "grep"])

    harness, _ = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=make_session(),
            model=Model(id="m", provider="test"),
            cwd="/workspace",
            tools=tools,
            active_tool_names=["grep"],
        )
    )

    assert await harness.get_active_tools() == ["grep"]
    assert [tool.name for tool in await harness.get_tools()] == ["read", "grep"]
