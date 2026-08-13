"""Port of `packages/coding-agent/test/experimental-tool-strict-mode.test.ts`."""

from __future__ import annotations

import os

import pytest
from pi_agent.types import AgentTool
from pi_ai.types import JsonSchemaConstrainedSampling
from pi_coding_agent.server.create_harness import create_coding_agent_harness_tools
from pi_coding_agent.tools import create_tool

BUILT_IN_TOOL_NAMES = ("read", "bash", "edit", "write")


def _built_in_tools() -> list[AgentTool]:
    return [create_tool(name, os.getcwd()) for name in BUILT_IN_TOOL_NAMES]


def test_only_enables_strict_prefer_sampling_in_experimental_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_EXPERIMENTAL", raising=False)
    normal_tools = _built_in_tools()
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    experimental_tools = _built_in_tools()

    for index, tool in enumerate(experimental_tools):
        assert tool.constrained_sampling == JsonSchemaConstrainedSampling(strict="prefer")
        # Strict sampling must not change the schema the model is shown.
        assert tool.parameters == normal_tools[index].parameters
        assert normal_tools[index].constrained_sampling is None


def test_each_tool_gets_an_independent_sampling_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    tools = _built_in_tools()
    first = tools[0].constrained_sampling
    assert first is not None
    first.strict = "require"
    assert tools[1].constrained_sampling == JsonSchemaConstrainedSampling(strict="prefer")


def test_harness_tools_carry_strict_sampling_in_experimental_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_EXPERIMENTAL", raising=False)
    normal = create_coding_agent_harness_tools(os.getcwd(), BUILT_IN_TOOL_NAMES)
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    experimental = create_coding_agent_harness_tools(os.getcwd(), BUILT_IN_TOOL_NAMES)

    assert all(harness_tool.tool.constrained_sampling is None for harness_tool in normal)
    assert all(
        harness_tool.tool.constrained_sampling == JsonSchemaConstrainedSampling(strict="prefer")
        for harness_tool in experimental
    )
