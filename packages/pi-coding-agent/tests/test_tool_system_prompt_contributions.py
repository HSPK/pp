"""Python port of `packages/coding-agent/test/tool-system-prompt-contributions.test.ts`.

The TypeScript test checks that each built-in tool's `ToolDefinition` carries
the same `promptSnippet`/`promptGuidelines` as its exported
`*SystemPromptContribution` constant. This port has no per-tool
`ToolDefinition` (`tools/__init__.py` only builds runnable `AgentTool`s), so
the single source of truth is `TOOL_PROMPT_CONTRIBUTIONS` in
`core/agent_session.py`. The test pins it against the verbatim TypeScript
strings, which is what the TypeScript assertion protects.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.core.agent_session import TOOL_PROMPT_CONTRIBUTIONS
from pi_coding_agent.tools.bash import BashAgentTool, create_bash_tool
from test_agent_session import build_session

_TS_CONTRIBUTIONS: dict[str, tuple[str, list[str]]] = {
    "read": ("Read file contents", ["Use read to examine files instead of cat or sed."]),
    "bash": (
        "Execute bash commands (ls, grep, find, etc.)",
        ["You can inspect PI_* environment variables for current model and session details."],
    ),
    "edit": (
        "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        [
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in "
            "edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. "
            "Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with "
            "large unchanged regions.",
        ],
    ),
    "write": ("Create or overwrite files", ["Use write only for new files or complete rewrites."]),
    "grep": ("Search file contents for patterns (respects .gitignore)", []),
    "find": ("Find files by glob pattern (respects .gitignore)", []),
    "ls": ("List directory contents", []),
}


@pytest.mark.parametrize("name", sorted(_TS_CONTRIBUTIONS))
def test_keeps_the_tool_contribution_aligned_with_the_typescript_constant(name: str):
    expected_snippet, expected_guidelines = _TS_CONTRIBUTIONS[name]
    snippet, guidelines = TOOL_PROMPT_CONTRIBUTIONS[name]

    assert snippet == expected_snippet
    assert guidelines == expected_guidelines


def test_contributions_cover_exactly_the_builtin_tools():
    assert set(TOOL_PROMPT_CONTRIBUTIONS) == set(_TS_CONTRIBUTIONS)


def test_keeps_bash_session_environment_guidance_conditional():
    """TS: `createBashToolDefinition("/workspace", {exposeSessionEnvironment: false})`
    has `promptGuidelines === undefined`. This port has no `ToolDefinition`, so the
    equivalent state is the tool's own empty `prompt_guidelines`, which
    `AgentSession._refresh_tool_registry` prefers over
    `TOOL_PROMPT_CONTRIBUTIONS["bash"]`.
    """
    default_tool = create_bash_tool("/workspace")
    assert isinstance(default_tool, BashAgentTool)
    assert default_tool.prompt_guidelines == list(_TS_CONTRIBUTIONS["bash"][1])

    tool = create_bash_tool("/workspace", expose_session_environment=False)
    assert isinstance(tool, BashAgentTool)
    assert tool.prompt_guidelines == []


async def test_session_omits_bash_guideline_when_session_environment_is_hidden(tmp_path):
    """The consequence TS's `promptGuidelines: undefined` has on the system prompt:
    a session whose bash tool hides the `PI_*` variables must not tell the model to
    inspect them.
    """
    guideline = _TS_CONTRIBUTIONS["bash"][1][0]

    exposed, _sm, _settings, _stream = await build_session(
        tmp_path / "exposed", custom_tools={"bash": create_bash_tool(str(tmp_path))}
    )
    assert guideline in exposed.system_prompt

    hidden, _sm2, _settings2, _stream2 = await build_session(
        tmp_path / "hidden",
        custom_tools={"bash": create_bash_tool(str(tmp_path), expose_session_environment=False)},
    )
    assert guideline not in hidden.system_prompt
    # The snippet is keyed by tool name and is unconditional in TypeScript too.
    assert _TS_CONTRIBUTIONS["bash"][0] in hidden.system_prompt
