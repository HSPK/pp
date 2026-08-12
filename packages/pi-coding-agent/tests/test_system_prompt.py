"""Tests for `pi_coding_agent.core.system_prompt`, ported from
`packages/coding-agent/test/system-prompt.test.ts` and extended to cover the
custom-prompt path, project context files, skills and the guideline rules.

The "Pi documentation" paragraph of the TypeScript prompt is deliberately not
ported (see the module docstring), so the assertions from the TypeScript test
covering those absolute paths have no counterpart here.
"""

from __future__ import annotations

from pi_coding_agent.core.resource_loader import Skill, create_synthetic_source_info
from pi_coding_agent.core.system_prompt import (
    BuildSystemPromptOptions,
    ContextFile,
    build_system_prompt,
)

CWD = "/workspace/project"


def make_skill(name: str, description: str = "does a thing", disable_model_invocation: bool = False) -> Skill:
    file_path = f"/skills/{name}/SKILL.md"
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=f"/skills/{name}",
        source_info=create_synthetic_source_info(file_path, "test"),
        disable_model_invocation=disable_model_invocation,
    )


def test_shows_none_for_an_empty_tools_list():
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, selected_tools=[]))

    assert "Available tools:\n(none)" in prompt
    assert "Show file paths clearly" in prompt
    # No bash tool means no bash guideline.
    assert "Use bash for file operations" not in prompt


def test_defaults_to_the_four_core_tools_when_selection_is_absent():
    snippets = {
        "read": "Read file contents",
        "bash": "Execute bash commands",
        "edit": "Make surgical edits",
        "write": "Create or overwrite files",
        "grep": "Search file contents",
    }
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, tool_snippets=snippets))

    assert "- read: Read file contents" in prompt
    assert "- bash: Execute bash commands" in prompt
    assert "- edit: Make surgical edits" in prompt
    assert "- write: Create or overwrite files" in prompt
    # grep is not a default tool, so its snippet is not listed.
    assert "- grep:" not in prompt


# TS's `instructs models to resolve pi docs and examples under absolute base paths`
# (asserting the prompt contains "- When reading pi docs or examples, resolve
# docs/... under Additional docs and examples/... under Examples, not the current
# working directory" and "environment variables (docs/environment-variables.md)")
# has no counterpart: that whole "Pi documentation" paragraph interpolates
# `getReadmePath()`/`getDocsPath()`/`getExamplesPath()` from `config.ts`, which
# `config.py` does not port because this Python port ships no `docs/`/`examples/`
# tree for the model to read. Emitting the paragraph would give the model absolute
# paths that resolve to nothing. See `core/system_prompt.py`'s module docstring.
def test_pi_documentation_paragraph_is_present():
    """Port of `system-prompt.ts:131-138`.

    This paragraph is the whole mechanism behind pi explaining its own
    features: it names absolute paths to the shipped README/`docs/`/
    `examples/` and maps topics to filenames, and the model reads them with
    the ordinary read tool. It was previously omitted because the port shipped
    no `docs/` tree; the docs are ported now, so it is emitted again.
    See `tests/test_self_documentation.py` for the resolve-on-disk guarantees.
    """
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD))

    assert "Pi documentation" in prompt
    assert "Additional docs:" in prompt
    assert "docs/environment-variables.md" in prompt


def test_tools_are_listed_in_selection_order_and_only_with_snippets():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            selected_tools=["write", "read", "dynamic_tool"],
            tool_snippets={"read": "Read files", "write": "Write files", "other": "Unused"},
        )
    )

    tools_block = prompt.split("Available tools:\n")[1].split("\n\nIn addition")[0]
    assert tools_block == "- write: Write files\n- read: Read files"
    assert "dynamic_tool" not in prompt
    assert "Unused" not in prompt


def test_custom_tool_with_a_snippet_is_listed():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            selected_tools=["read", "dynamic_tool"],
            tool_snippets={"dynamic_tool": "Run dynamic test behavior"},
        )
    )

    assert "- dynamic_tool: Run dynamic test behavior" in prompt


def test_empty_snippet_hides_a_selected_tool():
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, selected_tools=["read"], tool_snippets={"read": ""}))

    assert "Available tools:\n(none)" in prompt


def test_bash_guideline_only_when_no_search_tool_is_available():
    with_bash_only = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, selected_tools=["bash"]))
    assert "- Use bash for file operations like ls, rg, find" in with_bash_only

    for search_tool in ("grep", "find", "ls"):
        prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, selected_tools=["bash", search_tool]))
        assert "Use bash for file operations" not in prompt


def test_appends_prompt_guidelines_after_the_bash_guideline():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            selected_tools=["bash"],
            prompt_guidelines=["Use dynamic_tool for project summaries."],
        )
    )

    guidelines = prompt.split("Guidelines:\n")[1].split("\n\nPi documentation")[0]
    assert guidelines.splitlines() == [
        "- Use bash for file operations like ls, rg, find",
        "- Use dynamic_tool for project summaries.",
        "- Be concise in your responses",
        "- Show file paths clearly when working with files",
    ]


def test_deduplicates_and_trims_prompt_guidelines():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            selected_tools=["read", "dynamic_tool"],
            prompt_guidelines=[
                "Use dynamic_tool for summaries.",
                "  Use dynamic_tool for summaries.  ",
                "   ",
                "Be concise in your responses",
            ],
        )
    )

    assert prompt.count("- Use dynamic_tool for summaries.") == 1
    assert prompt.count("- Be concise in your responses") == 1
    guidelines = prompt.split("Guidelines:\n")[1].split("\n\nPi documentation")[0]
    assert guidelines.splitlines() == [
        "- Use dynamic_tool for summaries.",
        "- Be concise in your responses",
        "- Show file paths clearly when working with files",
    ]


def test_backslashes_in_cwd_are_normalized_to_forward_slashes():
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd="C:\\Users\\dev\\project"))

    assert prompt.endswith("\nCurrent working directory: C:/Users/dev/project")


def test_append_system_prompt_lands_before_the_project_context():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            append_system_prompt="EXTRA INSTRUCTIONS",
            context_files=[ContextFile(path="AGENTS.md", content="be nice")],
        )
    )

    assert "\n\nEXTRA INSTRUCTIONS\n\n<project_context>" in prompt


def test_project_context_files_are_wrapped_per_file():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            context_files=[
                ContextFile(path="AGENTS.md", content="root rules"),
                ContextFile(path="sub/AGENTS.md", content="sub rules"),
            ],
        )
    )

    context = prompt.split("<project_context>\n\n")[1].split("</project_context>")[0]
    assert context == (
        "Project-specific instructions and guidelines:\n\n"
        '<project_instructions path="AGENTS.md">\nroot rules\n</project_instructions>\n\n'
        '<project_instructions path="sub/AGENTS.md">\nsub rules\n</project_instructions>\n\n'
    )


def test_no_project_context_block_without_context_files():
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD))

    assert "<project_context>" not in prompt


def test_skills_are_appended_only_when_the_read_tool_is_selected():
    skills = [make_skill("pdf")]

    with_read = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, selected_tools=["read", "bash"], skills=skills))
    assert "<available_skills>" in with_read
    assert "<name>pdf</name>" in with_read

    without_read = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, selected_tools=["bash"], skills=skills))
    assert "<available_skills>" not in without_read


def test_skills_section_sits_between_project_context_and_cwd():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            skills=[make_skill("pdf")],
            context_files=[ContextFile(path="AGENTS.md", content="rules")],
        )
    )

    assert prompt.index("</project_context>") < prompt.index("<available_skills>")
    assert prompt.index("<available_skills>") < prompt.index("Current working directory:")


def test_hidden_skills_produce_no_skills_section():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(cwd=CWD, skills=[make_skill("pdf", disable_model_invocation=True)])
    )

    assert "<available_skills>" not in prompt


def test_custom_prompt_replaces_the_default_body():
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, custom_prompt="You are a haiku bot."))

    assert prompt == f"You are a haiku bot.\nCurrent working directory: {CWD}"
    assert "Available tools:" not in prompt
    assert "Guidelines:" not in prompt


def test_custom_prompt_composes_append_context_and_skills_in_order():
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=CWD,
            custom_prompt="BASE",
            append_system_prompt="APPENDED",
            context_files=[ContextFile(path="AGENTS.md", content="rules")],
            skills=[make_skill("pdf")],
        )
    )

    assert prompt.startswith("BASE\n\nAPPENDED\n\n<project_context>")
    assert prompt.index("</project_context>") < prompt.index("<available_skills>")
    assert prompt.endswith(f"\nCurrent working directory: {CWD}")


def test_custom_prompt_skills_require_the_read_tool():
    skills = [make_skill("pdf")]

    default_selection = build_system_prompt(BuildSystemPromptOptions(cwd=CWD, custom_prompt="BASE", skills=skills))
    assert "<available_skills>" in default_selection

    with_read = build_system_prompt(
        BuildSystemPromptOptions(cwd=CWD, custom_prompt="BASE", selected_tools=["read"], skills=skills)
    )
    assert "<available_skills>" in with_read

    without_read = build_system_prompt(
        BuildSystemPromptOptions(cwd=CWD, custom_prompt="BASE", selected_tools=["bash"], skills=skills)
    )
    assert "<available_skills>" not in without_read

    no_tools = build_system_prompt(
        BuildSystemPromptOptions(cwd=CWD, custom_prompt="BASE", selected_tools=[], skills=skills)
    )
    assert "<available_skills>" not in no_tools
