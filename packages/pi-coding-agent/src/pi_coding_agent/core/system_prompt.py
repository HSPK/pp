"""System prompt construction and project context loading.

Port of `packages/coding-agent/src/core/system-prompt.ts`.

Includes the "Pi documentation" paragraph (`_docs_section` below), which is
the entire mechanism behind pi being able to explain its own features: the
prompt names absolute paths to this package's `README.md`, `docs/` and
`examples/`, and the model opens them with the ordinary read tool. There is
no retrieval index or embedded copy of the manual.

That paragraph was previously omitted, because the Python port shipped no
`docs/` tree and emitting unresolvable paths is worse than saying nothing --
it invites the model to report a read failure as a broken install. The docs
are now ported, so the paragraph is emitted, but still only when
`get_docs_path()` actually exists on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pi_coding_agent.core.config import get_docs_path, get_examples_path, get_readme_path
from pi_coding_agent.core.resource_loader import Skill, format_skills_for_prompt

DEFAULT_SELECTED_TOOLS = ("read", "bash", "edit", "write")


@dataclass
class ContextFile:
    path: str
    content: str


@dataclass
class BuildSystemPromptOptions:
    cwd: str
    custom_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)


def _append_project_context(prompt: str, context_files: list[ContextFile]) -> str:
    if not context_files:
        return prompt
    prompt += "\n\n<project_context>\n\n"
    prompt += "Project-specific instructions and guidelines:\n\n"
    for context_file in context_files:
        prompt += (
            f'<project_instructions path="{context_file.path}">\n{context_file.content}\n</project_instructions>\n\n'
        )
    prompt += "</project_context>\n"
    return prompt


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """Build the system prompt with tools, guidelines, and context."""
    prompt_cwd = options.cwd.replace("\\", "/")
    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""
    context_files = options.context_files
    skills = options.skills

    if options.custom_prompt:
        prompt = options.custom_prompt
        if append_section:
            prompt += append_section

        prompt = _append_project_context(prompt, context_files)

        custom_prompt_has_read = options.selected_tools is None or "read" in options.selected_tools
        if custom_prompt_has_read and skills:
            prompt += format_skills_for_prompt(skills)

        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    # An explicit empty list means "no tools", not "use the defaults": the
    # TypeScript `selectedTools || [...]` only falls back when the field is
    # absent, and `[]` is truthy in JavaScript.
    tools = list(DEFAULT_SELECTED_TOOLS) if options.selected_tools is None else list(options.selected_tools)
    tool_snippets = options.tool_snippets or {}
    visible_tools = [name for name in tools if tool_snippets.get(name)]
    tools_list = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible_tools) if visible_tools else "(none)"

    guidelines_list: list[str] = []
    guidelines_set: set[str] = set()

    def add_guideline(guideline: str) -> None:
        if guideline in guidelines_set:
            return
        guidelines_set.add(guideline)
        guidelines_list.append(guideline)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")

    for guideline in options.prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    prompt = f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{guidelines}"""

    prompt += _docs_section()

    if append_section:
        prompt += append_section

    prompt = _append_project_context(prompt, context_files)

    if has_read and skills:
        prompt += format_skills_for_prompt(skills)

    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


def _docs_section() -> str:
    """The "Pi documentation" paragraph, port of `system-prompt.ts:131-138`.

    This is the whole mechanism behind pi explaining its own features: there is
    no retrieval index, just absolute paths to real files that the model opens
    with the ordinary read tool. The topic->file map matters as much as the
    paths -- without it the model guesses filenames.

    Emitted only when the docs actually exist on disk. A path that resolves to
    nothing is worse than silence: it invites the model to report a read
    failure as if the documentation were missing from the user's install.
    """
    readme_path = get_readme_path()
    docs_path = get_docs_path()
    examples_path = get_examples_path()
    if not os.path.isdir(docs_path):
        return ""

    lines = [
        "",
        "",
        "Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):",
        f"- Main documentation: {readme_path}",
        f"- Additional docs: {docs_path}",
    ]
    if os.path.isdir(examples_path):
        lines.append(f"- Examples: {examples_path} (extensions, custom tools, SDK)")
    lines.extend(
        [
            "- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory",
            "- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)",
            "- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing",
            "- Always read pi .md files completely and follow links to related docs (e.g. tui.md for TUI API details)",
        ]
    )
    return "\n".join(lines)
