"""Discovery and loading of AGENTS.md context, skills, and prompt templates.

Python port of the project/user-directory discovery logic in
`packages/coding-agent/src/core/resource-loader.ts`, plus the self-contained
helpers it composes from `core/skills.ts`, `core/prompt-templates.ts`,
`core/source-info.ts`, and `core/diagnostics.ts` (`ResourceCollision`/
`ResourceDiagnostic` below are a direct port of that file's two interfaces).

**Scope deviation.** The upstream `DefaultResourceLoader` also loads themes
(`modes/interactive/theme/`), which have no equivalent anywhere else in this
Python port yet (there is no TUI theme system here). Porting that would mean
inventing a whole unported subsystem rather than porting existing behavior,
so this module ports only the resource kinds the task calls for: project/user
AGENTS.md context files, skills, and prompt templates (which double as slash
commands, expanded via `expand_prompt_template`). `ResourceLoader` mirrors
`DefaultResourceLoader`'s precedence rules (project overrides user on a name
collision) for just these resource kinds.

Extensions (`core/extensions/`) and npm/git package sources
(`core/package-manager.ts`) are *not* out of scope: `extensions/loader.py`
ports the extension runtime, and `core/package_manager.py` ports the
installer (git and local-path sources; npm sources are unsupported -- see
that module's docstring). This module's own `extra_paths` parameter is how
callers plug in `PackageManager.resolve()`'s auto-discovered
project/user-scope resource paths, mirroring `addAutoDiscoveredResources`'s
call into `DefaultResourceLoader`.

**Auto-discovery precedence.** `DefaultResourceLoader.reload()` builds its
skill/prompt path list as project-default-dir, then user-default-dir, then
explicit extra paths (`packageManager.resolve()` adds project resources
before user resources when the project is trusted; see
`package-manager.ts`'s `addAutoDiscoveredResources`). `ResourceLoader.reload()`
reproduces that exact order so first-wins name-collision merging (in
`load_skills` / the local prompt dedupe) matches: project beats user, and
both beat explicit extra paths.

**Late-arriving resources.** `extend_resources()` ports `extendResources`:
extensions handling `resources_discover` can add skill and prompt paths after
`reload()` has already run. Discovery re-runs over the original paths plus the
new ones, so the collision precedence above still holds. Its `themePaths`
parameter is absent, per the scope deviation above.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Literal

import yaml

from pi_coding_agent.core.config import CONFIG_DIR_NAME
from pi_coding_agent.core.timings import reset_timings
from pi_coding_agent.tools.gitignore import GitignoreMatcher
from pi_coding_agent.utils.paths import PathInputOptions, canonicalize_path, resolve_path

# --------------------------------------------------------------------------
# Diagnostics / source info (ports of core/diagnostics.ts, core/source-info.ts)
# --------------------------------------------------------------------------

SourceScope = Literal["user", "project", "temporary"]
SourceOrigin = Literal["package", "top-level"]
ResourceType = Literal["skill", "prompt"]
DiagnosticType = Literal["warning", "error", "collision"]


@dataclass
class SourceInfo:
    path: str
    source: str
    scope: SourceScope = "temporary"
    origin: SourceOrigin = "top-level"
    base_dir: str | None = None


def create_synthetic_source_info(
    path: str,
    source: str,
    scope: SourceScope = "temporary",
    origin: SourceOrigin = "top-level",
    base_dir: str | None = None,
) -> SourceInfo:
    return SourceInfo(path=path, source=source, scope=scope, origin=origin, base_dir=base_dir)


@dataclass
class ResourceCollision:
    """Port of `ResourceCollision` from `packages/coding-agent/src/core/diagnostics.ts`."""

    resource_type: ResourceType
    name: str
    winner_path: str
    loser_path: str


@dataclass
class ResourceDiagnostic:
    """Port of `ResourceDiagnostic` from `packages/coding-agent/src/core/diagnostics.ts`."""

    type: DiagnosticType
    message: str
    path: str | None = None
    collision: ResourceCollision | None = None


# --------------------------------------------------------------------------
# Frontmatter (port of utils/frontmatter.ts)
# --------------------------------------------------------------------------


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split a leading ``---\\n...\\n---`` YAML block from the rest of the content."""
    normalized = _normalize_newlines(content)
    if not normalized.startswith("---"):
        return {}, normalized

    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized

    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    # The trailing newline before the closing `---` is not part of the slice.
    # The `yaml` npm package still terminates a block scalar with it, so append
    # it here or `description: |` would lose its final newline relative to TS.
    parsed = yaml.safe_load(yaml_string + "\n")
    return (parsed or {}), body


def strip_frontmatter(content: str) -> str:
    return parse_frontmatter(content)[1]


# --------------------------------------------------------------------------
# Skills (port of core/skills.ts)
# --------------------------------------------------------------------------

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
_IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")
_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]+$")


@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source_info: SourceInfo
    disable_model_invocation: bool = False


@dataclass
class LoadSkillsResult:
    skills: list[Skill] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


def _validate_name(name: str) -> list[str]:
    errors: list[str] = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not _SKILL_NAME_RE.match(name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if not description or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


def _create_skill_source_info(file_path: str, base_dir: str, source: str) -> SourceInfo:
    if source == "user":
        return create_synthetic_source_info(file_path, "local", scope="user", base_dir=base_dir)
    if source == "project":
        return create_synthetic_source_info(file_path, "local", scope="project", base_dir=base_dir)
    if source == "path":
        return create_synthetic_source_info(file_path, "local", base_dir=base_dir)
    return create_synthetic_source_info(file_path, source, base_dir=base_dir)


def _add_ignore_rules(matcher: GitignoreMatcher, directory: str, root_dir: str) -> None:
    rel_dir = os.path.relpath(directory, root_dir).replace(os.sep, "/")
    prefix = f"{rel_dir}/" if rel_dir != "." else ""
    for filename in _IGNORE_FILE_NAMES:
        ignore_path = os.path.join(directory, filename)
        if not os.path.exists(ignore_path):
            continue
        try:
            with open(ignore_path, encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue
        lines = []
        for raw_line in content.splitlines():
            trimmed = raw_line.strip()
            if not trimmed or (trimmed.startswith("#") and not trimmed.startswith("\\#")):
                continue
            negated = raw_line.startswith("!")
            body = raw_line[1:] if negated else raw_line
            if body.startswith("\\!"):
                body = body[1:]
            if body.startswith("/"):
                body = body[1:]
            prefixed = f"{prefix}{body}" if prefix else body
            lines.append(f"!{prefixed}" if negated else prefixed)
        if lines:
            matcher.add(lines)


def load_skills_from_dir(directory: str, source: str) -> LoadSkillsResult:
    """Load skills from a directory.

    Discovery rules:
    - if a directory contains SKILL.md, treat it as a skill root and do not recurse further
    - otherwise, load direct .md children in the root
    - recurse into subdirectories to find SKILL.md
    """
    return _load_skills_from_dir_internal(directory, source, True)


def _load_skills_from_dir_internal(
    directory: str,
    source: str,
    include_root_files: bool,
    matcher: GitignoreMatcher | None = None,
    root_dir: str | None = None,
) -> LoadSkillsResult:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []

    if not os.path.exists(directory):
        return LoadSkillsResult(skills, diagnostics)

    root = root_dir or directory
    matcher = matcher if matcher is not None else GitignoreMatcher()
    _add_ignore_rules(matcher, directory, root)

    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return LoadSkillsResult(skills, diagnostics)

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        full_path = os.path.join(directory, entry.name)
        try:
            is_file = os.path.isfile(full_path)
        except OSError:
            continue
        rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
        if not is_file or matcher.is_ignored(rel_path, False):
            continue
        skill, skill_diagnostics = _load_skill_from_file(full_path, source)
        if skill:
            skills.append(skill)
        diagnostics.extend(skill_diagnostics)
        return LoadSkillsResult(skills, diagnostics)

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        full_path = os.path.join(directory, entry.name)
        try:
            is_dir = entry.is_dir(follow_symlinks=True)
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue

        rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if matcher.is_ignored(ignore_path, is_dir):
            continue

        if is_dir:
            sub_result = _load_skills_from_dir_internal(full_path, source, False, matcher, root)
            skills.extend(sub_result.skills)
            diagnostics.extend(sub_result.diagnostics)
            continue

        if not is_file or not include_root_files or not entry.name.endswith(".md"):
            continue

        skill, skill_diagnostics = _load_skill_from_file(full_path, source)
        if skill:
            skills.append(skill)
        diagnostics.extend(skill_diagnostics)

    return LoadSkillsResult(skills, diagnostics)


def _load_skill_from_file(file_path: str, source: str) -> tuple[Skill | None, list[ResourceDiagnostic]]:
    diagnostics: list[ResourceDiagnostic] = []
    try:
        with open(file_path, encoding="utf-8") as fh:
            raw_content = fh.read()
        frontmatter, _body = parse_frontmatter(raw_content)
        skill_dir = os.path.dirname(file_path)
        parent_dir_name = os.path.basename(skill_dir)

        description = frontmatter.get("description")
        for error in _validate_description(description):
            diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

        name = frontmatter.get("name") or parent_dir_name
        for error in _validate_name(name):
            diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

        if not description or (isinstance(description, str) and description.strip() == ""):
            return None, diagnostics

        return (
            Skill(
                name=name,
                description=description,
                file_path=file_path,
                base_dir=skill_dir,
                source_info=_create_skill_source_info(file_path, skill_dir, source),
                disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
            ),
            diagnostics,
        )
    except Exception as error:
        diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=file_path))
        return None, diagnostics


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format skills for inclusion in a system prompt (Agent Skills XML format)."""
    visible_skills = [s for s in skills if not s.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _is_under_path(target: str, root: str) -> bool:
    normalized_root = os.path.abspath(root)
    if target == normalized_root:
        return True
    prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
    return target.startswith(prefix)


def load_skills(
    *,
    cwd: str,
    agent_dir: str,
    skill_paths: list[str],
    include_defaults: bool,
) -> LoadSkillsResult:
    """Load skills from all configured locations.

    When ``include_defaults`` is set, ``agent_dir/skills`` (source "user") is
    loaded before ``cwd/<CONFIG_DIR_NAME>/skills`` (source "project"), so a
    user skill wins a name collision -- this matches `loadSkills`'s own
    default ordering in `skills.ts`. `ResourceLoader` below does not use
    ``include_defaults``; it builds its own project-then-user path list to
    match `DefaultResourceLoader`'s collision precedence instead (see the
    module docstring).
    """
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    skill_map: dict[str, Skill] = {}
    real_path_set: set[str] = set()
    all_diagnostics: list[ResourceDiagnostic] = []
    collision_diagnostics: list[ResourceDiagnostic] = []

    def add_skills(result: LoadSkillsResult) -> None:
        all_diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            real_path = os.path.realpath(skill.file_path)
            if real_path in real_path_set:
                continue
            existing = skill_map.get(skill.name)
            if existing:
                collision_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        collision=ResourceCollision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=existing.file_path,
                            loser_path=skill.file_path,
                        ),
                    )
                )
            else:
                skill_map[skill.name] = skill
                real_path_set.add(real_path)

    user_skills_dir = os.path.join(resolved_agent_dir, "skills")
    project_skills_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "skills")

    if include_defaults:
        add_skills(_load_skills_from_dir_internal(user_skills_dir, "user", True))
        add_skills(_load_skills_from_dir_internal(project_skills_dir, "project", True))

    def get_source(resolved_path: str) -> str:
        if not include_defaults:
            if _is_under_path(resolved_path, user_skills_dir):
                return "user"
            if _is_under_path(resolved_path, project_skills_dir):
                return "project"
        return "path"

    for raw_path in skill_paths:
        resolved_path = resolve_path(raw_path.strip(), resolved_cwd)
        if not os.path.exists(resolved_path):
            all_diagnostics.append(
                ResourceDiagnostic(type="warning", message="skill path does not exist", path=resolved_path)
            )
            continue
        try:
            source = get_source(resolved_path)
            if os.path.isdir(resolved_path):
                add_skills(_load_skills_from_dir_internal(resolved_path, source, True))
            elif os.path.isfile(resolved_path) and resolved_path.endswith(".md"):
                skill, skill_diagnostics = _load_skill_from_file(resolved_path, source)
                if skill:
                    add_skills(LoadSkillsResult([skill], skill_diagnostics))
                else:
                    all_diagnostics.extend(skill_diagnostics)
            else:
                all_diagnostics.append(
                    ResourceDiagnostic(type="warning", message="skill path is not a markdown file", path=resolved_path)
                )
        except OSError as error:
            all_diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=resolved_path))

    return LoadSkillsResult(list(skill_map.values()), [*all_diagnostics, *collision_diagnostics])


# --------------------------------------------------------------------------
# Prompt templates (port of core/prompt-templates.ts)
# --------------------------------------------------------------------------

_SUBSTITUTE_ARGS_RE = re.compile(r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)")


@dataclass
class PromptTemplate:
    name: str
    description: str
    content: str
    source_info: SourceInfo
    file_path: str
    argument_hint: str | None = None


def parse_command_args(args_string: str) -> list[str]:
    """Parse command arguments respecting quoted strings (bash-style)."""
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char.isspace():
            if current:
                args.append(current)
                current = ""
        else:
            current += char

    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    """Substitute `$1`, `$@`/`$ARGUMENTS`, `${N:-default}`, `${@:N[:L]}` placeholders."""
    all_args = " ".join(args)

    def replace(match: re.Match[str]) -> str:
        default_target, default_value, slice_start, slice_length, simple = match.groups()

        if default_target:
            value = all_args if default_target in ("@", "ARGUMENTS") else _arg_at(args, int(default_target))
            return value if value else (default_value or "")

        if slice_start:
            start = max(int(slice_start) - 1, 0)
            if slice_length:
                return " ".join(args[start : start + int(slice_length)])
            return " ".join(args[start:])

        if simple in ("ARGUMENTS", "@"):
            return all_args

        return _arg_at(args, int(simple)) or ""

    return _SUBSTITUTE_ARGS_RE.sub(replace, content)


def _arg_at(args: list[str], one_indexed: int) -> str | None:
    index = one_indexed - 1
    return args[index] if 0 <= index < len(args) else None


def _load_template_from_file(file_path: str, source_info: SourceInfo) -> PromptTemplate | None:
    try:
        with open(file_path, encoding="utf-8") as fh:
            raw_content = fh.read()
        frontmatter, body = parse_frontmatter(raw_content)

        name = os.path.basename(file_path)
        if name.endswith(".md"):
            name = name[: -len(".md")]

        description = frontmatter.get("description") or ""
        if not description:
            first_line = next((line for line in body.split("\n") if line.strip()), None)
            if first_line:
                description = first_line[:60] + ("..." if len(first_line) > 60 else "")

        argument_hint = frontmatter.get("argument-hint")
        return PromptTemplate(
            name=name,
            description=description,
            argument_hint=argument_hint or None,
            content=body,
            source_info=source_info,
            file_path=file_path,
        )
    except OSError:
        return None


def _load_templates_from_dir(directory: str, get_source_info) -> list[PromptTemplate]:
    templates: list[PromptTemplate] = []
    if not os.path.exists(directory):
        return templates
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return templates

    for entry in entries:
        full_path = os.path.join(directory, entry.name)
        try:
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue
        if is_file and entry.name.endswith(".md"):
            template = _load_template_from_file(full_path, get_source_info(full_path))
            if template:
                templates.append(template)
    return templates


def load_prompt_templates(
    *,
    cwd: str,
    agent_dir: str,
    prompt_paths: list[str],
    include_defaults: bool,
) -> list[PromptTemplate]:
    """Load prompt templates from ``agent_dir/prompts``, ``cwd/<CONFIG_DIR_NAME>/prompts``, and explicit paths."""
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    templates: list[PromptTemplate] = []
    global_prompts_dir = os.path.join(resolved_agent_dir, "prompts")
    project_prompts_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "prompts")

    def get_source_info(resolved_path: str) -> SourceInfo:
        if _is_under_path(resolved_path, global_prompts_dir):
            return create_synthetic_source_info(resolved_path, "local", scope="user", base_dir=global_prompts_dir)
        if _is_under_path(resolved_path, project_prompts_dir):
            return create_synthetic_source_info(resolved_path, "local", scope="project", base_dir=project_prompts_dir)
        base_dir = resolved_path if os.path.isdir(resolved_path) else os.path.dirname(resolved_path)
        return create_synthetic_source_info(resolved_path, "local", base_dir=base_dir)

    if include_defaults:
        templates.extend(_load_templates_from_dir(global_prompts_dir, get_source_info))
        templates.extend(_load_templates_from_dir(project_prompts_dir, get_source_info))

    for raw_path in prompt_paths:
        resolved_path = resolve_path(raw_path.strip(), resolved_cwd)
        if not os.path.exists(resolved_path):
            continue
        try:
            if os.path.isdir(resolved_path):
                templates.extend(_load_templates_from_dir(resolved_path, get_source_info))
            elif os.path.isfile(resolved_path) and resolved_path.endswith(".md"):
                template = _load_template_from_file(resolved_path, get_source_info(resolved_path))
                if template:
                    templates.append(template)
        except OSError:
            continue

    return templates


def expand_prompt_template(text: str, templates: list[PromptTemplate]) -> str:
    """Expand a `/name args...` slash command into its template content, else return `text` unchanged."""
    if not text.startswith("/"):
        return text
    match = re.match(r"^/(\S+)(?:\s+([\s\S]*))?$", text)
    if not match:
        return text
    template_name = match.group(1)
    args_string = match.group(2) or ""
    template = next((t for t in templates if t.name == template_name), None)
    if template:
        args = parse_command_args(args_string)
        return substitute_args(template.content, args)
    return text


def _dedupe_prompts(prompts: list[PromptTemplate]) -> tuple[list[PromptTemplate], list[ResourceDiagnostic]]:
    seen: dict[str, PromptTemplate] = {}
    diagnostics: list[ResourceDiagnostic] = []
    for prompt in prompts:
        existing = seen.get(prompt.name)
        if existing:
            diagnostics.append(
                ResourceDiagnostic(
                    type="collision",
                    message=f'name "/{prompt.name}" collision',
                    path=prompt.file_path,
                    collision=ResourceCollision(
                        resource_type="prompt",
                        name=prompt.name,
                        winner_path=existing.file_path,
                        loser_path=prompt.file_path,
                    ),
                )
            )
        else:
            seen[prompt.name] = prompt
    return list(seen.values()), diagnostics


# --------------------------------------------------------------------------
# AGENTS.md / CLAUDE.md context file discovery (port of the relevant parts of
# core/resource-loader.ts and the git-worktree helper from core/footer-data-provider.ts)
# --------------------------------------------------------------------------

_CONTEXT_FILE_CANDIDATES = ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


@dataclass
class GitPaths:
    repo_dir: str
    common_git_dir: str
    head_path: str


def find_git_paths(cwd: str) -> GitPaths | None:
    """Find git metadata paths by walking up from ``cwd``.

    Handles both regular git repos (``.git`` is a directory) and worktrees
    (``.git`` is a file pointing at ``gitdir: <path>``).
    """
    directory = cwd
    while True:
        git_path = os.path.join(directory, ".git")
        if os.path.exists(git_path):
            try:
                if os.path.isfile(git_path):
                    with open(git_path, encoding="utf-8") as fh:
                        content = fh.read().strip()
                    if content.startswith("gitdir: "):
                        git_dir = os.path.normpath(os.path.join(directory, content[8:].strip()))
                        head_path = os.path.join(git_dir, "HEAD")
                        if not os.path.exists(head_path):
                            return None
                        common_dir_path = os.path.join(git_dir, "commondir")
                        if os.path.exists(common_dir_path):
                            with open(common_dir_path, encoding="utf-8") as fh:
                                common_git_dir = os.path.normpath(os.path.join(git_dir, fh.read().strip()))
                        else:
                            common_git_dir = git_dir
                        return GitPaths(repo_dir=directory, common_git_dir=common_git_dir, head_path=head_path)
                elif os.path.isdir(git_path):
                    head_path = os.path.join(git_path, "HEAD")
                    if not os.path.exists(head_path):
                        return None
                    return GitPaths(repo_dir=directory, common_git_dir=git_path, head_path=head_path)
            except OSError:
                return None
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def _canonicalize_path(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def _load_context_file_from_dir(directory: str) -> tuple[str, str] | None:
    """Return ``(path, content)`` for the first matching context file, else ``None``."""
    for filename in _CONTEXT_FILE_CANDIDATES:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, encoding="utf-8") as fh:
                    return file_path, fh.read()
            except OSError as error:
                print(f"Warning: Could not read {file_path}: {error}")
    return None


def _find_shadowed_context_file(cwd: str) -> str | None:
    """The main repo's context file that a linked worktree's own copy shadows.

    Both occupy the same logical repository scope, so loading both would
    apply that context twice. Returns ``None`` when nothing is shadowed,
    leaving normal ancestor inheritance alone.
    """
    git_paths = find_git_paths(cwd)
    if git_paths is None:
        return None
    common_git_dir = _canonicalize_path(git_paths.common_git_dir)
    worktree_root = _canonicalize_path(git_paths.repo_dir)
    main_repo_root = os.path.dirname(common_git_dir)
    if not worktree_root.startswith(f"{main_repo_root}{os.sep}"):
        return None
    if _canonicalize_path(os.path.join(main_repo_root, ".git")) != common_git_dir:
        return None
    worktree_context_file = _load_context_file_from_dir(worktree_root)
    if worktree_context_file is None:
        return None
    return os.path.join(main_repo_root, os.path.basename(worktree_context_file[0]))


def load_project_context_files(cwd: str, agent_dir: str) -> list[dict[str, str]]:
    """Discover AGENTS.md/CLAUDE.md context files: global, then ancestors-to-cwd.

    Returns a list of ``{"path": ..., "content": ...}`` in the order the
    context should be assembled: the global file first, then each directory
    from the repository root down to ``cwd``.
    """
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    context_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context is not None:
        context_files.append({"path": global_context[0], "content": global_context[1]})
        seen_paths.add(global_context[0])

    ancestor_context_files: list[dict[str, str]] = []
    shadowed_context_file = _find_shadowed_context_file(resolved_cwd)
    current_dir = resolved_cwd

    while True:
        context_file = _load_context_file_from_dir(current_dir)
        is_shadowed = shadowed_context_file is not None and context_file is not None
        if is_shadowed:
            is_shadowed = _canonicalize_path(context_file[0]) == shadowed_context_file
        if context_file is not None and not is_shadowed and context_file[0] not in seen_paths:
            ancestor_context_files.insert(0, {"path": context_file[0], "content": context_file[1]})
            seen_paths.add(context_file[0])

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    context_files.extend(ancestor_context_files)
    return context_files


# --------------------------------------------------------------------------
# SYSTEM.md / APPEND_SYSTEM.md discovery
# --------------------------------------------------------------------------


def resolve_prompt_input(input_value: str | None, description: str) -> str | None:
    """Read ``input_value`` as a file path if it exists, else treat it as literal text."""
    if not input_value:
        return None
    if os.path.exists(input_value):
        try:
            with open(input_value, encoding="utf-8") as fh:
                return fh.read()
        except OSError as error:
            print(f"Warning: Could not read {description} file {input_value}: {error}")
            return input_value
    return input_value


def discover_system_prompt_file(cwd: str, agent_dir: str, project_trusted: bool) -> str | None:
    project_path = os.path.join(cwd, CONFIG_DIR_NAME, "SYSTEM.md")
    if project_trusted and os.path.exists(project_path):
        return project_path
    global_path = os.path.join(agent_dir, "SYSTEM.md")
    if os.path.exists(global_path):
        return global_path
    return None


def discover_append_system_prompt_file(cwd: str, agent_dir: str, project_trusted: bool) -> str | None:
    project_path = os.path.join(cwd, CONFIG_DIR_NAME, "APPEND_SYSTEM.md")
    if project_trusted and os.path.exists(project_path):
        return project_path
    global_path = os.path.join(agent_dir, "APPEND_SYSTEM.md")
    if os.path.exists(global_path):
        return global_path
    return None


# --------------------------------------------------------------------------
# ResourceLoader: composes the pieces above with the precedence rules
# `DefaultResourceLoader.reload()` applies for skills/prompts/context/system prompt.
# --------------------------------------------------------------------------


@dataclass
class ResourcePathMetadata:
    """Where an extension-contributed resource path came from.

    Port of TS `PathMetadata`: `extendResources` stamps this onto every
    resource loaded from the path, so `/skills` and `/commands` can say which
    extension supplied an entry instead of showing it as a bare local file.
    """

    source: str
    scope: SourceScope = "temporary"
    origin: SourceOrigin = "top-level"
    base_dir: str | None = None


@dataclass
class ExtensionResourcePath:
    """One path an extension contributed, with its metadata. Port of TS `ResourceExtensionPaths` entries."""

    path: str
    metadata: ResourcePathMetadata


@dataclass
class ResourceLoaderOptions:
    cwd: str
    agent_dir: str
    project_trusted: bool = True
    additional_skill_paths: list[str] = field(default_factory=list)
    additional_prompt_template_paths: list[str] = field(default_factory=list)
    no_skills: bool = False
    no_prompt_templates: bool = False
    no_context_files: bool = False
    system_prompt: str | None = None
    append_system_prompt: list[str] | None = None


class ResourceLoader:
    """Discovers skills, prompt templates, AGENTS.md context, and the system prompt.

    Project resources win name collisions over user resources, and both win
    over ``additional_*_paths`` -- see the module docstring for why (mirrors
    `DefaultResourceLoader`'s actual path ordering, not `skills.ts`'s own
    `loadSkills(includeDefaults=True)` convenience default).
    """

    def __init__(self, options: ResourceLoaderOptions) -> None:
        self._cwd = resolve_path(options.cwd)
        self._agent_dir = resolve_path(options.agent_dir)
        self._project_trusted = options.project_trusted
        self._additional_skill_paths = list(options.additional_skill_paths)
        self._additional_prompt_template_paths = list(options.additional_prompt_template_paths)
        self._no_skills = options.no_skills
        self._no_prompt_templates = options.no_prompt_templates
        self._no_context_files = options.no_context_files
        self._system_prompt_source = options.system_prompt
        self._append_system_prompt_source = options.append_system_prompt

        self._skills: list[Skill] = []
        self._skill_diagnostics: list[ResourceDiagnostic] = []
        self._prompts: list[PromptTemplate] = []
        self._prompt_diagnostics: list[ResourceDiagnostic] = []
        self._agents_files: list[dict[str, str]] = []
        self._system_prompt: str | None = None
        self._system_prompt_source_path: str | None = None
        self._append_system_prompt: list[str] = []
        self._append_system_prompt_source_paths: list[str] = []
        # Retained so `extend_resources` can re-run discovery over the original
        # paths plus the extension-contributed ones, as TypeScript's
        # `lastSkillPaths`/`lastPromptPaths` do.
        self._last_skill_paths: list[str] = []
        self._last_prompt_paths: list[str] = []
        self._extension_skill_source_infos: dict[str, SourceInfo] = {}
        self._extension_prompt_source_infos: dict[str, SourceInfo] = {}

    def get_skills(self) -> LoadSkillsResult:
        return LoadSkillsResult(self._skills, self._skill_diagnostics)

    def get_prompts(self) -> tuple[list[PromptTemplate], list[ResourceDiagnostic]]:
        return self._prompts, self._prompt_diagnostics

    def get_agents_files(self) -> list[dict[str, str]]:
        return self._agents_files

    def get_system_prompt(self) -> str | None:
        return self._system_prompt

    def get_system_prompt_source(self) -> str | None:
        return self._system_prompt_source_path

    def get_append_system_prompt(self) -> list[str]:
        return self._append_system_prompt

    def get_append_system_prompt_sources(self) -> list[str]:
        return self._append_system_prompt_source_paths

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    def set_project_trusted(self, trusted: bool) -> None:
        self._project_trusted = trusted

    def reload(self) -> None:
        reset_timings("extensions")
        # Project-local dirs (.pi/skills, .pi/prompts) require project trust, matching
        # package-manager.ts's addAutoDiscoveredResources: it only calls addResources
        # for the project skills/prompts/extensions/themes dirs when isProjectTrusted()
        # is true, while user (~/.pi/agent) dirs are always added unconditionally.
        project_skills_dir = os.path.join(self._cwd, CONFIG_DIR_NAME, "skills")
        user_skills_dir = os.path.join(self._agent_dir, "skills")
        skill_paths: list[str] = []
        if not self._no_skills:
            # Only existing dirs: TS reaches `loadSkills` through
            # `collectAutoSkillEntries`, which yields nothing for a missing dir, so a
            # project with no `.pi/skills` produces no diagnostic. Passing the dir
            # unconditionally would make `loadSkills` warn "skill path does not exist"
            # for every project that simply has no skills. Explicitly configured paths
            # below are still diagnosed when missing, as in TS.
            if self._project_trusted and os.path.isdir(project_skills_dir):
                skill_paths.append(project_skills_dir)
            if os.path.isdir(user_skills_dir):
                skill_paths.append(user_skills_dir)
        skill_paths.extend(self._additional_skill_paths)
        self._last_skill_paths = list(skill_paths)

        if self._no_skills and not skill_paths:
            self._skills, self._skill_diagnostics = [], []
        else:
            result = load_skills(
                cwd=self._cwd, agent_dir=self._agent_dir, skill_paths=skill_paths, include_defaults=False
            )
            self._skills, self._skill_diagnostics = result.skills, result.diagnostics

        project_prompts_dir = os.path.join(self._cwd, CONFIG_DIR_NAME, "prompts")
        user_prompts_dir = os.path.join(self._agent_dir, "prompts")
        prompt_paths: list[str] = []
        if not self._no_prompt_templates:
            if self._project_trusted:
                prompt_paths.append(project_prompts_dir)
            prompt_paths.append(user_prompts_dir)
        prompt_paths.extend(self._additional_prompt_template_paths)
        self._last_prompt_paths = list(prompt_paths)

        if self._no_prompt_templates and not prompt_paths:
            self._prompts, self._prompt_diagnostics = [], []
        else:
            all_prompts = load_prompt_templates(
                cwd=self._cwd, agent_dir=self._agent_dir, prompt_paths=prompt_paths, include_defaults=False
            )
            self._prompts, self._prompt_diagnostics = _dedupe_prompts(all_prompts)

        self._agents_files = [] if self._no_context_files else load_project_context_files(self._cwd, self._agent_dir)

        system_prompt_source = self._system_prompt_source or discover_system_prompt_file(
            self._cwd, self._agent_dir, self._project_trusted
        )
        self._system_prompt = resolve_prompt_input(system_prompt_source, "system prompt")
        self._system_prompt_source_path = (
            resolve_path(system_prompt_source)
            if system_prompt_source and os.path.exists(system_prompt_source)
            else None
        )

        append_sources = self._append_system_prompt_source
        if not append_sources:
            discovered = discover_append_system_prompt_file(self._cwd, self._agent_dir, self._project_trusted)
            append_sources = [discovered] if discovered else []
        self._append_system_prompt = [
            resolved for source in append_sources if (resolved := resolve_prompt_input(source, "append system prompt"))
        ]
        self._append_system_prompt_source_paths = [
            resolve_path(source) for source in append_sources if os.path.exists(source)
        ]

    # ----------------------------------------------------------------------
    # Extension-contributed resources (port of `extendResources`)
    # ----------------------------------------------------------------------

    def _resolve_resource_path(self, path: str) -> str:
        return resolve_path(path, self._cwd, PathInputOptions(trim=True))

    def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        """Append `additional` to `primary`, dropping paths already present.

        Dedup is on the canonical (symlink-resolved) path, so a directory
        reached through a symlink is not loaded twice.
        """
        merged: list[str] = []
        seen: set[str] = set()
        for path in [*primary, *additional]:
            resolved = self._resolve_resource_path(path)
            canonical = canonicalize_path(resolved)
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(resolved)
        return merged

    def _normalize_extension_paths(self, entries: list[ExtensionResourcePath]) -> list[ExtensionResourcePath]:
        normalized: list[ExtensionResourcePath] = []
        for entry in entries:
            metadata = entry.metadata
            if metadata.base_dir:
                metadata = replace(metadata, base_dir=self._resolve_resource_path(metadata.base_dir))
            normalized.append(ExtensionResourcePath(path=self._resolve_resource_path(entry.path), metadata=metadata))
        return normalized

    @staticmethod
    def _find_extension_source_info(resource_path: str, source_infos: dict[str, SourceInfo]) -> SourceInfo | None:
        """The metadata of the contributed path that contains `resource_path`, if any."""
        if not resource_path or resource_path.startswith("<"):
            return None
        normalized = os.path.abspath(resource_path)
        for source_path, source_info in source_infos.items():
            normalized_source = os.path.abspath(source_path)
            if normalized == normalized_source or normalized.startswith(f"{normalized_source}{os.sep}"):
                return replace(source_info, path=resource_path)
        return None

    def extend_resources(
        self,
        *,
        skill_paths: list[ExtensionResourcePath] | None = None,
        prompt_paths: list[ExtensionResourcePath] | None = None,
    ) -> None:
        """Add extension-contributed skill/prompt paths after the initial load.

        Port of `DefaultResourceLoader.extendResources`. Discovery is re-run
        over the original paths plus the new ones (rather than appending the
        results) so name-collision precedence stays the same as a fresh load:
        the paths found first still win.

        TypeScript also accepts `themePaths`; this port has no theme system
        (see the module docstring), so that parameter is absent rather than
        accepted and ignored.
        """
        resolved_skill_paths = self._normalize_extension_paths(skill_paths or [])
        resolved_prompt_paths = self._normalize_extension_paths(prompt_paths or [])

        for entry in resolved_skill_paths:
            self._extension_skill_source_infos[entry.path] = SourceInfo(
                path=entry.path,
                source=entry.metadata.source,
                scope=entry.metadata.scope,
                origin=entry.metadata.origin,
                base_dir=entry.metadata.base_dir,
            )
        for entry in resolved_prompt_paths:
            self._extension_prompt_source_infos[entry.path] = SourceInfo(
                path=entry.path,
                source=entry.metadata.source,
                scope=entry.metadata.scope,
                origin=entry.metadata.origin,
                base_dir=entry.metadata.base_dir,
            )

        if resolved_skill_paths:
            self._last_skill_paths = self._merge_paths(
                self._last_skill_paths, [entry.path for entry in resolved_skill_paths]
            )
            result = load_skills(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                skill_paths=self._last_skill_paths,
                include_defaults=False,
            )
            self._skills = [
                replace(skill, source_info=extension_info)
                if (
                    extension_info := self._find_extension_source_info(
                        skill.file_path, self._extension_skill_source_infos
                    )
                )
                else skill
                for skill in result.skills
            ]
            self._skill_diagnostics = result.diagnostics

        if resolved_prompt_paths:
            self._last_prompt_paths = self._merge_paths(
                self._last_prompt_paths, [entry.path for entry in resolved_prompt_paths]
            )
            all_prompts = load_prompt_templates(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                prompt_paths=self._last_prompt_paths,
                include_defaults=False,
            )
            prompts, diagnostics = _dedupe_prompts(all_prompts)
            self._prompts = [
                replace(prompt, source_info=extension_info)
                if (
                    extension_info := self._find_extension_source_info(
                        prompt.file_path, self._extension_prompt_source_infos
                    )
                )
                else prompt
                for prompt in prompts
            ]
            self._prompt_diagnostics = diagnostics


__all__ = [
    "GitPaths",
    "LoadSkillsResult",
    "PromptTemplate",
    "ResourceCollision",
    "ResourceDiagnostic",
    "ResourceLoader",
    "ResourceLoaderOptions",
    "Skill",
    "SourceInfo",
    "create_synthetic_source_info",
    "discover_append_system_prompt_file",
    "discover_system_prompt_file",
    "expand_prompt_template",
    "find_git_paths",
    "format_skills_for_prompt",
    "load_project_context_files",
    "load_prompt_templates",
    "load_skills",
    "load_skills_from_dir",
    "parse_command_args",
    "parse_frontmatter",
    "resolve_prompt_input",
    "strip_frontmatter",
    "substitute_args",
]
