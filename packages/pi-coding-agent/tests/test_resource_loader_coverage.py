"""Additional coverage tests for `pi_coding_agent.core.resource_loader`.

Targets uncovered lines from the baseline run:
- `parse_frontmatter`: no-frontmatter, missing end marker, CRLF normalisation (lines 112, 153, 157, 159)
- `_validate_name` / `_validate_description` error paths (line 168)
- `_load_skill_from_file`: invalid skill, exception path (lines 189-208)
- `_load_skills_from_dir_internal`: OSError scandir, ignore rules (lines 241-242, 250-254, 263, 268-286)
- `load_skills`: include_defaults, path warnings (lines 324-326, 401, 429->434, 447-458, 488-491, 493, 495-497)
- `substitute_args`: `${N:-default}`, `${@:N:L}`, `$ARGUMENTS` forms (lines 501->503, 514-515, 518-524)
- `_dedupe_prompts`: collision path (lines 543->546, 549->552)
- `find_git_paths`: worktree (.git file) and regular dir (lines 561-608, 611-637)
- `load_project_context_files`: shadowed context, global file, ancestors (lines 700-743, 742-743)
- `discover_system_prompt_file` / `discover_append_system_prompt_file`: all branches (lines 823-845)
- `ResourceLoader.reload`: no_skills, no_prompt_templates, no_context_files, system/append prompt paths
"""

from __future__ import annotations

import os

from pi_coding_agent.core.resource_loader import (
    ResourceLoader,
    ResourceLoaderOptions,
    _validate_description,
    _validate_name,
    discover_append_system_prompt_file,
    discover_system_prompt_file,
    find_git_paths,
    load_project_context_files,
    load_skills,
    load_skills_from_dir,
    parse_command_args,
    parse_frontmatter,
    strip_frontmatter,
    substitute_args,
)

SKILL_MD = """---
name: {name}
description: {description}
---
{content}"""


def _mkloader(tmp_path, cwd=None, agent_dir=None, **kwargs) -> ResourceLoader:
    cwd = cwd or (tmp_path / "project")
    agent_dir = agent_dir or (tmp_path / "agent")
    os.makedirs(cwd, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)
    return ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir), **kwargs))


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


def test_parse_frontmatter_no_leading_dashes():
    meta, body = parse_frontmatter("No frontmatter here.")
    assert meta == {}
    assert "No frontmatter" in body


def test_parse_frontmatter_missing_end_marker():
    """A '---' that starts but never closes returns no frontmatter."""
    meta, _body = parse_frontmatter("---\nname: foo\n")
    assert meta == {}


def test_parse_frontmatter_crlf_normalised():
    """CRLF line endings are normalised before parsing."""
    content = "---\r\nname: skill-x\r\ndescription: desc\r\n---\r\nBody here."
    meta, body = parse_frontmatter(content)
    assert meta.get("name") == "skill-x"
    assert "Body here" in body


def test_strip_frontmatter():
    content = "---\ndescription: d\n---\nContent"
    assert strip_frontmatter(content) == "Content"


# ---------------------------------------------------------------------------
# _validate_name / _validate_description
# ---------------------------------------------------------------------------


def test_validate_name_too_long():
    long_name = "a" * 65
    errors = _validate_name(long_name)
    assert any("exceeds" in e for e in errors)


def test_validate_name_invalid_chars():
    errors = _validate_name("UPPER_CASE")
    assert any("invalid characters" in e for e in errors)


def test_validate_name_leading_hyphen():
    errors = _validate_name("-bad")
    assert any("hyphen" in e for e in errors)


def test_validate_name_trailing_hyphen():
    errors = _validate_name("bad-")
    assert any("hyphen" in e for e in errors)


def test_validate_name_consecutive_hyphens():
    errors = _validate_name("a--b")
    assert any("consecutive" in e for e in errors)


def test_validate_description_empty():
    errors = _validate_description("")
    assert any("required" in e for e in errors)


def test_validate_description_too_long():
    errors = _validate_description("x" * 1025)
    assert any("exceeds" in e for e in errors)


# ---------------------------------------------------------------------------
# load_skills_from_dir: invalid skill, exception path
# ---------------------------------------------------------------------------


def test_load_skill_with_missing_description_returns_diagnostic(tmp_path):
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nContent")
    result = load_skills_from_dir(str(tmp_path / "skills"), "user")
    assert all(s.name != "my-skill" for s in result.skills)
    assert any("description" in d.message for d in result.diagnostics)


def test_load_skill_with_invalid_name_returns_diagnostic(tmp_path):
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: INVALID_NAME\ndescription: A skill\n---\nContent")
    result = load_skills_from_dir(str(tmp_path / "skills"), "user")
    # Skill may or may not be loaded depending on name validation, but diagnostics should appear
    assert any("invalid" in d.message.lower() for d in result.diagnostics)


def test_load_skills_from_dir_nonexistent():
    result = load_skills_from_dir("/this/does/not/exist", "user")
    assert result.skills == []
    assert result.diagnostics == []


def test_load_skills_from_dir_with_gitignore(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / ".gitignore").write_text("ignored-skill.md\n")
    (skills_dir / "kept-skill.md").write_text(
        SKILL_MD.format(name="kept-skill", description="A kept skill", content="Content")
    )
    (skills_dir / "ignored-skill.md").write_text(
        SKILL_MD.format(name="ignored-skill", description="Should be ignored", content="Content")
    )
    result = load_skills_from_dir(str(skills_dir), "user")
    names = [s.name for s in result.skills]
    assert "kept-skill" in names
    assert "ignored-skill" not in names


def test_load_skills_from_dir_dot_hidden_files_skipped(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / ".hidden-skill.md").write_text(
        SKILL_MD.format(name="hidden", description="A hidden skill", content="Content")
    )
    result = load_skills_from_dir(str(skills_dir), "user")
    assert all(s.name != "hidden" for s in result.skills)


def test_load_skills_include_defaults(tmp_path):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    user_skills = agent_dir / "skills"
    user_skills.mkdir(parents=True)
    (user_skills / "user-skill.md").write_text(
        SKILL_MD.format(name="user-skill", description="User skill", content="Content")
    )
    result = load_skills(cwd=str(cwd), agent_dir=str(agent_dir), skill_paths=[], include_defaults=True)
    names = [s.name for s in result.skills]
    assert "user-skill" in names


def test_load_skills_path_does_not_exist_warning(tmp_path):
    result = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(tmp_path / "nonexistent.md")],
        include_defaults=False,
    )
    assert any("does not exist" in d.message for d in result.diagnostics)


def test_load_skills_path_not_markdown_warning(tmp_path):
    not_md = tmp_path / "something.txt"
    not_md.write_text("not markdown")
    result = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(not_md)],
        include_defaults=False,
    )
    assert any("not a markdown" in d.message for d in result.diagnostics)


def test_load_skills_explicit_file_path(tmp_path):
    skill_file = tmp_path / "my-skill.md"
    skill_file.write_text(SKILL_MD.format(name="my-skill", description="A skill", content="Content"))
    result = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(skill_file)],
        include_defaults=False,
    )
    assert any(s.name == "my-skill" for s in result.skills)


def test_load_skills_collision_diagnostic(tmp_path):
    skill_dir1 = tmp_path / "dir1"
    skill_dir2 = tmp_path / "dir2"
    skill_dir1.mkdir()
    skill_dir2.mkdir()
    (skill_dir1 / "dupe-skill.md").write_text(
        SKILL_MD.format(name="dupe-skill", description="First", content="Content")
    )
    (skill_dir2 / "dupe-skill.md").write_text(
        SKILL_MD.format(name="dupe-skill", description="Second", content="Content")
    )
    result = load_skills(
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(skill_dir1), str(skill_dir2)],
        include_defaults=False,
    )
    assert any(d.type == "collision" for d in result.diagnostics)
    # Only one skill with that name survives
    assert len([s for s in result.skills if s.name == "dupe-skill"]) == 1


# ---------------------------------------------------------------------------
# substitute_args
# ---------------------------------------------------------------------------


def test_substitute_args_dollar_at():
    assert substitute_args("all: $@", ["a", "b"]) == "all: a b"


def test_substitute_args_dollar_arguments():
    assert substitute_args("$ARGUMENTS", ["x", "y"]) == "x y"


def test_substitute_args_numbered():
    assert substitute_args("$1 and $2", ["foo", "bar"]) == "foo and bar"


def test_substitute_args_default_value_used_when_missing():
    assert substitute_args("${1:-default}", []) == "default"


def test_substitute_args_default_value_not_used_when_present():
    assert substitute_args("${1:-default}", ["actual"]) == "actual"


def test_substitute_args_at_default():
    assert substitute_args("${@:-fallback}", []) == "fallback"


def test_substitute_args_slice_start():
    assert substitute_args("${@:2}", ["a", "b", "c"]) == "b c"


def test_substitute_args_slice_start_length():
    assert substitute_args("${@:1:2}", ["a", "b", "c"]) == "a b"


def test_parse_command_args_quoted():
    args = parse_command_args('"hello world" foo')
    assert args == ["hello world", "foo"]


def test_parse_command_args_single_quote():
    args = parse_command_args("'hello world'")
    assert args == ["hello world"]


# ---------------------------------------------------------------------------
# _dedupe_prompts
# ---------------------------------------------------------------------------


def test_dedupe_prompts_collision_diagnostic(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    # Create same name in two dirs by loading from explicit paths

    dir2 = tmp_path / "prompts2"
    dir2.mkdir()
    (prompts_dir / "my-prompt.md").write_text("First prompt")
    (dir2 / "my-prompt.md").write_text("Second prompt")

    # Build the loader with two extra prompt directories that have colliding names
    loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(tmp_path / "project"),
            agent_dir=str(tmp_path / "agent"),
            additional_prompt_template_paths=[str(prompts_dir), str(dir2)],
            no_prompt_templates=False,
        )
    )
    os.makedirs(str(tmp_path / "project"), exist_ok=True)
    os.makedirs(str(tmp_path / "agent"), exist_ok=True)
    loader.reload()
    _, diagnostics = loader.get_prompts()
    assert any(d.type == "collision" for d in diagnostics)


# ---------------------------------------------------------------------------
# find_git_paths
# ---------------------------------------------------------------------------


def test_find_git_paths_regular_repo(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    result = find_git_paths(str(tmp_path))
    assert result is not None
    assert result.repo_dir == str(tmp_path)
    assert str(git_dir) in result.common_git_dir


def test_find_git_paths_no_head(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    # No HEAD file
    result = find_git_paths(str(tmp_path))
    assert result is None


def test_find_git_paths_not_found(tmp_path):
    # No .git anywhere
    result = find_git_paths(str(tmp_path / "some" / "deep" / "dir"))
    assert result is None


def test_find_git_paths_worktree(tmp_path):
    """A .git file (worktree) is resolved to the actual git directory."""
    main_git = tmp_path / "main" / ".git"
    main_git.mkdir(parents=True)
    (main_git / "HEAD").write_text("ref: refs/heads/main\n")

    worktree_git_dir = main_git / "worktrees" / "wt"
    worktree_git_dir.mkdir(parents=True)
    (worktree_git_dir / "HEAD").write_text("ref: refs/heads/feature\n")

    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()
    (worktree_dir / ".git").write_text(f"gitdir: {worktree_git_dir}\n")

    result = find_git_paths(str(worktree_dir))
    assert result is not None
    assert result.repo_dir == str(worktree_dir)


def test_find_git_paths_worktree_with_commondir(tmp_path):
    """A worktree .git file with a commondir pointer."""
    main_git = tmp_path / "main" / ".git"
    main_git.mkdir(parents=True)
    (main_git / "HEAD").write_text("ref: refs/heads/main\n")

    worktree_git_dir = main_git / "worktrees" / "wt"
    worktree_git_dir.mkdir(parents=True)
    (worktree_git_dir / "HEAD").write_text("ref: refs/heads/feature\n")
    (worktree_git_dir / "commondir").write_text("../..\n")

    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()
    (worktree_dir / ".git").write_text(f"gitdir: {worktree_git_dir}\n")

    result = find_git_paths(str(worktree_dir))
    assert result is not None
    assert "commondir" not in result.common_git_dir


# ---------------------------------------------------------------------------
# load_project_context_files
# ---------------------------------------------------------------------------


def test_load_project_context_files_no_files(tmp_path):
    result = load_project_context_files(str(tmp_path / "project"), str(tmp_path / "agent"))
    assert result == []


def test_load_project_context_files_global_agents_md(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "AGENTS.md").write_text("Global AGENTS")
    project = tmp_path / "project"
    project.mkdir()

    result = load_project_context_files(str(project), str(agent_dir))
    paths = [r["path"] for r in result]
    assert any("AGENTS.md" in p for p in paths)
    contents = [r["content"] for r in result]
    assert any("Global AGENTS" in c for c in contents)


def test_load_project_context_files_project_agents_md(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("Project AGENTS")

    result = load_project_context_files(str(project), str(agent_dir))
    assert any("Project AGENTS" in r["content"] for r in result)


def test_load_project_context_files_ancestor_ordering(tmp_path):
    """Context files in ancestor directories are loaded in outer-to-inner order."""
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True)
    (root / "AGENTS.md").write_text("Root context")
    (sub / "AGENTS.md").write_text("Sub context")

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = load_project_context_files(str(sub), str(agent_dir))
    assert len(result) >= 2
    contents = [r["content"] for r in result]
    root_idx = next(i for i, c in enumerate(contents) if "Root context" in c)
    sub_idx = next(i for i, c in enumerate(contents) if "Sub context" in c)
    assert root_idx < sub_idx


# ---------------------------------------------------------------------------
# discover_system_prompt_file / discover_append_system_prompt_file
# ---------------------------------------------------------------------------


def test_discover_system_prompt_project_trusted(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    (pi_dir / "SYSTEM.md").write_text("Project system prompt")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = discover_system_prompt_file(str(cwd), str(agent_dir), project_trusted=True)
    assert result is not None
    assert "SYSTEM.md" in result


def test_discover_system_prompt_not_trusted_uses_global(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    (pi_dir / "SYSTEM.md").write_text("Project system prompt")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "SYSTEM.md").write_text("Global system prompt")

    result = discover_system_prompt_file(str(cwd), str(agent_dir), project_trusted=False)
    assert result is not None
    assert str(agent_dir) in result


def test_discover_system_prompt_falls_back_to_global(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "SYSTEM.md").write_text("Global only")

    result = discover_system_prompt_file(str(cwd), str(agent_dir), project_trusted=True)
    assert result is not None
    assert "SYSTEM.md" in result


def test_discover_system_prompt_returns_none_when_missing(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    assert discover_system_prompt_file(str(cwd), str(agent_dir), project_trusted=True) is None


def test_discover_append_system_prompt_project_trusted(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    (pi_dir / "APPEND_SYSTEM.md").write_text("Append system prompt")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = discover_append_system_prompt_file(str(cwd), str(agent_dir), project_trusted=True)
    assert result is not None
    assert "APPEND_SYSTEM.md" in result


def test_discover_append_system_prompt_global_fallback(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "APPEND_SYSTEM.md").write_text("Global append")

    result = discover_append_system_prompt_file(str(cwd), str(agent_dir), project_trusted=False)
    assert result is not None
    assert str(agent_dir) in result


def test_discover_append_system_prompt_returns_none(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    assert discover_append_system_prompt_file(str(cwd), str(agent_dir), project_trusted=True) is None


# ---------------------------------------------------------------------------
# ResourceLoader.reload: no_skills / no_prompt_templates / system prompt paths
# ---------------------------------------------------------------------------


def test_resource_loader_no_skills_flag(tmp_path):
    loader = _mkloader(tmp_path, no_skills=True)
    loader.reload()
    assert loader.get_skills().skills == []


def test_resource_loader_no_prompt_templates_flag(tmp_path):
    loader = _mkloader(tmp_path, no_prompt_templates=True)
    loader.reload()
    prompts, _ = loader.get_prompts()
    assert prompts == []


def test_resource_loader_no_context_files_flag(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("Context")
    loader = _mkloader(tmp_path, cwd=cwd, no_context_files=True)
    loader.reload()
    assert loader.get_agents_files() == []


def test_resource_loader_system_prompt_from_literal(tmp_path):
    loader = _mkloader(tmp_path, system_prompt="My inline system prompt")
    loader.reload()
    assert loader.get_system_prompt() == "My inline system prompt"
    assert loader.get_system_prompt_source() is None


def test_resource_loader_system_prompt_from_file(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    sys_file = agent_dir / "SYSTEM.md"
    sys_file.write_text("File-based system prompt")
    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()
    assert loader.get_system_prompt() == "File-based system prompt"
    assert loader.get_system_prompt_source() is not None


def test_resource_loader_append_system_prompt_from_list(tmp_path):
    loader = _mkloader(tmp_path, append_system_prompt=["Appended text"])
    loader.reload()
    assert "Appended text" in loader.get_append_system_prompt()


def test_resource_loader_append_system_prompt_from_file(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    append_file = agent_dir / "APPEND_SYSTEM.md"
    append_file.write_text("Appended from file")
    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()
    assert any("Appended from file" in s for s in loader.get_append_system_prompt())


def test_resource_loader_untrusted_project_skips_project_resources(tmp_path):
    cwd = tmp_path / "project"
    project_skills = cwd / ".pi" / "skills"
    project_skills.mkdir(parents=True)
    (project_skills / "proj-skill.md").write_text(
        SKILL_MD.format(name="proj-skill", description="Project skill", content="Content")
    )
    loader = _mkloader(tmp_path, cwd=cwd, project_trusted=False)
    loader.reload()
    names = [s.name for s in loader.get_skills().skills]
    assert "proj-skill" not in names


def test_resource_loader_set_project_trusted(tmp_path):
    loader = _mkloader(tmp_path)
    loader.set_project_trusted(False)
    assert loader.is_project_trusted() is False
    loader.set_project_trusted(True)
    assert loader.is_project_trusted() is True
