"""Tests for pi_coding_agent.core.resource_loader.

Ported from packages/coding-agent/test/resource-loader.test.ts.

`DefaultResourceLoader` also owns extension and theme discovery; this port's
`ResourceLoader` covers only AGENTS.md context files, skills, and prompt
templates (see `core/resource_loader.py`'s module docstring). The TypeScript
cases that exercise the extension/auto-discovery side of the loader are
therefore ported against the modules that own that behavior here --
`core/extensions/loader.py` and `core/package_manager.py` -- and the handful
that pin `DefaultResourceLoader`-only seams (`extendResources`,
`skillsOverride`, `reload({ resolveProjectTrust })`, CLI-extension ordering,
symlink canonicalization) are skipped individually at the bottom of this file
with the reason on each marker. Themes appear only through
`PackageManager.resolve()`, which does port them.
"""

import os
from pathlib import Path

import pytest
from pi_ai.auth.types import Credential
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.extensions.loader import (
    ExtensionAPI,
    NamedInlineExtension,
    discover_and_load_extensions,
    load_extension_factories,
)
from pi_coding_agent.core.extensions.runner import ExtensionRunner
from pi_coding_agent.core.extensions.types import ResourcesDiscoverResult
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.package_manager import PackageManager
from pi_coding_agent.core.resource_loader import (
    ExtensionResourcePath,
    ResourceLoader,
    ResourceLoaderOptions,
    ResourcePathMetadata,
    load_project_context_files,
)
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager


def _write(path: str, content: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _any_enabled(resources, suffix: str) -> bool:
    return any(r.path.replace("\\", "/").endswith(suffix) and r.enabled for r in resources)


def _mkloader(tmp_path, cwd=None, agent_dir=None, **kwargs) -> ResourceLoader:
    cwd = cwd or (tmp_path / "project")
    agent_dir = agent_dir or (tmp_path / "agent")
    os.makedirs(cwd, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)
    loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir), **kwargs))
    return loader


SKILL_MD = """---
name: {name}
description: {description}
---
{content}"""


def test_initializes_with_empty_results_before_reload(tmp_path):
    loader = _mkloader(tmp_path)
    assert loader.get_skills().skills == []
    assert loader.get_prompts()[0] == []


def test_discovers_skills_from_agent_dir(tmp_path):
    agent_dir = tmp_path / "agent"
    skills_dir = agent_dir / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "test-skill.md").write_text(
        SKILL_MD.format(name="test-skill", description="A test skill", content="Skill content here.")
    )

    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()

    skills = loader.get_skills().skills
    assert any(s.name == "test-skill" for s in skills)


def test_ignores_extra_markdown_files_in_auto_discovered_skill_dirs(tmp_path):
    agent_dir = tmp_path / "agent"
    skill_dir = agent_dir / "skills" / "pi-skills" / "browser-tools"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        SKILL_MD.format(name="browser-tools", description="Browser tools", content="Skill content here.")
    )
    (skill_dir / "EFFICIENCY.md").write_text("No frontmatter here")

    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()

    result = loader.get_skills()
    assert any(s.name == "browser-tools" for s in result.skills)
    assert not any((d.path or "").endswith("EFFICIENCY.md") for d in result.diagnostics)


def test_discovers_prompts_from_agent_dir(tmp_path):
    agent_dir = tmp_path / "agent"
    prompts_dir = agent_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "test-prompt.md").write_text("---\ndescription: A test prompt\n---\nPrompt content.")

    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()

    prompts, _diagnostics = loader.get_prompts()
    assert any(p.name == "test-prompt" for p in prompts)


def test_prefers_project_resources_over_user_on_name_collisions(tmp_path):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    user_prompts_dir = agent_dir / "prompts"
    project_prompts_dir = cwd / ".pi" / "prompts"
    user_prompts_dir.mkdir(parents=True)
    project_prompts_dir.mkdir(parents=True)
    user_prompt_path = user_prompts_dir / "commit.md"
    project_prompt_path = project_prompts_dir / "commit.md"
    user_prompt_path.write_text("User prompt")
    project_prompt_path.write_text("Project prompt")

    user_skill_dir = agent_dir / "skills" / "collision-skill"
    project_skill_dir = cwd / ".pi" / "skills" / "collision-skill"
    user_skill_dir.mkdir(parents=True)
    project_skill_dir.mkdir(parents=True)
    (user_skill_dir / "SKILL.md").write_text(
        SKILL_MD.format(name="collision-skill", description="user", content="User skill")
    )
    project_skill_path = project_skill_dir / "SKILL.md"
    project_skill_path.write_text(
        SKILL_MD.format(name="collision-skill", description="project", content="Project skill")
    )

    loader = _mkloader(tmp_path, cwd=cwd, agent_dir=agent_dir)
    loader.reload()

    prompts, _ = loader.get_prompts()
    prompt = next(p for p in prompts if p.name == "commit")
    assert prompt.file_path == str(project_prompt_path)

    skill = next(s for s in loader.get_skills().skills if s.name == "collision-skill")
    assert skill.file_path == str(project_skill_path)
    # TS also collides a theme (`getThemes().themes` sourcePath), which this port's
    # `ResourceLoader` cannot: it does no theme loading at all (module docstring).
    # Theme precedence is pinned instead by `PackageManager.resolve()` in
    # `test_honors_overrides_for_auto_discovered_resources`.


def test_discovers_agents_md_context_files(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "AGENTS.md").write_text("# Project Guidelines\n\nBe helpful.")

    loader = _mkloader(tmp_path, cwd=cwd)
    loader.reload()

    agents_files = loader.get_agents_files()
    assert any("AGENTS.md" in f["path"] for f in agents_files)


def test_prefers_agents_override_md_while_preserving_ancestor_layering(tmp_path):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    nested_cwd = cwd / "service"
    nested_cwd.mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    (agent_dir / "AGENTS.md").write_text("global instructions")
    (agent_dir / "AGENTS.override.md").write_text("global override")
    (cwd / "AGENTS.md").write_text("project instructions")
    (nested_cwd / "AGENTS.md").write_text("service instructions")
    (nested_cwd / "AGENTS.override.md").write_text("service override")

    loader = _mkloader(tmp_path, cwd=nested_cwd, agent_dir=agent_dir)
    loader.reload()

    assert loader.get_agents_files() == [
        {"path": str(agent_dir / "AGENTS.override.md"), "content": "global override"},
        {"path": str(cwd / "AGENTS.md"), "content": "project instructions"},
        {"path": str(nested_cwd / "AGENTS.override.md"), "content": "service override"},
    ]


def test_ignores_context_file_candidates_that_are_directories(tmp_path, capsys):
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "AGENTS.override.md").mkdir()
    (cwd / "AGENTS.md").mkdir()
    (cwd / "CLAUDE.md").write_text("Fallback instructions")

    loader = _mkloader(tmp_path, cwd=cwd)
    loader.reload()

    assert {"path": str(cwd / "CLAUDE.md"), "content": "Fallback instructions"} in loader.get_agents_files()
    # TS additionally asserts `console.error` was never called with either directory
    # path; this port's `_load_context_file_from_dir` warns on stdout instead.
    output = capsys.readouterr()
    combined = output.out + output.err
    assert str(cwd / "AGENTS.md") not in combined
    assert str(cwd / "AGENTS.override.md") not in combined


def test_skips_context_file_discovery_when_no_context_files(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "AGENTS.override.md").write_text("# Override Guidelines\n\nBe helpful.")
    (cwd / "AGENTS.md").write_text("# Project Guidelines\n\nBe helpful.")
    (cwd / "CLAUDE.md").write_text("# Claude Guidelines\n\nBe helpful.")

    loader = _mkloader(tmp_path, cwd=cwd, no_context_files=True)
    loader.reload()

    assert loader.get_agents_files() == []


def test_discovers_system_md_from_cwd_pi(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    (pi_dir / "SYSTEM.md").write_text("You are a helpful assistant.")

    loader = _mkloader(tmp_path, cwd=cwd)
    loader.reload()

    assert loader.get_system_prompt() == "You are a helpful assistant."


def test_skips_project_resources_that_require_trust_when_project_is_not_trusted(tmp_path):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    skill_dir = pi_dir / "skills" / "project-skill"
    prompts_dir = pi_dir / "prompts"
    skill_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    (pi_dir / "SYSTEM.md").write_text("Project system prompt.")
    (agent_dir / "SYSTEM.md").write_text("Global system prompt.")
    (agent_dir / "AGENTS.md").write_text("Global instructions")
    (cwd / "AGENTS.md").write_text("Project instructions")
    (skill_dir / "SKILL.md").write_text(
        SKILL_MD.format(name="project-skill", description="Project skill", content="Project skill content")
    )
    (prompts_dir / "project.md").write_text("Project prompt")

    loader = _mkloader(tmp_path, cwd=cwd, agent_dir=agent_dir, project_trusted=False)
    loader.reload()

    assert loader.get_system_prompt() == "Global system prompt."
    agents_files = loader.get_agents_files()
    assert any(f["path"] == str(agent_dir / "AGENTS.md") for f in agents_files)
    assert any(f["path"] == str(cwd / "AGENTS.md") for f in agents_files)
    assert not any(s.name == "project-skill" for s in loader.get_skills().skills)
    prompts, _ = loader.get_prompts()
    assert not any(p.name == "project" for p in prompts)
    # TS also asserts `getExtensions().extensions` is empty with no errors, and that no
    # `project-theme` is loaded. This port's `ResourceLoader` owns neither extensions nor
    # themes (module docstring); untrusted-project extension skipping is pinned by
    # `test_untrusted_project_refuses_project_local_extensions` in
    # `tests/test_extensions_loader.py`.


def test_discovers_append_system_md(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    (pi_dir / "APPEND_SYSTEM.md").write_text("Additional instructions.")

    loader = _mkloader(tmp_path, cwd=cwd)
    loader.reload()

    assert "Additional instructions." in loader.get_append_system_prompt()


# --------------------------------------------------------------------------
# system prompt sources
# --------------------------------------------------------------------------


def test_exposes_discovered_project_system_md_as_the_system_prompt_source(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    system_prompt_path = pi_dir / "SYSTEM.md"
    system_prompt_path.write_text("Project system prompt.")

    loader = _mkloader(tmp_path, cwd=cwd)
    loader.reload()

    assert loader.get_system_prompt() == "Project system prompt."
    assert loader.get_system_prompt_source() == str(system_prompt_path)


def test_exposes_discovered_global_system_md_as_the_system_prompt_source(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    system_prompt_path = agent_dir / "SYSTEM.md"
    system_prompt_path.write_text("Global system prompt.")

    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()

    assert loader.get_system_prompt() == "Global system prompt."
    assert loader.get_system_prompt_source() == str(system_prompt_path)


def test_does_not_expose_literal_system_prompt_text_as_a_source(tmp_path):
    loader = _mkloader(tmp_path, system_prompt="Literal system prompt.")
    loader.reload()

    assert loader.get_system_prompt() == "Literal system prompt."
    assert loader.get_system_prompt_source() is None


def test_exposes_file_backed_system_prompt_options_as_a_source(tmp_path):
    system_prompt_path = tmp_path / "custom-system.md"
    system_prompt_path.write_text("Custom system prompt.")

    loader = _mkloader(tmp_path, system_prompt=str(system_prompt_path))
    loader.reload()

    assert loader.get_system_prompt() == "Custom system prompt."
    assert loader.get_system_prompt_source() == str(system_prompt_path)


def test_exposes_discovered_append_system_md_as_an_append_system_prompt_source(tmp_path):
    cwd = tmp_path / "project"
    pi_dir = cwd / ".pi"
    pi_dir.mkdir(parents=True)
    append_system_prompt_path = pi_dir / "APPEND_SYSTEM.md"
    append_system_prompt_path.write_text("Project append prompt.")

    loader = _mkloader(tmp_path, cwd=cwd)
    loader.reload()

    assert loader.get_append_system_prompt() == ["Project append prompt."]
    assert loader.get_append_system_prompt_sources() == [str(append_system_prompt_path)]


def test_does_not_expose_literal_append_system_prompt_text_as_a_source(tmp_path):
    loader = _mkloader(tmp_path, append_system_prompt=["Literal append prompt."])
    loader.reload()

    assert loader.get_append_system_prompt() == ["Literal append prompt."]
    assert loader.get_append_system_prompt_sources() == []


def test_only_exposes_file_backed_append_system_prompt_options_as_sources(tmp_path):
    append_system_prompt_path = tmp_path / "custom-append.md"
    append_system_prompt_path.write_text("Custom append prompt.")

    loader = _mkloader(tmp_path, append_system_prompt=[str(append_system_prompt_path), "Literal append prompt."])
    loader.reload()

    assert loader.get_append_system_prompt() == ["Custom append prompt.", "Literal append prompt."]
    assert loader.get_append_system_prompt_sources() == [str(append_system_prompt_path)]


# --------------------------------------------------------------------------
# noSkills option
# --------------------------------------------------------------------------


def test_skips_skill_discovery_when_no_skills_is_true(tmp_path):
    agent_dir = tmp_path / "agent"
    skills_dir = agent_dir / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "test-skill.md").write_text(
        SKILL_MD.format(name="test-skill", description="A test skill", content="Content")
    )

    loader = _mkloader(tmp_path, agent_dir=agent_dir, no_skills=True)
    loader.reload()

    assert loader.get_skills().skills == []


def test_still_loads_additional_skill_paths_when_no_skills_is_true(tmp_path):
    custom_skill_dir = tmp_path / "custom-skills"
    custom_skill_dir.mkdir(parents=True)
    (custom_skill_dir / "custom.md").write_text(
        SKILL_MD.format(name="custom", description="Custom skill", content="Content")
    )

    loader = _mkloader(tmp_path, no_skills=True, additional_skill_paths=[str(custom_skill_dir)])
    loader.reload()

    assert any(s.name == "custom" for s in loader.get_skills().skills)


# --------------------------------------------------------------------------
# loadProjectContextFiles - nested worktree dedup
# --------------------------------------------------------------------------


def _link_worktree(main_dir, worktree_dir, name):
    git_dir = main_dir / ".git" / "worktrees" / name
    git_dir.mkdir(parents=True)
    (main_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/feat\n")
    (git_dir / "commondir").write_text("../..")
    (worktree_dir / ".git").write_text(f"gitdir: {git_dir}\n")


def _setup_nested_worktree(tmp_path):
    outer = tmp_path / "outer"
    main = outer / "main"
    worktree = main / "worktrees" / "feat"
    worktree_src = worktree / "src"
    worktree_src.mkdir(parents=True)
    _link_worktree(main, worktree, "feat")
    return outer, main, worktree, worktree_src


def test_skips_the_main_repos_duplicate_when_the_worktree_root_has_its_own_context(tmp_path):
    _outer, main, worktree, worktree_src = _setup_nested_worktree(tmp_path)
    (main / "AGENTS.md").write_text("main repo instructions")
    (worktree / "AGENTS.md").write_text("worktree instructions")

    files = load_project_context_files(str(worktree_src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["worktree instructions"]


def test_still_inherits_the_main_repos_context_when_the_worktree_root_has_none(tmp_path):
    _outer, main, _worktree, worktree_src = _setup_nested_worktree(tmp_path)
    (main / "AGENTS.md").write_text("main repo instructions")

    files = load_project_context_files(str(worktree_src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["main repo instructions"]


def test_only_skips_the_same_filename_not_a_differently_named_context_file(tmp_path):
    _outer, main, worktree, worktree_src = _setup_nested_worktree(tmp_path)
    (main / "CLAUDE.md").write_text("main repo instructions")
    (worktree / "AGENTS.md").write_text("worktree instructions")

    files = load_project_context_files(str(worktree_src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["main repo instructions", "worktree instructions"]


def test_does_not_skip_the_containers_context_in_a_bare_layout(tmp_path):
    proj = tmp_path / "proj"
    bare = proj / ".bare"
    worktree = proj / "main"
    worktree_git_dir = bare / "worktrees" / "main"
    worktree_git_dir.mkdir(parents=True)
    worktree.mkdir(parents=True)
    (bare / "HEAD").write_text("ref: refs/heads/main\n")
    (worktree_git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (worktree_git_dir / "commondir").write_text("../..")
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}\n")
    (proj / "AGENTS.md").write_text("container instructions")
    (worktree / "AGENTS.md").write_text("worktree instructions")

    files = load_project_context_files(str(worktree), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["container instructions", "worktree instructions"]


def test_keeps_loading_ancestors_above_the_main_repo(tmp_path):
    outer, main, worktree, worktree_src = _setup_nested_worktree(tmp_path)
    (outer / "AGENTS.md").write_text("outer instructions")
    (main / "AGENTS.md").write_text("main repo instructions")
    (worktree / "AGENTS.md").write_text("worktree instructions")

    files = load_project_context_files(str(worktree_src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["outer instructions", "worktree instructions"]


def test_does_not_skip_anything_for_a_sibling_worktree(tmp_path):
    outer = tmp_path / "outer"
    main = outer / "main"
    sib = outer / "sib-feat"
    sib_src = sib / "src"
    sib_src.mkdir(parents=True)
    main.mkdir(parents=True)
    (outer / "AGENTS.md").write_text("outer instructions")
    (sib / "AGENTS.md").write_text("sibling worktree instructions")
    _link_worktree(main, sib, "sib")

    files = load_project_context_files(str(sib_src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["outer instructions", "sibling worktree instructions"]


def test_does_not_skip_the_superprojects_context_from_inside_a_submodule(tmp_path):
    sup = tmp_path / "super"
    sub = sup / "vendor" / "lib"
    sub_src = sub / "src"
    sub_src.mkdir(parents=True)
    (sup / "AGENTS.md").write_text("superproject instructions")
    (sub / "AGENTS.md").write_text("submodule instructions")
    sub_git_dir = sup / ".git" / "modules" / "vendor" / "lib"
    sub_git_dir.mkdir(parents=True)
    (sub_git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (sub / ".git").write_text(f"gitdir: {sub_git_dir}\n")

    files = load_project_context_files(str(sub_src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["superproject instructions", "submodule instructions"]


def test_keeps_climbing_past_an_ordinary_repo_root(tmp_path):
    outer = tmp_path / "outer"
    repo = outer / "repo"
    leaf = repo / "src"
    leaf.mkdir(parents=True)
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (outer / "AGENTS.md").write_text("outer instructions")
    (repo / "AGENTS.md").write_text("repo instructions")
    (leaf / "AGENTS.md").write_text("leaf instructions")

    files = load_project_context_files(str(leaf), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["outer instructions", "repo instructions", "leaf instructions"]


def test_climbs_normally_when_the_gitdir_target_does_not_exist(tmp_path):
    repo = tmp_path / "corrupt"
    src = repo / "src"
    src.mkdir(parents=True)
    (repo / ".git").write_text("gitdir: /nonexistent/path/worktrees/feat\n")
    (repo / "AGENTS.md").write_text("repo instructions")
    (src / "AGENTS.md").write_text("src instructions")

    files = load_project_context_files(str(src), str(tmp_path / "agent"))

    assert [f["content"] for f in files] == ["repo instructions", "src instructions"]


# --------------------------------------------------------------------------
# Extension loading and auto-discovery overrides
#
# `DefaultResourceLoader` also owns extension/theme discovery; this port's
# `ResourceLoader` covers only context files, skills, and prompt templates
# (see its module docstring), so the extension-facing cases below are ported
# against the modules that actually own the behavior here:
# `discover_and_load_extensions()` and `PackageManager.resolve()`.
# --------------------------------------------------------------------------


_DEPLOY_EXTENSION = """
def pi_extension(pi):
    async def _handler(args, ctx):
        return None

    pi.register_command("deploy", handler=_handler, description="{scope} deploy")
    pi.register_command("{scope}-only", handler=_handler, description="{scope} only")
"""


def _duplicate_tool_extension(description: str) -> str:
    return f"""
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition


async def _execute(tool_call_id, params, signal, on_update, ctx):
    from pi_agent.types import AgentToolResult

    return AgentToolResult(content=[TextContent(text="ok")])


def pi_extension(pi):
    pi.register_tool(
        ToolDefinition(
            name="duplicate-tool",
            label="duplicate-tool",
            description="{description}",
            execute=_execute,
        )
    )
"""


async def test_keeps_both_extensions_loaded_when_command_names_collide(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    _write(str(cwd / ".pi" / "extensions" / "project.py"), _DEPLOY_EXTENSION.format(scope="project"))
    _write(str(agent_dir / "extensions" / "user.py"), _DEPLOY_EXTENSION.format(scope="user"))

    result = await discover_and_load_extensions([], str(cwd), agent_dir=str(agent_dir))

    assert len(result.extensions) == 2
    # TypeScript only reports tool and flag conflicts, never command-name ones:
    # colliding command names are disambiguated into `deploy:1`/`deploy:2` below.
    assert result.errors == []
    assert result.conflicts == []

    runner = ExtensionRunner(result.extensions, cwd=str(cwd))

    assert runner.get_command("deploy:1").description == "project deploy"
    assert runner.get_command("deploy:2").description == "user deploy"
    assert runner.get_command("project-only").description == "project only"
    assert runner.get_command("user-only").description == "user only"
    assert [name for name, _command in runner.get_registered_commands()] == [
        "deploy:1",
        "project-only",
        "deploy:2",
        "user-only",
    ]


async def test_detects_tool_conflicts_between_extensions(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    ext_dir = agent_dir / "extensions"
    _write(str(ext_dir / "ext1" / "__init__.py"), _duplicate_tool_extension("First"))
    _write(str(ext_dir / "ext2" / "__init__.py"), _duplicate_tool_extension("Second"))

    result = await discover_and_load_extensions([], str(cwd), agent_dir=str(agent_dir))

    assert len(result.extensions) == 2
    # Both extensions stay loaded; the clash is advisory. TypeScript's
    # `DefaultResourceLoader` folds these into `getExtensions().errors` after
    # loading, this port reports them on `LoadExtensionsResult.conflicts`
    # (see `detect_extension_conflicts`'s docstring).
    assert result.errors == []
    assert any(
        "duplicate-tool" in conflict["error"] and "conflicts" in conflict["error"] for conflict in result.conflicts
    )


async def test_loads_symlinked_user_and_project_extensions_once(tmp_path):
    """TS pins this on `DefaultResourceLoader`, whose `mergePaths()` dedups on
    `canonicalizePath` (realpath). This port has no `DefaultResourceLoader`, so
    `discover_and_load_extensions()` is the production discovery path and must
    canonicalize itself -- otherwise a shared extension dir symlinked into both
    the user and project locations registers every command twice.
    """
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    shared_ext_dir = tmp_path / "shared-extensions"
    _write(
        str(shared_ext_dir / "shared.py"),
        "def pi_extension(pi):\n"
        "    async def _handler(args, ctx):\n"
        "        return None\n\n"
        "    pi.register_command('shared', handler=_handler, description='shared command')\n",
    )
    (cwd / ".pi").mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    os.symlink(shared_ext_dir, agent_dir / "extensions", target_is_directory=True)
    os.symlink(shared_ext_dir, cwd / ".pi" / "extensions", target_is_directory=True)

    result = await discover_and_load_extensions([], str(cwd), agent_dir=str(agent_dir))

    assert len(result.extensions) == 1
    assert result.errors == []
    # Project paths are processed first, so the project alias is the survivor.
    assert result.extensions[0].path == str(cwd / ".pi" / "extensions" / "shared.py")

    runner = ExtensionRunner(result.extensions, cwd=str(cwd))
    assert [name for name, _command in runner.get_registered_commands()] == ["shared"]


async def test_honors_overrides_for_auto_discovered_resources(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    settings_manager = SettingsManager.in_memory()
    settings_manager.set_extension_paths(["-extensions/disabled.py"])
    settings_manager.set_skill_paths(["-skills/skip-skill"])
    settings_manager.set_prompt_template_paths(["-prompts/skip.md"])
    settings_manager.set_theme_paths(["-themes/skip.json"])

    _write(str(agent_dir / "extensions" / "disabled.py"), "def pi_extension(pi):\n    pass\n")
    _write(
        str(agent_dir / "skills" / "skip-skill" / "SKILL.md"),
        SKILL_MD.format(name="skip-skill", description="Skip me", content="Content"),
    )
    _write(str(agent_dir / "prompts" / "skip.md"), "Skip prompt")
    _write(str(agent_dir / "themes" / "skip.json"), "{}")

    manager = PackageManager(str(cwd), str(agent_dir), settings_manager)
    resolved = await manager.resolve()

    assert not _any_enabled(resolved.extensions, "extensions/disabled.py")
    assert not _any_enabled(resolved.skills, "skills/skip-skill")
    assert not _any_enabled(resolved.prompts, "prompts/skip.md")
    assert not _any_enabled(resolved.themes, "themes/skip.json")


# --------------------------------------------------------------------------
# extendResources
# --------------------------------------------------------------------------


def test_extend_resources_loads_skills_and_prompts_with_extension_metadata(tmp_path):
    extra_skill_dir = tmp_path / "extra-skills" / "extra-skill"
    skill_path = _write(
        str(extra_skill_dir / "SKILL.md"),
        SKILL_MD.format(name="extra-skill", description="Extra skill", content="Extra content"),
    )
    extra_prompt_dir = tmp_path / "extra-prompts"
    prompt_path = _write(
        str(extra_prompt_dir / "extra.md"),
        "---\ndescription: Extra prompt\n---\nExtra prompt content",
    )

    loader = _mkloader(tmp_path)
    loader.reload()

    loader.extend_resources(
        skill_paths=[
            ExtensionResourcePath(
                path=str(extra_skill_dir),
                metadata=ResourcePathMetadata(
                    source="extension:extra", scope="temporary", origin="top-level", base_dir=str(extra_skill_dir)
                ),
            )
        ],
        prompt_paths=[
            ExtensionResourcePath(
                path=prompt_path,
                metadata=ResourcePathMetadata(
                    source="extension:extra", scope="temporary", origin="top-level", base_dir=str(extra_prompt_dir)
                ),
            )
        ],
    )

    skills = loader.get_skills().skills
    loaded_skill = next((s for s in skills if s.name == "extra-skill"), None)
    assert loaded_skill is not None
    assert loaded_skill.source_info is not None
    assert loaded_skill.source_info.source == "extension:extra"
    assert loaded_skill.source_info.path == skill_path

    prompts, _ = loader.get_prompts()
    loaded_prompt = next((p for p in prompts if p.name == "extra"), None)
    assert loaded_prompt is not None
    assert loaded_prompt.source_info is not None
    assert loaded_prompt.source_info.source == "extension:extra"
    assert loaded_prompt.source_info.path == prompt_path


def test_extend_resources_loads_extension_resources_returned_as_file_urls(tmp_path):
    # The space in the directory name is the point: a file URL percent-escapes it,
    # so a loader that treated the href as a plain path would look for "%20".
    extra_skill_dir = tmp_path / "extra skills" / "file-url-skill"
    skill_path = _write(
        str(extra_skill_dir / "SKILL.md"),
        SKILL_MD.format(name="file-url-skill", description="File URL skill", content="Extra content"),
    )

    loader = _mkloader(tmp_path)
    loader.reload()

    loader.extend_resources(
        skill_paths=[
            ExtensionResourcePath(
                path=Path(str(extra_skill_dir)).as_uri(),
                metadata=ResourcePathMetadata(
                    source="extension:file-url", scope="temporary", origin="top-level", base_dir=str(extra_skill_dir)
                ),
            )
        ]
    )

    result = loader.get_skills()
    skills, diagnostics = result.skills, result.diagnostics
    assert diagnostics == []
    loaded_skill = next((s for s in skills if s.name == "file-url-skill"), None)
    assert loaded_skill is not None
    assert loaded_skill.file_path == skill_path
    assert loaded_skill.source_info is not None
    assert loaded_skill.source_info.source == "extension:file-url"


def test_extend_resources_keeps_previously_loaded_resources(tmp_path):
    agent_dir = tmp_path / "agent"
    _write(
        str(agent_dir / "skills" / "base-skill" / "SKILL.md"),
        SKILL_MD.format(name="base-skill", description="Base skill", content="Base"),
    )
    extra_skill_dir = tmp_path / "extra-skills" / "extra-skill"
    _write(
        str(extra_skill_dir / "SKILL.md"),
        SKILL_MD.format(name="extra-skill", description="Extra skill", content="Extra"),
    )

    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()
    assert {s.name for s in loader.get_skills().skills} == {"base-skill"}

    loader.extend_resources(
        skill_paths=[
            ExtensionResourcePath(
                path=str(extra_skill_dir),
                metadata=ResourcePathMetadata(source="extension:extra"),
            )
        ]
    )
    assert {s.name for s in loader.get_skills().skills} == {"base-skill", "extra-skill"}

    # The original discovery keeps its own source info; only the contributed path is tagged.
    by_name = {s.name: s for s in loader.get_skills().skills}
    assert by_name["base-skill"].source_info is not None
    assert by_name["base-skill"].source_info.source != "extension:extra"


def test_extend_resources_does_not_load_the_same_directory_twice(tmp_path):
    extra_skill_dir = tmp_path / "extra-skills" / "dup-skill"
    _write(
        str(extra_skill_dir / "SKILL.md"),
        SKILL_MD.format(name="dup-skill", description="Dup skill", content="Dup"),
    )
    link = tmp_path / "linked-skills"
    os.symlink(str(tmp_path / "extra-skills"), str(link))

    loader = _mkloader(tmp_path)
    loader.reload()
    loader.extend_resources(
        skill_paths=[
            ExtensionResourcePath(path=str(tmp_path / "extra-skills"), metadata=ResourcePathMetadata(source="a"))
        ]
    )
    loader.extend_resources(
        skill_paths=[ExtensionResourcePath(path=str(link), metadata=ResourcePathMetadata(source="b"))]
    )

    skills = loader.get_skills().skills
    assert [s.name for s in skills] == ["dup-skill"]


async def test_resources_discover_contributes_skills_and_prompts_to_a_session(tmp_path):
    """The `resources_discover` extension hook end to end.

    No TypeScript test drives this; `examples/extensions/dynamic-resources`
    does, and the hook is the only reason `extendResources` exists.
    """
    contributed = tmp_path / "contributed"
    _write(
        str(contributed / "SKILL.md"),
        SKILL_MD.format(name="dynamic-skill", description="Dynamic skill", content="Dynamic content"),
    )
    _write(str(contributed / "dynamic.md"), "---\ndescription: Dynamic prompt\n---\nDynamic prompt content")

    def factory(pi: ExtensionAPI) -> None:
        def on_discover(_event, _ctx) -> ResourcesDiscoverResult:
            return ResourcesDiscoverResult(
                skill_paths=[str(contributed / "SKILL.md")],
                prompt_paths=[str(contributed / "dynamic.md")],
                theme_paths=[str(contributed / "dynamic.json")],
            )

        pi.on("resources_discover", on_discover)

    loaded = await load_extension_factories([NamedInlineExtension(name="dynamic", factory=factory)], str(tmp_path))
    assert loaded.errors == []

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    auth_storage = AuthStorage.create(str(agent_dir / "auth.json"))
    await auth_storage.set("anthropic", Credential(type="api_key", key="test-key"))
    model_runtime = await ModelRuntime.create(credentials=auth_storage, agent_dir=str(agent_dir))
    loader = _mkloader(tmp_path, agent_dir=agent_dir)
    loader.reload()

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            model=model_runtime.get_model("anthropic", "claude-sonnet-4-5"),
            model_runtime=model_runtime,
            settings_manager=SettingsManager.create(str(tmp_path), str(agent_dir)),
            session_manager=SessionManager.in_memory(),
            resource_loader=loader,
            extensions=loaded.extensions,
        )
    )
    session = result.session
    try:
        assert [s.name for s in loader.get_skills().skills] == []
        assert "dynamic-skill" not in session.system_prompt

        await session.bind_extensions()

        skill = next((s for s in loader.get_skills().skills if s.name == "dynamic-skill"), None)
        assert skill is not None
        assert skill.source_info is not None
        assert skill.source_info.source == "extension:inline:dynamic"
        prompt = next((p for p in loader.get_prompts()[0] if p.name == "dynamic"), None)
        assert prompt is not None
        assert prompt.source_info is not None
        assert prompt.source_info.source == "extension:inline:dynamic"

        # The system prompt embeds the skill list, so it has to be rebuilt.
        assert "dynamic-skill" in session.system_prompt
    finally:
        session.dispose()


# --------------------------------------------------------------------------
# Deliberately-omitted surfaces
# --------------------------------------------------------------------------


@pytest.mark.skip(
    reason="`should load user extensions before trust and reuse them after trust resolves`: "
    "`reload({ resolveProjectTrust })` is a `DefaultResourceLoader` two-pass load. Here trust is "
    "resolved in `cli/entry.py::resolve_startup_trust` before extensions load at all (see "
    "`core/project_trust.py`'s module docstring), so there is no second pass to reuse."
)
def test_loads_user_extensions_before_trust_and_reuses_them_after() -> None:
    pass


@pytest.mark.skip(
    reason="`should keep package metadata for skills, prompts, and themes`: the package half "
    "needs `DefaultResourceLoader`'s `resourceMetadataByPath`, built by its own npm-package "
    "auto-discovery pass, and the theme half needs theme loading. This port's `ResourceLoader` "
    "has neither (see its module docstring); packages are resolved by `PackageManager.resolve()` "
    "instead, and its skill/prompt/theme source-info tagging is pinned by "
    "`test_package_manager.py`. The extension half of the same case is covered by "
    "`test_extend_resources_loads_skills_and_prompts_with_extension_metadata`."
)
def test_extend_resources_keeps_package_metadata() -> None:
    pass


@pytest.mark.skip(
    reason="`should apply skillsOverride`: `DefaultResourceLoaderOptions.skillsOverride` is a "
    "dependency-injection hook that replaces skill discovery wholesale. "
    "`ResourceLoaderOptions` exposes no override callables. (`systemPromptOverride` is covered: "
    "`ResourceLoaderOptions.system_prompt` takes literal text, pinned by "
    "`test_does_not_expose_literal_system_prompt_text_as_a_source`.)"
)
def test_applies_skills_override() -> None:
    pass


@pytest.mark.skip(
    reason="`should prefer explicit CLI extensions over discovered extensions when commands and "
    "tools conflict`: the CLI-first ordering comes from `DefaultResourceLoader.reload()` merging "
    "`cliEnabledExtensions` ahead of discovered ones. `discover_and_load_extensions()` appends "
    "configured paths last, matching TypeScript's own `discoverAndLoadExtensions()`; nothing in "
    "this port re-orders them, because no module here owns the CLI extension-path wiring."
)
def test_prefers_explicit_cli_extensions_over_discovered() -> None:
    pass
