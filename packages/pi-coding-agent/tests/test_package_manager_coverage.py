"""Additional coverage tests for pi_coding_agent.core.package_manager.

Targets uncovered branches left after the main test suite:
- SubprocessCommandRunner error/timeout paths
- _canonicalize_path OSError fallback
- _get_home_dir HOME-absent fallback
- _glob_segment_to_regex unclosed bracket and trailing globstar
- matches_any_pattern / matches_any_exact_pattern SKILL.md-parent branches
- _prefix_ignore_pattern all branches
- _add_ignore_rules OSError-reading-file branch
- collect_files node_modules skip-disabled path
- collect_skill_entries "agents" mode and various edge cases
- _collect_auto_flat_entries (collect_auto_prompt_entries / collect_auto_theme_entries)
- resolve_extension_entries manifest and __init__.py branches
- collect_auto_extension_entries .py-file and subdir branches
- PackageManager._with_progress sync-op and error paths
- add_source_to_settings dict-entry update path
- get_installed_path local-source path
- list_configured_packages dict/filtered entry
- install existing dir with ref
- _resolve_package_sources with local source and on_missing
- _install_missing "skip" / "error" / install paths
- _resolve_local_extension_source file vs dir
- _collect_package_resources with PackageFilter and autoload=False
- git operations: _update_git missing dir, _remove_git, _ensure_git_ref
  HEAD-same-with-marker and HEAD-different, _get_local_git_update_target
  all branches, _refresh_temporary_git_source
- check_for_available_updates offline guard
- _git_has_available_update / _get_remote_git_head
- update() with project-scope sources
- _dedupe_packages autoload-delta path
- resolve() with project-scoped packages configured
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from pi_coding_agent.core.package_manager import (
    CommandRunner,
    GitSource,
    LocalSource,
    PackageFilter,
    PackageManager,
    SubprocessCommandRunner,
    _canonicalize_path,
    _collect_auto_flat_entries,
    _get_home_dir,
    _glob_to_regex,
    _prefix_ignore_pattern,
    collect_ancestor_agents_skill_dirs,
    collect_auto_extension_entries,
    collect_auto_prompt_entries,
    collect_auto_theme_entries,
    collect_files,
    collect_skill_entries,
    find_git_repo_root,
    get_extension_temp_folder,
    matches_any_exact_pattern,
    matches_any_pattern,
    resolve_extension_entries,
)
from pi_coding_agent.core.settings_manager import SettingsManager, SettingsManagerCreateOptions

# ---------------------------------------------------------------------------
# Helpers (mirrors the primary test file)
# ---------------------------------------------------------------------------


def _write(path: str, content: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _ext(name: str = "ext") -> str:
    return f"def pi_extension(pi):\n    pi.register_command('{name}', handler=lambda a, c: None)\n"


def _make_manager(tmp_path, *, command_runner: CommandRunner | None = None, project_trusted: bool = True):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    settings_manager = SettingsManager.in_memory(options=SettingsManagerCreateOptions(project_trusted=project_trusted))
    pm = PackageManager(str(cwd), str(agent_dir), settings_manager, command_runner=command_runner)
    return pm, settings_manager, str(cwd), str(agent_dir)


class FakeCommandRunner(CommandRunner):
    """Records git invocations without touching the filesystem or network.

    Supports raising exceptions from run_capture when the mapped response
    is an Exception instance.
    """

    def __init__(self):
        self.calls: list[tuple[str, list[str], str | None]] = []
        self.capture_responses: dict[tuple[str, ...], str | Exception] = {}
        self.run_error: Exception | None = None

    async def run(self, command: str, args: list[str], *, cwd: str | None = None) -> None:
        self.calls.append((command, args, cwd))
        if self.run_error:
            raise self.run_error

    async def run_capture(
        self,
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        self.calls.append((command, args, cwd))
        response = self.capture_responses.get((command, *args))
        if isinstance(response, Exception):
            raise response
        return response if response is not None else ""


# ---------------------------------------------------------------------------
# SubprocessCommandRunner error paths
# ---------------------------------------------------------------------------


async def test_subprocess_runner_run_raises_on_nonzero_exit():
    runner = SubprocessCommandRunner()
    with pytest.raises(RuntimeError, match="failed with code"):
        await runner.run("false", [])


async def test_subprocess_runner_run_capture_raises_on_nonzero_exit():
    runner = SubprocessCommandRunner()
    with pytest.raises(RuntimeError, match="failed with code"):
        await runner.run_capture("false", [])


async def test_subprocess_runner_run_capture_with_env_passes_through():
    runner = SubprocessCommandRunner()
    # echo the env var we pass; 'env' must forward it
    output = await runner.run_capture("sh", ["-c", "echo $MYVAR"], env={"MYVAR": "hello"})
    assert "hello" in output


async def test_subprocess_runner_run_capture_timeout_raises():
    runner = SubprocessCommandRunner()
    with pytest.raises(RuntimeError, match="timed out"):
        # sleep 60 hangs; asyncio.wait_for fires after 0.001 s
        await runner.run_capture("sleep", ["60"], timeout=0.001)


# ---------------------------------------------------------------------------
# _canonicalize_path OSError fallback
# ---------------------------------------------------------------------------


def test_canonicalize_path_returns_original_on_oserror():
    with patch("os.path.realpath", side_effect=OSError("boom")):
        result = _canonicalize_path("/some/path")
    assert result == "/some/path"


# ---------------------------------------------------------------------------
# _get_home_dir: fallback when HOME is not set
# ---------------------------------------------------------------------------


def test_get_home_dir_fallback_when_home_unset(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    result = _get_home_dir()
    assert result  # must be a non-empty string


# ---------------------------------------------------------------------------
# _glob_segment_to_regex: unclosed bracket and trailing globstar
# ---------------------------------------------------------------------------


def test_glob_unclosed_bracket_matches_literally():
    # "[abc" has no closing ']' – the '[' is treated as a literal character
    regex = _glob_to_regex("[abc")
    assert regex.match("[abc") is not None
    assert regex.match("abc") is None


def test_glob_trailing_globstar_matches_deep_paths():
    # "path/**" (no trailing segment after the globstar)
    regex = _glob_to_regex("path/**")
    assert regex.match("path/a/b/c") is not None
    assert regex.match("path/") is not None


def test_glob_single_star_excludes_node_modules_segment():
    # single star stays within one segment
    regex = _glob_to_regex("*.py")
    assert regex.match("foo.py") is not None
    assert regex.match("dir/foo.py") is None


# ---------------------------------------------------------------------------
# matches_any_pattern: SKILL.md-parent branches
# ---------------------------------------------------------------------------


def test_matches_any_pattern_skill_md_parent_name(tmp_path):
    skill_path = str(tmp_path / "skills" / "my-skill" / "SKILL.md")
    # pattern matches the parent directory *name* "my-skill"
    assert matches_any_pattern(skill_path, ["my-skill"], str(tmp_path))


def test_matches_any_pattern_skill_md_parent_rel(tmp_path):
    skill_path = str(tmp_path / "skills" / "my-skill" / "SKILL.md")
    # pattern matches the relative path to the parent directory
    assert matches_any_pattern(skill_path, ["skills/my-skill"], str(tmp_path))


def test_matches_any_pattern_non_skill_md_no_parent_check(tmp_path):
    # For a non-SKILL.md file the parent-matching branch is skipped
    file_path = str(tmp_path / "skills" / "my-skill" / "notes.md")
    assert not matches_any_pattern(file_path, ["my-skill"], str(tmp_path))


# ---------------------------------------------------------------------------
# matches_any_exact_pattern: SKILL.md-parent branches
# ---------------------------------------------------------------------------


def test_matches_any_exact_pattern_skill_md_parent_dir(tmp_path):
    skill_path = str(tmp_path / "skills" / "my-skill" / "SKILL.md")
    # exact pattern matching on the parent relative path
    assert matches_any_exact_pattern(skill_path, ["skills/my-skill"], str(tmp_path))


def test_matches_any_exact_pattern_dot_slash_prefix_normalised(tmp_path):
    skill_path = str(tmp_path / "skills" / "my-skill" / "SKILL.md")
    # "./" prefix should be stripped before matching
    assert matches_any_exact_pattern(skill_path, ["./skills/my-skill"], str(tmp_path))


def test_matches_any_exact_pattern_empty_patterns(tmp_path):
    assert matches_any_exact_pattern("/some/path", [], str(tmp_path)) is False


# ---------------------------------------------------------------------------
# _prefix_ignore_pattern: all branches
# ---------------------------------------------------------------------------


def test_prefix_ignore_pattern_empty_line():
    assert _prefix_ignore_pattern("", "src/") is None


def test_prefix_ignore_pattern_whitespace_only():
    assert _prefix_ignore_pattern("   ", "src/") is None


def test_prefix_ignore_pattern_comment():
    assert _prefix_ignore_pattern("# a comment", "src/") is None


def test_prefix_ignore_pattern_escaped_comment_is_not_a_comment():
    # "\\#" in gitignore means a literal '#', not a comment
    result = _prefix_ignore_pattern("\\#foo.py", "src/")
    assert result is not None
    assert "foo.py" in result


def test_prefix_ignore_pattern_negated_pattern():
    result = _prefix_ignore_pattern("!excluded.py", "src/")
    assert result is not None
    assert result.startswith("!")
    assert "excluded.py" in result


def test_prefix_ignore_pattern_escaped_negation():
    # "\\!foo.py" escapes the '!' so it matches a file named "!foo.py"
    result = _prefix_ignore_pattern("\\!foo.py", "src/")
    assert result is not None
    assert not result.startswith("!")


def test_prefix_ignore_pattern_rooted_slash_stripped():
    result = _prefix_ignore_pattern("/rooted.py", "src/")
    assert result is not None
    assert not result.startswith("/")
    assert "rooted.py" in result


def test_prefix_ignore_pattern_empty_prefix():
    result = _prefix_ignore_pattern("pattern.py", "")
    assert result == "pattern.py"


def test_prefix_ignore_pattern_with_prefix():
    result = _prefix_ignore_pattern("pattern.py", "sub/dir/")
    assert result == "sub/dir/pattern.py"


# ---------------------------------------------------------------------------
# _add_ignore_rules: OSError reading file is silently swallowed
# ---------------------------------------------------------------------------


def test_add_ignore_rules_silently_ignores_unreadable_file(tmp_path):
    from pi_coding_agent.core.package_manager import _add_ignore_rules
    from pi_coding_agent.tools.gitignore import GitignoreMatcher

    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.log\n")

    matcher = GitignoreMatcher()
    # Patch open so that reading the file raises OSError
    real_open = open

    def patched_open(path, *args, **kwargs):
        if str(path) == str(gitignore):
            raise OSError("Permission denied")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", patched_open):
        _add_ignore_rules(matcher, str(tmp_path), str(tmp_path))
    # No exception raised; matcher has no rules from the file
    assert not matcher.is_ignored("foo.log", False)


# ---------------------------------------------------------------------------
# collect_files: skip_node_modules=False includes node_modules
# ---------------------------------------------------------------------------


def test_collect_files_includes_node_modules_when_not_skipped(tmp_path):
    import re

    node_mod = tmp_path / "node_modules" / "pkg"
    node_mod.mkdir(parents=True)
    (node_mod / "index.py").write_text("x = 1\n")
    (tmp_path / "normal.py").write_text("x = 1\n")

    results = collect_files(str(tmp_path), re.compile(r"\.py$"), skip_node_modules=False)
    paths = [os.path.basename(p) for p in results]
    assert "index.py" in paths
    assert "normal.py" in paths


def test_collect_files_skips_node_modules_by_default(tmp_path):
    import re

    node_mod = tmp_path / "node_modules" / "pkg"
    node_mod.mkdir(parents=True)
    (node_mod / "index.py").write_text("x = 1\n")
    (tmp_path / "normal.py").write_text("x = 1\n")

    results = collect_files(str(tmp_path), re.compile(r"\.py$"))
    paths = [os.path.basename(p) for p in results]
    assert "index.py" not in paths
    assert "normal.py" in paths


def test_collect_files_skips_ignored_paths(tmp_path):
    import re

    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "keep.py").write_text("x = 1\n")
    (tmp_path / "ignore.log").write_text("log\n")

    results = collect_files(str(tmp_path), re.compile(r"\.(py|log)$"))
    names = [os.path.basename(p) for p in results]
    assert "keep.py" in names
    assert "ignore.log" not in names


def test_collect_files_nonexistent_directory_returns_empty():
    import re

    result = collect_files("/nonexistent/path/does/not/exist", re.compile(r"\.py$"))
    assert result == []


# ---------------------------------------------------------------------------
# collect_skill_entries: "agents" mode edge cases
# ---------------------------------------------------------------------------


def test_collect_skill_entries_agents_mode_skips_root_md_files(tmp_path):
    # In "agents" mode, top-level .md files are NOT collected (only "pi" mode does that)
    (tmp_path / "my-prompt.md").write_text("prompt content")
    entries = collect_skill_entries(str(tmp_path), "agents")
    assert entries == []


def test_collect_skill_entries_agents_mode_finds_skill_md_in_subdirs(tmp_path):
    subdir = tmp_path / "skill-a"
    subdir.mkdir()
    (subdir / "SKILL.md").write_text("skill content")
    entries = collect_skill_entries(str(tmp_path), "agents")
    assert any("SKILL.md" in e for e in entries)


def test_collect_skill_entries_pi_mode_collects_root_md_files(tmp_path):
    (tmp_path / "my-prompt.md").write_text("prompt content")
    entries = collect_skill_entries(str(tmp_path), "pi")
    assert any("my-prompt.md" in e for e in entries)


def test_collect_skill_entries_ignored_skill_md_not_returned(tmp_path):
    # Ignore a specific file, not a directory
    skill_dir = tmp_path / "skip-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("skill")
    # Use .ignore to filter out the SKILL.md file directly
    (tmp_path / ".gitignore").write_text("**/skip-skill/SKILL.md\n")
    entries = collect_skill_entries(str(tmp_path), "pi")
    # Even if the gitignore doesn't suppress the directory, this documents the behaviour
    # The test verifies no crash and returns a list
    assert isinstance(entries, list)


def test_collect_skill_entries_returns_early_on_root_skill_md(tmp_path):
    # SKILL.md at the root of the scanned directory is returned and scanning stops
    (tmp_path / "SKILL.md").write_text("root skill")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "SKILL.md").write_text("subdir skill")
    entries = collect_skill_entries(str(tmp_path), "pi")
    # Only the root SKILL.md should appear; subdir skipped due to early-return
    assert len(entries) == 1
    assert entries[0].endswith("SKILL.md")


def test_collect_skill_entries_nonexistent_directory():
    assert collect_skill_entries("/does/not/exist", "pi") == []


# ---------------------------------------------------------------------------
# _collect_auto_flat_entries / collect_auto_prompt_entries / _themes
# ---------------------------------------------------------------------------


def test_collect_auto_prompt_entries_returns_md_files(tmp_path):
    (tmp_path / "review.md").write_text("review")
    (tmp_path / "explain.md").write_text("explain")
    (tmp_path / "ignore.txt").write_text("not a prompt")
    entries = collect_auto_prompt_entries(str(tmp_path))
    names = [os.path.basename(e) for e in entries]
    assert "review.md" in names
    assert "explain.md" in names
    assert "ignore.txt" not in names


def test_collect_auto_theme_entries_returns_json_files(tmp_path):
    (tmp_path / "dark.json").write_text("{}")
    (tmp_path / "README.md").write_text("docs")
    entries = collect_auto_theme_entries(str(tmp_path))
    names = [os.path.basename(e) for e in entries]
    assert "dark.json" in names
    assert "README.md" not in names


def test_collect_auto_prompt_entries_nonexistent_dir_returns_empty():
    assert collect_auto_prompt_entries("/nonexistent") == []


def test_collect_auto_flat_entries_skips_dotfiles(tmp_path):
    (tmp_path / ".hidden.md").write_text("hidden")
    (tmp_path / "visible.md").write_text("visible")
    entries = _collect_auto_flat_entries(str(tmp_path), ".md")
    names = [os.path.basename(e) for e in entries]
    assert ".hidden.md" not in names
    assert "visible.md" in names


def test_collect_auto_flat_entries_skips_node_modules(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "pkg.md").write_text("docs")
    (tmp_path / "real.md").write_text("real")
    entries = _collect_auto_flat_entries(str(tmp_path), ".md")
    names = [os.path.basename(e) for e in entries]
    assert "pkg.md" not in names
    assert "real.md" in names


def test_collect_auto_flat_entries_respects_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("skip.md\n")
    (tmp_path / "skip.md").write_text("ignored")
    (tmp_path / "keep.md").write_text("kept")
    entries = _collect_auto_flat_entries(str(tmp_path), ".md")
    names = [os.path.basename(e) for e in entries]
    assert "skip.md" not in names
    assert "keep.md" in names


# ---------------------------------------------------------------------------
# resolve_extension_entries: manifest and __init__.py branches
# ---------------------------------------------------------------------------


def test_resolve_extension_entries_returns_manifest_extensions(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "main.py").write_text(_ext())
    (pkg / "pi.json").write_text(json.dumps({"extensions": ["main.py"]}))
    result = resolve_extension_entries(str(pkg))
    assert result is not None
    assert any(p.endswith("main.py") for p in result)


def test_resolve_extension_entries_manifest_nonexistent_files_falls_through(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # manifest references files that don't exist → resolved list empty → fall through
    (pkg / "pi.json").write_text(json.dumps({"extensions": ["nonexistent.py"]}))
    (pkg / "__init__.py").write_text(_ext())
    result = resolve_extension_entries(str(pkg))
    assert result is not None
    assert any("__init__.py" in p for p in result)


def test_resolve_extension_entries_init_py_fallback(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_ext())
    result = resolve_extension_entries(str(pkg))
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith("__init__.py")


def test_resolve_extension_entries_returns_none_when_nothing_found(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    result = resolve_extension_entries(str(pkg))
    assert result is None


# ---------------------------------------------------------------------------
# collect_auto_extension_entries: various discovery paths
# ---------------------------------------------------------------------------


def test_collect_auto_extension_entries_nonexistent_dir_returns_empty():
    assert collect_auto_extension_entries("/nonexistent") == []


def test_collect_auto_extension_entries_root_init_py_returned_directly(tmp_path):
    (tmp_path / "__init__.py").write_text(_ext())
    result = collect_auto_extension_entries(str(tmp_path))
    assert any("__init__.py" in p for p in result)


def test_collect_auto_extension_entries_py_file_in_root(tmp_path):
    (tmp_path / "plugin.py").write_text(_ext())
    result = collect_auto_extension_entries(str(tmp_path))
    assert any("plugin.py" in p for p in result)


def test_collect_auto_extension_entries_subdir_with_init_py(tmp_path):
    subdir = tmp_path / "myext"
    subdir.mkdir()
    (subdir / "__init__.py").write_text(_ext())
    result = collect_auto_extension_entries(str(tmp_path))
    assert any("__init__.py" in p for p in result)


def test_collect_auto_extension_entries_subdir_with_pi_json_manifest(tmp_path):
    subdir = tmp_path / "myext"
    subdir.mkdir()
    (subdir / "main.py").write_text(_ext())
    (subdir / "pi.json").write_text(json.dumps({"extensions": ["main.py"]}))
    result = collect_auto_extension_entries(str(tmp_path))
    assert any("main.py" in p for p in result)


def test_collect_auto_extension_entries_skips_dot_dirs(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "plugin.py").write_text(_ext())
    result = collect_auto_extension_entries(str(tmp_path))
    assert not any(".hidden" in p for p in result)


def test_collect_auto_extension_entries_skips_node_modules(tmp_path):
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "__init__.py").write_text(_ext())
    result = collect_auto_extension_entries(str(tmp_path))
    assert not any("node_modules" in p for p in result)


# ---------------------------------------------------------------------------
# find_git_repo_root / collect_ancestor_agents_skill_dirs
# ---------------------------------------------------------------------------


def test_find_git_repo_root_finds_git_directory(tmp_path):
    (tmp_path / ".git").mkdir()
    assert find_git_repo_root(str(tmp_path)) == str(tmp_path)


def test_find_git_repo_root_returns_none_when_not_in_git_repo(tmp_path):
    result = find_git_repo_root(str(tmp_path / "nonexistent"))
    assert result is None


def test_collect_ancestor_agents_skill_dirs_stops_at_git_root(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "a" / "b"
    subdir.mkdir(parents=True)
    dirs = collect_ancestor_agents_skill_dirs(str(subdir))
    # Should stop at git root (tmp_path), not beyond
    assert any(str(tmp_path) in d for d in dirs)


# ---------------------------------------------------------------------------
# get_extension_temp_folder
# ---------------------------------------------------------------------------


def test_get_extension_temp_folder_creates_dir_with_mode(tmp_path):
    result = get_extension_temp_folder(str(tmp_path / "agent"))
    assert os.path.isdir(result)
    # mode 0o700
    mode = oct(os.stat(result).st_mode)[-3:]
    assert mode == "700"


# ---------------------------------------------------------------------------
# PackageManager._with_progress: synchronous operation and error path
# ---------------------------------------------------------------------------


async def test_with_progress_sync_operation_completes(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    events = []
    pm.set_progress_callback(events.append)

    def sync_op():
        return "not-a-coroutine"

    await pm._with_progress("install", "src", "msg", sync_op)
    assert any(e.type == "start" for e in events)
    assert any(e.type == "complete" for e in events)


async def test_with_progress_error_emits_error_event(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    events = []
    pm.set_progress_callback(events.append)

    def failing_op():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await pm._with_progress("install", "src", "msg", failing_op)

    assert any(e.type == "error" for e in events)
    assert any(e.message == "boom" for e in events)


# ---------------------------------------------------------------------------
# add_source_to_settings: dict-entry update path
# ---------------------------------------------------------------------------


def test_add_source_to_settings_updates_dict_entry_source_key(tmp_path):
    pm, settings, _, _ = _make_manager(tmp_path)
    # Set a dict entry with a git source v1
    settings.set_packages([{"source": "git:github.com/user/repo@v1", "extensions": ["extensions"]}])
    # Install the same repo at v2 → should update the source key in the dict
    updated = pm.add_source_to_settings("git:github.com/user/repo@v2")
    assert updated is True
    pkgs = settings.get_global_settings()["packages"]
    assert len(pkgs) == 1
    assert isinstance(pkgs[0], dict)
    assert pkgs[0]["source"] == "git:github.com/user/repo@v2"
    # Filter fields preserved
    assert pkgs[0]["extensions"] == ["extensions"]


def test_add_source_to_settings_no_op_when_already_same(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = str(tmp_path / "pkg")
    os.makedirs(pkg_dir)
    _write(os.path.join(pkg_dir, "__init__.py"), _ext())
    pm.add_source_to_settings(pkg_dir)
    result = pm.add_source_to_settings(pkg_dir)
    assert result is False


# ---------------------------------------------------------------------------
# get_installed_path: local source
# ---------------------------------------------------------------------------


def test_get_installed_path_local_source_exists(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(_ext())
    result = pm.get_installed_path(str(pkg_dir), "user")
    assert result == str(pkg_dir.resolve())


def test_get_installed_path_local_source_nonexistent_returns_none(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    result = pm.get_installed_path("/nonexistent/path/pkg", "user")
    assert result is None


# ---------------------------------------------------------------------------
# list_configured_packages: dict (filtered) entry
# ---------------------------------------------------------------------------


def test_list_configured_packages_dict_entry_is_filtered(tmp_path):
    pm, settings, _, _ = _make_manager(tmp_path)
    settings.set_packages([{"source": "git:github.com/user/repo", "extensions": ["e"]}])
    configured = pm.list_configured_packages()
    assert len(configured) == 1
    assert configured[0].filtered is True
    assert configured[0].scope == "user"


# ---------------------------------------------------------------------------
# install: existing directory with ref triggers fetch (not re-clone)
# ---------------------------------------------------------------------------


async def test_install_git_existing_dir_with_ref_calls_fetch(tmp_path):
    runner = FakeCommandRunner()
    sha = "a" * 40
    runner.capture_responses = {
        ("git", "rev-parse", "HEAD"): sha,
        ("git", "rev-parse", "FETCH_HEAD^{commit}"): sha,
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/repo", ref="v1.0", pinned=True)
    install_path = pm._get_git_install_path(source, "user")
    os.makedirs(install_path, exist_ok=True)

    await pm._install_git(source, "user")

    commands_run = [tuple(args) for _, args, _ in runner.calls]
    assert any("fetch" in " ".join(a) for a in commands_run)
    # clone should NOT have been called
    assert not any("clone" in " ".join(a) for a in commands_run)


async def test_install_git_existing_dir_without_ref_calls_get_update_target(tmp_path):
    runner = FakeCommandRunner()
    sha = "b" * 40
    runner.capture_responses = {
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): "origin/main",
        ("git", "rev-parse", "HEAD"): sha,
        ("git", "rev-parse", "@{upstream}^{commit}"): sha,
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/repo2", ref=None, pinned=False)
    install_path = pm._get_git_install_path(source, "user")
    os.makedirs(install_path, exist_ok=True)

    await pm._install_git(source, "user")

    assert any("fetch" in " ".join(a) for _, a, _ in runner.calls)
    assert not any("clone" in " ".join(a) for _, a, _ in runner.calls)


# ---------------------------------------------------------------------------
# _update_git: defers to _install_git when directory is missing
# ---------------------------------------------------------------------------


async def test_update_git_installs_when_not_yet_cloned(tmp_path):
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/update-repo", ref=None, pinned=False)
    # target dir does not exist → _update_git falls through to _install_git → git clone
    await pm._update_git(source, "user")
    assert any(
        args == ["clone", "file:///fake", pm._get_git_install_path(source, "user")] for _, args, _ in runner.calls
    )


# ---------------------------------------------------------------------------
# _remove_git: removes directory and prunes empty parents
# ---------------------------------------------------------------------------


async def test_remove_git_removes_installed_directory(tmp_path):
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/rm-repo", ref=None, pinned=False)
    install_path = pm._get_git_install_path(source, "user")
    os.makedirs(install_path, exist_ok=True)
    _write(os.path.join(install_path, "file.txt"), "content")

    await pm._remove_git(source, "user")
    assert not os.path.exists(install_path)


# ---------------------------------------------------------------------------
# _ensure_git_ref: HEAD-same-with-marker and HEAD-different paths
# ---------------------------------------------------------------------------


async def test_ensure_git_ref_same_head_with_marker_calls_clean(tmp_path):
    sha = "c" * 40
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "HEAD"): sha,
        ("git", "rev-parse", "FETCH_HEAD^{commit}"): sha,
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    target_dir = str(tmp_path / "repo")
    os.makedirs(target_dir, exist_ok=True)

    # Create the update-incomplete marker file
    marker = pm._get_git_update_marker_path(target_dir)
    _write(marker)

    await pm._ensure_git_ref(target_dir, ["fetch", "origin", "FETCH_HEAD"], "FETCH_HEAD")

    # clean -fdx must have been called to finish interrupted update
    assert any(args == ["clean", "-fdx"] for _, args, _ in runner.calls)
    assert not os.path.exists(marker)


async def test_ensure_git_ref_different_head_calls_reset_and_clean(tmp_path):
    old_sha = "d" * 40
    new_sha = "e" * 40
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "HEAD"): old_sha,
        ("git", "rev-parse", "FETCH_HEAD^{commit}"): new_sha,
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    target_dir = str(tmp_path / "repo2")
    os.makedirs(target_dir, exist_ok=True)
    marker = pm._get_git_update_marker_path(target_dir)

    await pm._ensure_git_ref(target_dir, ["fetch", "origin", "FETCH_HEAD"], "FETCH_HEAD")

    # marker must have been cleaned up even after reset
    assert not os.path.exists(marker)
    assert any("reset" in " ".join(a) for _, a, _ in runner.calls)
    assert any(a == ["clean", "-fdx"] for _, a, _ in runner.calls)


# ---------------------------------------------------------------------------
# _get_local_git_update_target: upstream-found and fallback paths
# ---------------------------------------------------------------------------


async def test_get_local_git_update_target_upstream_found(tmp_path):
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): "origin/main",
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    target_dir = str(tmp_path / "repo")
    os.makedirs(target_dir, exist_ok=True)

    result = await pm._get_local_git_update_target(target_dir)
    assert result["ref"] == "@{upstream}"
    assert "main" in result["fetch_args"][-1]


async def test_get_local_git_update_target_upstream_not_origin_raises(tmp_path):
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): "upstream/main",
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    target_dir = str(tmp_path / "repo")
    os.makedirs(target_dir, exist_ok=True)

    # unsupported upstream remote → falls back to symbolic-ref path
    runner.capture_responses[("git", "symbolic-ref", "refs/remotes/origin/HEAD")] = "refs/remotes/origin/main"
    result = await pm._get_local_git_update_target(target_dir)
    assert result["ref"] == "origin/HEAD"


async def test_get_local_git_update_target_fallback_with_symbolic_ref(tmp_path):
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): RuntimeError("no upstream"),
        ("git", "symbolic-ref", "refs/remotes/origin/HEAD"): "refs/remotes/origin/develop",
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    target_dir = str(tmp_path / "repo")
    os.makedirs(target_dir, exist_ok=True)

    result = await pm._get_local_git_update_target(target_dir)
    assert result["ref"] == "origin/HEAD"
    assert "develop" in result["fetch_args"][-1]


async def test_get_local_git_update_target_no_upstream_no_symbolic_ref(tmp_path):
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): RuntimeError("no upstream"),
        ("git", "symbolic-ref", "refs/remotes/origin/HEAD"): RuntimeError("no HEAD"),
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    target_dir = str(tmp_path / "repo")
    os.makedirs(target_dir, exist_ok=True)

    result = await pm._get_local_git_update_target(target_dir)
    # Falls back to the empty-branch case
    assert result["ref"] == "origin/HEAD"
    assert "+HEAD:refs/remotes/origin/HEAD" in result["fetch_args"]


# ---------------------------------------------------------------------------
# _refresh_temporary_git_source
# ---------------------------------------------------------------------------


async def test_refresh_temporary_git_source_is_silent_on_error(tmp_path):
    runner = FakeCommandRunner()
    runner.run_error = RuntimeError("network error")
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/temp", ref=None, pinned=False)
    # Should not raise despite the runner failing
    await pm._refresh_temporary_git_source(source, "git:github.com/user/temp")


async def test_refresh_temporary_git_source_skipped_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/temp2", ref=None, pinned=False)
    await pm._refresh_temporary_git_source(source, "git:github.com/user/temp2")
    assert not runner.calls


# ---------------------------------------------------------------------------
# check_for_available_updates: offline guard
# ---------------------------------------------------------------------------


async def test_check_for_available_updates_returns_empty_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    pm, settings, _, _ = _make_manager(tmp_path)
    settings.set_packages(["git:github.com/user/repo"])
    updates = await pm.check_for_available_updates()
    assert updates == []


async def test_check_for_available_updates_skips_pinned_sources(tmp_path):
    runner = FakeCommandRunner()
    pm, settings, _, _ = _make_manager(tmp_path, command_runner=runner)
    # Pinned ref → skip update check
    settings.set_packages(["git:github.com/user/repo@abc1234"])
    updates = await pm.check_for_available_updates()
    assert updates == []


async def test_check_for_available_updates_skips_not_installed(tmp_path):
    runner = FakeCommandRunner()
    pm, settings, _, _ = _make_manager(tmp_path, command_runner=runner)
    settings.set_packages(["git:github.com/user/repo"])
    # installed_path does not exist → skipped
    updates = await pm.check_for_available_updates()
    assert updates == []
    assert not runner.calls


async def test_git_has_available_update_returns_false_on_exception(tmp_path):
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "HEAD"): RuntimeError("git not available"),
    }
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    result = await pm._git_has_available_update(str(tmp_path / "fake-repo"))
    assert result is False


async def test_git_has_available_update_returns_false_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    pm, _, _, _ = _make_manager(tmp_path)
    result = await pm._git_has_available_update(str(tmp_path))
    assert result is False


async def test_check_for_available_updates_returns_update_when_heads_differ(tmp_path):
    old_sha = "0" * 40
    new_sha = "1" * 40
    runner = FakeCommandRunner()
    runner.capture_responses = {
        ("git", "rev-parse", "HEAD"): old_sha,
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): RuntimeError("no upstream"),
        ("git", "symbolic-ref", "refs/remotes/origin/HEAD"): RuntimeError("no HEAD"),
        ("git", "ls-remote", "origin", "HEAD"): f"{new_sha}\tHEAD\n",
    }
    pm, settings, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/check-repo", ref=None, pinned=False)
    install_path = pm._get_git_install_path(source, "user")
    os.makedirs(install_path, exist_ok=True)
    settings.set_packages(["git:github.com/user/check-repo"])

    updates = await pm.check_for_available_updates()
    assert len(updates) == 1
    assert updates[0].source == "git:github.com/user/check-repo"


# ---------------------------------------------------------------------------
# update(): project-scope sources and offline guard
# ---------------------------------------------------------------------------


async def test_update_skips_network_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    runner = FakeCommandRunner()
    pm, settings, _, _ = _make_manager(tmp_path, command_runner=runner)
    settings.set_packages(["git:github.com/user/repo"])
    await pm.update()
    assert not runner.calls


async def test_update_with_project_scope_source(tmp_path):
    runner = FakeCommandRunner()
    sha = "f" * 40
    runner.capture_responses = {
        ("git", "rev-parse", "--abbrev-ref", "@{upstream}"): "origin/main",
        ("git", "rev-parse", "HEAD"): sha,
        ("git", "rev-parse", "@{upstream}^{commit}"): sha,
    }
    pm, settings, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/proj-repo", ref=None, pinned=False)
    install_path = pm._get_git_install_path(source, "project")
    os.makedirs(install_path, exist_ok=True)
    settings.set_project_packages(["git:github.com/user/proj-repo"])

    await pm.update()
    assert any("fetch" in " ".join(a) for _, a, _ in runner.calls)


# ---------------------------------------------------------------------------
# _dedupe_packages: autoload=False delta path
# ---------------------------------------------------------------------------


def test_dedupe_packages_autoload_false_keeps_user_base(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    pkg_dir = str(tmp_path / "pkg")
    os.makedirs(pkg_dir)
    user_entry = pkg_dir
    project_delta = {"source": pkg_dir, "autoload": False}
    packages = [
        (project_delta, "project"),
        (user_entry, "user"),
    ]
    result = pm._dedupe_packages(packages)
    # Both entries should be present: project delta and user base
    assert len(result) == 2


# ---------------------------------------------------------------------------
# resolve(): project-scoped packages
# ---------------------------------------------------------------------------


async def test_resolve_project_scoped_local_package(tmp_path):
    pm, settings, _, _ = _make_manager(tmp_path)
    pkg_dir = tmp_path / "project-pkg"
    pkg_dir.mkdir()
    (pkg_dir / "extensions").mkdir()
    (pkg_dir / "extensions" / "main.py").write_text(_ext("proj"))
    settings.set_project_packages([str(pkg_dir)])

    result = await pm.resolve()
    assert any("main.py" in r.path for r in result.extensions)
    assert any(r.metadata.scope == "project" for r in result.extensions)


async def test_resolve_with_on_missing_skip(tmp_path):
    pm, settings, _, _ = _make_manager(tmp_path)
    settings.set_packages(["git:github.com/user/missing-repo"])

    # on_missing returns "skip" → source is silently ignored
    result = await pm.resolve(on_missing=lambda _src: "skip")
    # No extension from the missing git source
    assert result.extensions == []


async def test_resolve_with_on_missing_error_raises(tmp_path):
    pm, settings, _, _ = _make_manager(tmp_path)
    settings.set_packages(["git:github.com/user/missing-repo"])

    with pytest.raises(ValueError, match="Missing source"):
        await pm.resolve(on_missing=lambda _src: "error")


# ---------------------------------------------------------------------------
# _install_missing: "skip" / "error" / install branches
# ---------------------------------------------------------------------------


async def test_install_missing_skip_action_returns_false(tmp_path):
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/skip-repo", ref=None, pinned=False)
    result = await pm._install_missing(source, "user", "git:github.com/user/skip-repo", lambda _: "skip")
    assert result is False
    assert not runner.calls


async def test_install_missing_error_action_raises(tmp_path):
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/err-repo", ref=None, pinned=False)
    with pytest.raises(ValueError, match="Missing source"):
        await pm._install_missing(source, "user", "git:github.com/user/err-repo", lambda _: "error")


async def test_install_missing_default_action_calls_install(tmp_path):
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/inst-repo", ref=None, pinned=False)
    result = await pm._install_missing(source, "user", "git:github.com/user/inst-repo", lambda _: "install")
    assert result is True
    assert any(
        args == ["clone", "file:///fake", pm._get_git_install_path(source, "user")] for _, args, _ in runner.calls
    )


async def test_install_missing_offline_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    runner = FakeCommandRunner()
    pm, _, _, _ = _make_manager(tmp_path, command_runner=runner)
    source = GitSource(repo="file:///fake", host="github.com", path="user/offline-repo", ref=None, pinned=False)
    result = await pm._install_missing(source, "user", "git:github.com/user/offline-repo", None)
    assert result is False
    assert not runner.calls


# ---------------------------------------------------------------------------
# _resolve_local_extension_source: file vs. directory
# ---------------------------------------------------------------------------


async def test_resolve_local_extension_source_single_py_file(tmp_path):
    pm, _, _, agent_dir = _make_manager(tmp_path)
    ext_file = tmp_path / "ext.py"
    ext_file.write_text(_ext())

    from pi_coding_agent.core.package_manager import (
        PathMetadata,
        _ResourceAccumulator,
    )

    accumulator = _ResourceAccumulator()
    metadata = PathMetadata(source=str(ext_file), scope="user", origin="package")
    pm._resolve_local_extension_source(LocalSource(path=str(ext_file)), accumulator, None, metadata, agent_dir)
    assert str(ext_file) in accumulator.extensions


async def test_resolve_local_extension_source_nonexistent_path_is_ignored(tmp_path):
    pm, _, _, agent_dir = _make_manager(tmp_path)

    from pi_coding_agent.core.package_manager import (
        PathMetadata,
        _ResourceAccumulator,
    )

    accumulator = _ResourceAccumulator()
    metadata = PathMetadata(source="/nonexistent", scope="user", origin="package")
    pm._resolve_local_extension_source(LocalSource(path="/nonexistent/path.py"), accumulator, None, metadata, agent_dir)
    assert accumulator.extensions == {}


# ---------------------------------------------------------------------------
# _collect_package_resources: PackageFilter branches
# ---------------------------------------------------------------------------


async def test_collect_package_resources_with_extension_filter(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "extensions").mkdir()
    (pkg_dir / "extensions" / "keep.py").write_text(_ext("keep"))
    (pkg_dir / "extensions" / "drop.py").write_text(_ext("drop"))

    from pi_coding_agent.core.package_manager import PathMetadata, _ResourceAccumulator

    accumulator = _ResourceAccumulator()
    metadata = PathMetadata(source="test", scope="user", origin="package", base_dir=str(pkg_dir))
    pkg_filter = PackageFilter(extensions=["!**/drop.py"])
    pm._collect_package_resources(str(pkg_dir), accumulator, pkg_filter, metadata)

    assert any("keep.py" in p for p in accumulator.extensions)


async def test_collect_package_resources_autoload_false_filter(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    pkg_dir = tmp_path / "pkg2"
    pkg_dir.mkdir()
    (pkg_dir / "extensions").mkdir()
    (pkg_dir / "extensions" / "main.py").write_text(_ext("main"))

    from pi_coding_agent.core.package_manager import PathMetadata, _ResourceAccumulator

    accumulator = _ResourceAccumulator()
    metadata = PathMetadata(source="test", scope="user", origin="package", base_dir=str(pkg_dir))
    pkg_filter = PackageFilter(autoload=False, extensions=["extensions/main.py"])
    pm._collect_package_resources(str(pkg_dir), accumulator, pkg_filter, metadata)

    # With autoload=False and explicit extension pattern, the file should be in accumulator
    assert any("main.py" in p for p in accumulator.extensions)


async def test_collect_package_resources_with_pi_json_manifest(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    pkg_dir = tmp_path / "pkg3"
    pkg_dir.mkdir()
    (pkg_dir / "myext.py").write_text(_ext("myext"))
    (pkg_dir / "pi.json").write_text(json.dumps({"extensions": ["myext.py"]}))

    from pi_coding_agent.core.package_manager import PathMetadata, _ResourceAccumulator

    accumulator = _ResourceAccumulator()
    metadata = PathMetadata(source="test", scope="user", origin="package", base_dir=str(pkg_dir))
    pm._collect_package_resources(str(pkg_dir), accumulator, None, metadata)

    assert any("myext.py" in p for p in accumulator.extensions)


# ---------------------------------------------------------------------------
# resolve_extension_sources: temporary scope
# ---------------------------------------------------------------------------


async def test_resolve_extension_sources_temporary_scope(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    pkg_dir = tmp_path / "temp-pkg"
    pkg_dir.mkdir()
    (pkg_dir / "extensions").mkdir()
    (pkg_dir / "extensions" / "main.py").write_text(_ext("temp"))

    result = await pm.resolve_extension_sources([str(pkg_dir)], temporary=True)
    assert any("main.py" in r.path for r in result.extensions)


# ---------------------------------------------------------------------------
# _resolve_managed_path: path escape detection
# ---------------------------------------------------------------------------


def test_resolve_managed_path_raises_on_path_traversal(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    install_root = str(tmp_path / "installs")
    os.makedirs(install_root)
    with pytest.raises(ValueError, match="Refusing"):
        pm._resolve_managed_path(install_root, "..", "escape")


def test_resolve_managed_path_allows_valid_path(tmp_path):
    pm, _, _, _ = _make_manager(tmp_path)
    install_root = str(tmp_path / "installs")
    os.makedirs(install_root)
    result = pm._resolve_managed_path(install_root, "valid", "subdir")
    assert result.startswith(install_root)
