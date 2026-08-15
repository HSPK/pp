"""Tests for pi_coding_agent.core.package_manager.

Ported from packages/coding-agent/test/package-manager.test.ts (2593 lines) and
packages/coding-agent/test/package-manager-ssh.test.ts. Cases relying on
TypeScript-only behavior with no Python equivalent are skipped, per
package_manager.py's module docstring:

- All `npm:` source tests (npm registry installs, npmCommand/pnpm/bun
  wrappers, dependency-install-after-clone, offline npm version lookups,
  self-update/model-refresh) -- there is no Python package registry
  equivalent; `parse_source()` raises `ValueError` for `npm:` input instead,
  tested below.
- `.ts`/`.js`/`index.ts` extension file conventions are replaced with the
  `.py`/`__init__.py` convention `core/extensions/loader.py` already uses.
- `package.json`'s nested `"pi"` field is replaced with a top-level
  `pi.json` file (see pi_manifest.py's module docstring).

Git-source tests use two strategies, both stated per-test:
- A *fake* `CommandRunner` (`FakeCommandRunner` below) that records calls
  without touching the filesystem/network, used where the test only cares
  that certain git subprocess arguments were/weren't issued.
- A *real* local git repository created with `git init` in `tmp_path`,
  installed/updated/removed through the real `SubprocessCommandRunner`
  (default) -- this never touches the network since the "remote" is a local
  filesystem path, matching git-update.test.ts's approach in the TypeScript
  suite.

Trust gating (`install`/`remove` refusing untrusted-project scope) is ported
and tested exactly like `core/extensions/loader.py`'s equivalent tests.
"""

import asyncio
import json
import os
import stat
import subprocess
import sys
import tempfile

import pytest

from pi_coding_agent.core.package_manager import (
    CommandRunner,
    GitSource,
    LocalSource,
    PackageManager,
    ProgressEvent,
    _is_offline_mode_enabled,
)
from pi_coding_agent.core.settings_manager import SettingsManager, SettingsManagerCreateOptions


def _write(path: str, content: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _ext(name: str = "ext") -> str:
    return f"def pi_extension(pi):\n    pi.register_command('{name}', handler=lambda args, ctx: None)\n"


def _is_enabled(resources, suffix: str) -> bool:
    return any(r.path.replace("\\", "/").endswith(suffix) and r.enabled for r in resources)


def _is_disabled(resources, suffix: str) -> bool:
    return any(r.path.replace("\\", "/").endswith(suffix) and not r.enabled for r in resources)


def _contains_enabled(resources, needle: str) -> bool:
    return any(needle in r.path.replace("\\", "/") and r.enabled for r in resources)


def _contains(resources, needle: str) -> bool:
    return any(needle in r.path.replace("\\", "/") for r in resources)


def _make_manager(tmp_path, *, command_runner: CommandRunner | None = None, project_trusted: bool = True):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    settings_manager = SettingsManager.in_memory(options=SettingsManagerCreateOptions(project_trusted=project_trusted))
    pm = PackageManager(str(cwd), str(agent_dir), settings_manager, command_runner=command_runner)
    return pm, settings_manager, str(cwd), str(agent_dir)


class FakeCommandRunner(CommandRunner):
    """Records git invocations without touching the filesystem or network."""

    def __init__(self):
        self.calls: list[tuple[str, list[str], str | None]] = []
        self.capture_responses: dict[tuple[str, ...], str] = {}

    async def run(self, command, args, *, cwd=None):
        self.calls.append((command, args, cwd))

    async def run_capture(self, command, args, *, cwd=None, timeout=None, env=None):
        self.calls.append((command, args, cwd))
        return self.capture_responses.get((command, *args), "")


def _init_local_git_remote(path: str, files: dict[str, str]) -> None:
    """Create a real local git repository at `path` to act as a fake 'remote'.

    No network access: cloning/fetching from a local filesystem path.
    """
    os.makedirs(path, exist_ok=True)
    subprocess.run(["git", "init", "--initial-branch=main", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    for rel_path, content in files.items():
        _write(os.path.join(path, rel_path), content)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=path, check=True)


# ---------------------------------------------------------------------------
# resolve(): local extension/skill/prompt/theme paths from settings
# ---------------------------------------------------------------------------


async def test_resolve_returns_empty_when_no_sources_configured(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    result = await pm.resolve()
    assert result.extensions == []
    assert result.skills == []
    assert result.prompts == []
    assert result.themes == []


async def test_resolve_local_extension_paths_from_settings(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    ext_dir = os.path.join(agent_dir, "extensions")
    _write(os.path.join(ext_dir, "my_ext.py"), _ext())
    settings.set_extension_paths(["extensions"])

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "my_ext.py")


async def test_resolve_skill_paths_from_settings(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    skill_dir = os.path.join(agent_dir, "skills", "my-skill")
    _write(os.path.join(skill_dir, "SKILL.md"), "---\nname: my-skill\ndescription: Test\n---\nContent")
    settings.set_skill_paths(["skills"])

    result = await pm.resolve()
    assert _contains_enabled(result.skills, "my-skill")


async def test_resolve_project_paths_relative_to_pi(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    ext_dir = os.path.join(cwd, ".pi", "extensions")
    _write(os.path.join(ext_dir, "proj_ext.py"), _ext())
    settings.set_project_extension_paths(["extensions"])

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "proj_ext.py")
    assert next(r for r in result.extensions if r.path.endswith("proj_ext.py")).metadata.scope == "project"


# ---------------------------------------------------------------------------
# pattern filtering in top-level arrays
# ---------------------------------------------------------------------------


async def test_exclude_extensions_with_bang_pattern(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    ext_dir = os.path.join(agent_dir, "extensions")
    _write(os.path.join(ext_dir, "keep.py"), _ext("keep"))
    _write(os.path.join(ext_dir, "remove.py"), _ext("remove"))
    settings.set_extension_paths(["extensions", "!**/remove.py"])

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "keep.py")
    assert _is_disabled(result.extensions, "remove.py")


async def test_filter_themes_with_glob_patterns(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    themes_dir = os.path.join(agent_dir, "themes")
    _write(os.path.join(themes_dir, "dark.json"), "{}")
    _write(os.path.join(themes_dir, "light.json"), "{}")
    _write(os.path.join(themes_dir, "funky.json"), "{}")
    settings.set_theme_paths(["themes", "!funky.json"])

    result = await pm.resolve()
    assert _is_enabled(result.themes, "dark.json")
    assert _is_enabled(result.themes, "light.json")
    assert _is_disabled(result.themes, "funky.json")


async def test_filter_prompts_with_exclusion_pattern(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    prompts_dir = os.path.join(agent_dir, "prompts")
    _write(os.path.join(prompts_dir, "review.md"), "Review code")
    _write(os.path.join(prompts_dir, "explain.md"), "Explain code")
    settings.set_prompt_template_paths(["prompts", "!explain.md"])

    result = await pm.resolve()
    assert _is_enabled(result.prompts, "review.md")
    assert _is_disabled(result.prompts, "explain.md")


async def test_filter_skills_with_exclusion_pattern(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    skills_dir = os.path.join(agent_dir, "skills")
    _write(os.path.join(skills_dir, "good-skill", "SKILL.md"), "---\nname: good-skill\ndescription: Good\n---\nX")
    _write(os.path.join(skills_dir, "bad-skill", "SKILL.md"), "---\nname: bad-skill\ndescription: Bad\n---\nX")
    settings.set_skill_paths(["skills", "!**/bad-skill"])

    result = await pm.resolve()
    assert _contains_enabled(result.skills, "good-skill")
    assert any("bad-skill" in r.path and not r.enabled for r in result.skills)


async def test_resolve_without_patterns_backward_compatible(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    ext_path = _write(os.path.join(agent_dir, "extensions", "my_ext.py"), _ext())
    settings.set_extension_paths(["extensions/my_ext.py"])

    result = await pm.resolve()
    assert any(r.path == ext_path and r.enabled for r in result.extensions)


# ---------------------------------------------------------------------------
# resolve_extension_sources: local paths, pi.json manifest, auto-discovery
# ---------------------------------------------------------------------------


async def test_resolve_extension_sources_local_paths(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "local-pkg"
    ext_path = _write(str(pkg_dir / "extensions" / "main.py"), _ext())

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert any(r.path == ext_path and r.enabled for r in result.extensions)


async def test_resolve_extension_sources_with_pi_manifest_glob_patterns(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "manifest-pkg"
    _write(str(pkg_dir / "extensions" / "local.py"), _ext("local"))
    _write(str(pkg_dir / "vendor" / "dep" / "extensions" / "remote.py"), _ext("remote"))
    _write(str(pkg_dir / "vendor" / "dep" / "extensions" / "skip.py"), _ext("skip"))
    _write(
        str(pkg_dir / "pi.json"),
        json.dumps({"extensions": ["extensions", "vendor/dep/extensions", "!**/skip.py"]}),
    )

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert _is_enabled(result.extensions, "local.py")
    assert _is_enabled(result.extensions, "remote.py")
    assert not _contains(result.extensions, "skip.py")


async def test_resolve_extension_sources_pi_manifest_skills_glob(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "skill-manifest-pkg"
    _write(str(pkg_dir / "skills" / "good-skill" / "SKILL.md"), "---\nname: good-skill\ndescription: Good\n---\nX")
    _write(str(pkg_dir / "skills" / "bad-skill" / "SKILL.md"), "---\nname: bad-skill\ndescription: Bad\n---\nX")
    _write(str(pkg_dir / "pi.json"), json.dumps({"skills": ["skills", "!**/bad-skill"]}))

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert _contains_enabled(result.skills, "good-skill")
    assert not _contains(result.skills, "bad-skill")


async def test_resolve_extension_sources_expands_positive_glob_manifest_entries(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "skill-manifest-glob-pkg"
    _write(
        str(pkg_dir / "plugins" / "a" / "skills" / "skill-a" / "SKILL.md"), "---\nname: skill-a\ndescription: A\n---\nX"
    )
    _write(
        str(pkg_dir / "plugins" / "b" / "skills" / "skill-b" / "SKILL.md"), "---\nname: skill-b\ndescription: B\n---\nX"
    )
    _write(str(pkg_dir / "pi.json"), json.dumps({"skills": ["./plugins/*/skills"]}))

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert _contains_enabled(result.skills, "skill-a")
    assert _contains_enabled(result.skills, "skill-b")


# ---------------------------------------------------------------------------
# multi-file extension discovery (issue #1102 equivalent): __init__.py convention
# ---------------------------------------------------------------------------


async def test_only_loads_init_py_from_subdirectories_not_helper_modules(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "multifile-pkg"
    _write(
        str(pkg_dir / "extensions" / "subagent" / "__init__.py"),
        "from .helpers import helper\n\ndef pi_extension(pi):\n    pass\n",
    )
    _write(str(pkg_dir / "extensions" / "subagent" / "helpers.py"), "def helper():\n    return 'helper'\n")
    _write(str(pkg_dir / "extensions" / "standalone.py"), _ext("standalone"))

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert _is_enabled(result.extensions, "subagent/__init__.py")
    assert _is_enabled(result.extensions, "standalone.py")
    assert not _contains(result.extensions, "helpers.py")


async def test_respects_pi_json_manifest_in_subdirectories(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "manifest-subdir-pkg"
    _write(str(pkg_dir / "extensions" / "custom" / "pi.json"), json.dumps({"extensions": ["./main.py"]}))
    _write(str(pkg_dir / "extensions" / "custom" / "main.py"), _ext("main"))
    _write(str(pkg_dir / "extensions" / "custom" / "utils.py"), "util = 1\n")

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert _is_enabled(result.extensions, "custom/main.py")
    assert not _contains(result.extensions, "utils.py")


async def test_skips_subdirectories_without_init_or_manifest(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "no-entry-pkg"
    _write(str(pkg_dir / "extensions" / "broken" / "helper.py"), "x = 1\n")
    _write(str(pkg_dir / "extensions" / "valid.py"), _ext("valid"))

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    enabled = [r for r in result.extensions if r.enabled]
    assert len(enabled) == 1
    assert enabled[0].path.endswith("valid.py")


async def test_handles_directories_with_pi_manifest(tmp_path):
    """TS: "should handle directories with pi manifest" -- a package root whose
    manifest points extensions/skills at non-conventional locations. TS reads
    `package.json`'s nested `"pi"` field; this port reads a top-level
    `pi.json` (see module docstring).
    """
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "my-package"
    _write(str(pkg_dir / "pi.json"), json.dumps({"extensions": ["./src/main.py"], "skills": ["./skills"]}))
    ext_path = _write(str(pkg_dir / "src" / "main.py"), _ext("my-package"))
    skill_path = _write(
        str(pkg_dir / "skills" / "my-skill" / "SKILL.md"),
        "---\nname: my-skill\ndescription: Test\n---\nContent",
    )

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert any(r.path == ext_path and r.enabled for r in result.extensions)
    # Skills with SKILL.md are returned as file paths.
    assert any(r.path == skill_path and r.enabled for r in result.skills)


# ---------------------------------------------------------------------------
# pattern filtering in package filters (settings `packages` entries)
# ---------------------------------------------------------------------------


async def test_user_filters_layer_on_top_of_manifest_filters(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "layered-pkg"
    _write(str(pkg_dir / "extensions" / "foo.py"), _ext("foo"))
    _write(str(pkg_dir / "extensions" / "bar.py"), _ext("bar"))
    _write(str(pkg_dir / "extensions" / "baz.py"), _ext("baz"))
    _write(str(pkg_dir / "pi.json"), json.dumps({"extensions": ["extensions", "!**/baz.py"]}))

    settings.set_packages(
        [{"source": str(pkg_dir), "extensions": ["!**/bar.py"], "skills": [], "prompts": [], "themes": []}]
    )

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "foo.py")
    assert _is_disabled(result.extensions, "bar.py")
    assert not _contains(result.extensions, "baz.py")


async def test_exclude_extensions_from_package_with_bang_pattern(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "pattern-pkg"
    _write(str(pkg_dir / "extensions" / "foo.py"), _ext("foo"))
    _write(str(pkg_dir / "extensions" / "bar.py"), _ext("bar"))
    _write(str(pkg_dir / "extensions" / "baz.py"), _ext("baz"))

    settings.set_packages(
        [{"source": str(pkg_dir), "extensions": ["!**/baz.py"], "skills": [], "prompts": [], "themes": []}]
    )

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "foo.py")
    assert _is_enabled(result.extensions, "bar.py")
    assert _is_disabled(result.extensions, "baz.py")


async def test_filter_themes_from_package(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "theme-pkg"
    _write(str(pkg_dir / "themes" / "nice.json"), "{}")
    _write(str(pkg_dir / "themes" / "ugly.json"), "{}")

    settings.set_packages(
        [{"source": str(pkg_dir), "extensions": [], "skills": [], "prompts": [], "themes": ["!ugly.json"]}]
    )

    result = await pm.resolve()
    assert _is_enabled(result.themes, "nice.json")
    assert _is_disabled(result.themes, "ugly.json")


async def test_combine_include_and_exclude_patterns(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "combo-pkg"
    _write(str(pkg_dir / "extensions" / "alpha.py"), _ext("alpha"))
    _write(str(pkg_dir / "extensions" / "beta.py"), _ext("beta"))
    _write(str(pkg_dir / "extensions" / "gamma.py"), _ext("gamma"))

    settings.set_packages(
        [
            {
                "source": str(pkg_dir),
                "extensions": ["**/alpha.py", "**/beta.py", "!**/beta.py"],
                "skills": [],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "alpha.py")
    assert _is_disabled(result.extensions, "beta.py")
    assert _is_disabled(result.extensions, "gamma.py")


async def test_direct_paths_without_patterns(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "direct-pkg"
    _write(str(pkg_dir / "extensions" / "one.py"), _ext("one"))
    _write(str(pkg_dir / "extensions" / "two.py"), _ext("two"))

    settings.set_packages(
        [{"source": str(pkg_dir), "extensions": ["extensions/one.py"], "skills": [], "prompts": [], "themes": []}]
    )

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "one.py")
    assert _is_disabled(result.extensions, "two.py")


# ---------------------------------------------------------------------------
# force-include (+) / force-exclude (-) patterns
# ---------------------------------------------------------------------------


async def test_force_include_extensions_with_plus_pattern_after_exclusion(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    ext_dir = os.path.join(agent_dir, "extensions")
    _write(os.path.join(ext_dir, "keep.py"), _ext("keep"))
    _write(os.path.join(ext_dir, "excluded.py"), _ext("excluded"))
    _write(os.path.join(ext_dir, "force_back.py"), _ext("force_back"))

    settings.set_extension_paths(["extensions", "!extensions/*.py", "+extensions/force_back.py"])

    result = await pm.resolve()
    assert _is_disabled(result.extensions, "keep.py")
    assert _is_disabled(result.extensions, "excluded.py")
    assert _is_enabled(result.extensions, "force_back.py")


async def test_force_exclude_top_level_resources(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    ext_dir = os.path.join(agent_dir, "extensions")
    _write(os.path.join(ext_dir, "alpha.py"), _ext("alpha"))
    _write(os.path.join(ext_dir, "beta.py"), _ext("beta"))

    settings.set_extension_paths(["extensions", "+extensions/alpha.py", "-extensions/alpha.py"])

    result = await pm.resolve()
    assert _is_disabled(result.extensions, "alpha.py")
    assert _is_enabled(result.extensions, "beta.py")


async def test_force_exclude_in_package_filters(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "force-exclude-pkg"
    _write(str(pkg_dir / "extensions" / "alpha.py"), _ext("alpha"))
    _write(str(pkg_dir / "extensions" / "beta.py"), _ext("beta"))

    settings.set_packages(
        [
            {
                "source": str(pkg_dir),
                "extensions": ["extensions/*.py", "+extensions/alpha.py", "-extensions/alpha.py"],
                "skills": [],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await pm.resolve()
    assert _is_disabled(result.extensions, "alpha.py")
    assert _is_enabled(result.extensions, "beta.py")


# ---------------------------------------------------------------------------
# autoload-disabled project package deltas
# ---------------------------------------------------------------------------


async def test_autoload_disabled_project_package_resolves_as_delta_over_global(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(agent_dir, "packages", "shared-tools")
    _write(os.path.join(pkg_dir, "extensions", "foo.py"), _ext("foo"))
    _write(os.path.join(pkg_dir, "extensions", "bar.py"), _ext("bar"))

    settings.set_packages([pkg_dir])
    settings.set_project_packages([{"source": pkg_dir, "autoload": False, "extensions": ["-extensions/foo.py"]}])

    result = await pm.resolve()
    states = {r.path: (r.enabled, r.metadata.scope) for r in result.extensions}
    assert states[os.path.join(pkg_dir, "extensions", "foo.py")] == (False, "project")
    assert states[os.path.join(pkg_dir, "extensions", "bar.py")] == (True, "user")


async def test_autoload_disabled_package_entries_positive_only_without_global(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "positive-only-pkg"
    _write(str(pkg_dir / "extensions" / "foo.py"), _ext("foo"))
    _write(str(pkg_dir / "extensions" / "bar.py"), _ext("bar"))
    _write(str(pkg_dir / "skills" / "foo" / "SKILL.md"), "---\nname: foo\ndescription: F\n---\nX")

    rel = os.path.relpath(str(pkg_dir), os.path.join(cwd, ".pi"))
    settings.set_project_packages([{"source": rel, "autoload": False, "extensions": ["+extensions/foo.py"]}])

    result = await pm.resolve()
    assert [r.path for r in result.extensions] == [str(pkg_dir / "extensions" / "foo.py")]
    assert result.skills == []


# ---------------------------------------------------------------------------
# package deduplication
# ---------------------------------------------------------------------------


async def test_dedupe_same_local_package_in_global_and_project_project_wins(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "shared-pkg"
    _write(str(pkg_dir / "extensions" / "shared.py"), _ext("shared"))

    settings.set_packages([str(pkg_dir)])
    settings.set_project_packages([str(pkg_dir)])

    result = await pm.resolve()
    shared = [r for r in result.extensions if "shared-pkg" in r.path]
    assert len(shared) == 1
    assert shared[0].metadata.scope == "project"


async def test_keep_both_if_different_packages(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg1 = tmp_path / "pkg1"
    pkg2 = tmp_path / "pkg2"
    _write(str(pkg1 / "extensions" / "from_pkg1.py"), _ext("from_pkg1"))
    _write(str(pkg2 / "extensions" / "from_pkg2.py"), _ext("from_pkg2"))

    settings.set_packages([str(pkg1)])
    settings.set_project_packages([str(pkg2)])

    result = await pm.resolve()
    assert _contains(result.extensions, "pkg1")
    assert _contains(result.extensions, "pkg2")


# ---------------------------------------------------------------------------
# source parsing / identity / settings normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected_host", "expected_path", "expected_ref", "expected_pinned"),
    [
        ("https://github.com/user/repo", "github.com", "user/repo", None, False),
        ("git:https://github.com/user/repo", "github.com", "user/repo", None, False),
        ("https://github.com/user/repo@v1.2.3", "github.com", "user/repo", "v1.2.3", True),
        ("git:github.com/user/repo", "github.com", "user/repo", None, False),
        ("https://github.com/user/repo.git", "github.com", "user/repo", None, False),
        ("https://gitlab.com/user/repo", "gitlab.com", "user/repo", None, False),
        ("https://bitbucket.org/user/repo", "bitbucket.org", "user/repo", None, False),
        ("https://codeberg.org/user/repo", "codeberg.org", "user/repo", None, False),
        ("https://github.com/user/repo@main", "github.com", "user/repo", "main", True),
        ("https://github.com/user/repo@feature/branch", "github.com", "user/repo", "feature/branch", True),
    ],
)
def test_parse_source_https_git_urls(tmp_path, source, expected_host, expected_path, expected_ref, expected_pinned):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source(source)
    assert isinstance(parsed, GitSource)
    assert parsed.host == expected_host
    assert parsed.path == expected_path
    assert parsed.ref == expected_ref
    assert parsed.pinned == expected_pinned


def test_parse_source_host_path_shorthand_only_with_git_prefix(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("git:github.com/user/repo")
    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"


def test_parse_source_host_path_shorthand_without_prefix_is_local(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("github.com/user/repo")
    assert isinstance(parsed, LocalSource)


def test_parse_source_never_parses_dot_relative_paths_as_git(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    dot_slash = pm.parse_source("./packages/agent-timers")
    assert isinstance(dot_slash, LocalSource)
    assert dot_slash.path == "./packages/agent-timers"

    dot_dot_slash = pm.parse_source("../packages/agent-timers")
    assert isinstance(dot_dot_slash, LocalSource)
    assert dot_dot_slash.path == "../packages/agent-timers"


def test_parse_source_types_from_docs_examples(tmp_path):
    """TS: "should parse package source types from docs examples".

    The three `npm:` assertions of that case (pinned-ness of
    `npm:@scope/pkg@1.2.3`, `npm:@scope/pkg@^1.2.3`, `npm:pkg`) have no
    Python equivalent -- `parse_source()` rejects `npm:` outright, which
    `test_parse_source_npm_raises_documented_error` pins.
    """
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)

    assert isinstance(pm.parse_source("git:github.com/user/repo@v1"), GitSource)
    assert isinstance(pm.parse_source("https://github.com/user/repo@v1"), GitSource)
    assert isinstance(pm.parse_source("git:git@github.com:user/repo@v1"), GitSource)
    assert isinstance(pm.parse_source("ssh://git@github.com/user/repo@v1"), GitSource)

    assert isinstance(pm.parse_source("/absolute/path/to/package"), LocalSource)
    assert isinstance(pm.parse_source("./relative/path/to/package"), LocalSource)
    assert isinstance(pm.parse_source("../relative/path/to/package"), LocalSource)


@pytest.mark.parametrize("scope", ["user", "project", "temporary"])
def test_get_git_install_path_rejects_paths_outside_install_roots(tmp_path, scope):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    traversal = GitSource(
        repo="git@evil.example:../../victim/repo",
        host="evil.example",
        path="../../victim/repo",
        pinned=False,
    )

    with pytest.raises(ValueError, match="outside package install root"):
        pm._get_git_install_path(traversal, scope)


def test_parse_source_npm_raises_documented_error(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    with pytest.raises(ValueError, match="npm package sources are not supported"):
        pm.parse_source("npm:some-package")


@pytest.mark.parametrize(
    ("url1", "url2"),
    [
        ("https://github.com/user/repo", "https://github.com/user/repo@v1.0.0"),
        ("https://github.com/user/repo", "git:github.com/user/repo"),
        ("https://github.com/user/repo", "https://github.com/user/repo.git"),
        ("https://github.com/user/repo", "ssh://git@github.com/user/repo"),
        ("https://github.com/user/repo", "git:git@github.com:user/repo"),
        ("https://github.com/user/repo", "git:git@github.com:user/repo.git"),
    ],
)
def test_package_identity_dedupes_url_formats(tmp_path, url1, url2):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    assert pm._get_package_identity(url1) == "git:github.com/user/repo"
    assert pm._get_package_identity(url2) == "git:github.com/user/repo"


def test_package_identity_keeps_different_repos_separate(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    id1 = pm._get_package_identity("https://github.com/user/repo1")
    id2 = pm._get_package_identity("git:git@github.com:user/repo2")
    assert id1 == "git:github.com/user/repo1"
    assert id2 == "git:github.com/user/repo2"
    assert id1 != id2


def test_package_identity_normalizes_protocol_and_shorthand_prefixed_urls(tmp_path):
    """Ported from package-manager-ssh.test.ts's 'identity normalization'."""
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    prefixed = pm._get_package_identity("git:git@github.com:user/repo")
    https = pm._get_package_identity("https://github.com/user/repo")
    ssh = pm._get_package_identity("ssh://git@github.com/user/repo")
    assert prefixed == "git:github.com/user/repo"
    assert prefixed == https == ssh


def test_parse_source_ssh_protocol_url(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("ssh://git@github.com/user/repo")
    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"
    assert parsed.repo == "ssh://git@github.com/user/repo"


def test_parse_source_git_at_host_colon_path_format(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("git:git@github.com:user/repo")
    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"
    assert parsed.repo == "git@github.com:user/repo"
    assert parsed.pinned is False


def test_parse_source_shorthand_with_ref_is_pinned(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("git:git@github.com:user/repo@v1.0.0")
    assert isinstance(parsed, GitSource)
    assert parsed.ref == "v1.0.0"
    assert parsed.pinned is True


def test_parse_source_https_protocol_url(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("https://github.com/user/repo")
    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"


def test_parse_source_host_path_shorthand_with_git_prefix(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("git:github.com/user/repo")
    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"


def test_parse_source_git_at_host_colon_path_without_prefix_is_local(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    parsed = pm.parse_source("git@github.com:user/repo")
    assert isinstance(parsed, LocalSource)


def test_add_source_to_settings_stores_global_local_packages_relative_to_agent_dir(tmp_path):
    pm, settings, cwd, agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "packages", "local-global-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())

    added = pm.add_source_to_settings(pkg_dir)
    assert added is True
    rel = os.path.relpath(pkg_dir, agent_dir)
    stored = settings.get_global_settings()["packages"][0]
    assert stored == rel


def test_add_source_to_settings_stores_project_local_packages_relative_to_pi(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "project-local-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())

    added = pm.add_source_to_settings(pkg_dir, local=True)
    assert added is True
    rel = os.path.relpath(pkg_dir, os.path.join(cwd, ".pi"))
    stored = settings.get_project_settings()["packages"][0]
    assert stored == rel


def test_remove_source_from_settings_using_equivalent_path_forms(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "remove-local-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())

    pm.add_source_to_settings(pkg_dir)
    removed = pm.remove_source_from_settings(pkg_dir + os.sep)
    assert removed is True
    assert (settings.get_global_settings().get("packages") or []) == []


def test_add_source_to_settings_returns_false_for_same_git_source_and_ref(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    first = pm.add_source_to_settings("git:github.com/user/repo@v1")
    assert first is True
    second = pm.add_source_to_settings("git:github.com/user/repo@v1")
    assert second is False
    assert settings.get_global_settings()["packages"] == ["git:github.com/user/repo@v1"]


def test_add_source_to_settings_updates_ref_for_same_git_source(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pm.add_source_to_settings("git:github.com/user/repo@v1")
    updated = pm.add_source_to_settings("git:github.com/user/repo@v2")
    assert updated is True
    assert settings.get_global_settings()["packages"] == ["git:github.com/user/repo@v2"]


def test_add_source_to_settings_preserves_package_filters_when_replacing_ref(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    settings.set_packages(
        [
            {
                "source": "git:github.com/user/repo@v1",
                "extensions": ["extensions/main.py"],
                "skills": [],
                "prompts": ["prompts/review.md"],
                "themes": ["themes/dark.json"],
            }
        ]
    )

    updated = pm.add_source_to_settings("git:github.com/user/repo@v2")
    assert updated is True
    assert settings.get_global_settings()["packages"] == [
        {
            "source": "git:github.com/user/repo@v2",
            "extensions": ["extensions/main.py"],
            "skills": [],
            "prompts": ["prompts/review.md"],
            "themes": ["themes/dark.json"],
        }
    ]


# ---------------------------------------------------------------------------
# list_configured_packages
# ---------------------------------------------------------------------------


async def test_list_configured_packages_reports_scope_and_installed_path(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "listed-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())
    pm.add_source_to_settings(pkg_dir)

    configured = pm.list_configured_packages()
    assert len(configured) == 1
    assert configured[0].scope == "user"
    assert configured[0].filtered is False
    assert configured[0].installed_path == pkg_dir


async def test_list_configured_packages_reports_missing_install_path_as_none(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    settings.set_packages(["git:github.com/nonexistent/repo"])

    configured = pm.list_configured_packages()
    assert len(configured) == 1
    assert configured[0].installed_path is None


# ---------------------------------------------------------------------------
# manifest / corrupted pi.json
# ---------------------------------------------------------------------------


async def test_corrupted_pi_json_manifest_falls_back_to_directory_convention(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "corrupted-manifest-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(pkg_dir / "pi.json"), "{not valid json")

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert _is_enabled(result.extensions, "main.py")


async def test_empty_manifest_array_explicitly_disables_resource_type(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = tmp_path / "empty-manifest-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(pkg_dir / "pi.json"), json.dumps({"extensions": []}))

    result = await pm.resolve_extension_sources([str(pkg_dir)])
    assert result.extensions == []


async def test_malformed_manifest_field_ignored_without_dropping_valid_fields(tmp_path):
    """Ported from packages/coding-agent/test/suite/regressions/
    7187-malformed-package-manifest.test.ts: a manifest field with the wrong
    JSON shape (a bare string instead of an array) must be ignored rather
    than crashing or silently loading resources from an unintended path,
    while other, validly-shaped fields in the same manifest still resolve.
    """
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(agent_dir, "packages", "bad-package")
    skill_path = os.path.join(pkg_dir, "skills", "bad", "SKILL.md")
    prompt_path = os.path.join(pkg_dir, "prompts", "valid.md")
    _write(skill_path, "---\nname: bad\ndescription: Must not load\n---\n")
    _write(prompt_path, "Valid prompt\n")
    _write(os.path.join(pkg_dir, "pi.json"), json.dumps({"skills": "./skills", "prompts": ["./prompts"]}))
    settings.set_packages([pkg_dir])

    result = await pm.resolve()
    assert skill_path not in [r.path for r in result.skills]
    assert prompt_path in [r.path for r in result.prompts]


# ---------------------------------------------------------------------------
# trust gating
# ---------------------------------------------------------------------------


async def test_install_project_scope_refused_when_untrusted(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path, project_trusted=False)
    pkg_dir = os.path.join(cwd, "some-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())

    with pytest.raises(ValueError, match="not trusted"):
        await pm.install(pkg_dir, local=True)


async def test_remove_project_scope_refused_when_untrusted(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path, project_trusted=False)

    with pytest.raises(ValueError, match="not trusted"):
        await pm.remove("git:github.com/user/repo", local=True)


async def test_install_user_scope_allowed_when_project_untrusted(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path, project_trusted=False)
    pkg_dir = os.path.join(cwd, "some-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())

    await pm.install(pkg_dir, local=False)  # should not raise


async def test_resolve_skips_project_local_dirs_when_untrusted(tmp_path):
    pm, _settings, cwd, agent_dir = _make_manager(tmp_path, project_trusted=False)
    project_ext_dir = os.path.join(cwd, ".pi", "extensions")
    global_ext_dir = os.path.join(agent_dir, "extensions")
    _write(os.path.join(project_ext_dir, "untrusted.py"), _ext("untrusted"))
    _write(os.path.join(global_ext_dir, "trusted.py"), _ext("trusted"))
    # Project settings storage itself is inert while untrusted (SettingsManager
    # returns {} for the project scope), so no explicit project extension-path
    # setting is needed here to prove the gating: get_project_settings() is
    # already empty.
    result = await pm.resolve()
    assert not _contains(result.extensions, "untrusted.py")
    assert _is_enabled(result.extensions, "trusted.py")


async def test_get_base_dir_for_project_scope_raises_when_untrusted(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path, project_trusted=False)
    with pytest.raises(ValueError, match="not trusted"):
        pm._get_base_dir_for_scope("project")


# ---------------------------------------------------------------------------
# git install / update / remove using a real local git repository (no network)
# ---------------------------------------------------------------------------


async def test_install_from_git_source_clones_real_local_repo(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    remote = str(tmp_path / "remote.git")
    _init_local_git_remote(remote, {"extensions/main.py": _ext("cloned")})

    source = GitSource(repo=remote, host="example.test", path="user/repo", ref=None, pinned=False)
    await pm._install_git(source, "user")

    installed_path = pm._get_git_install_path(source, "user")
    assert os.path.isdir(installed_path)
    assert os.path.isfile(os.path.join(installed_path, "extensions", "main.py"))


async def test_install_from_git_source_with_pinned_ref(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    remote = str(tmp_path / "remote-pinned.git")
    _init_local_git_remote(remote, {"extensions/v1.py": _ext("v1")})
    subprocess.run(["git", "tag", "v1.0.0"], cwd=remote, check=True)
    _write(os.path.join(remote, "extensions", "v2.py"), _ext("v2"))
    subprocess.run(["git", "add", "."], cwd=remote, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "v2"], cwd=remote, check=True)

    source = GitSource(repo=remote, host="example.test", path="user/pinned-repo", ref="v1.0.0", pinned=True)
    await pm._install_git(source, "user")

    installed_path = pm._get_git_install_path(source, "user")
    assert os.path.isfile(os.path.join(installed_path, "extensions", "v1.py"))
    # Pinned to v1.0.0: the v2-only file must not be present.
    assert not os.path.isfile(os.path.join(installed_path, "extensions", "v2.py"))


async def test_update_git_source_pulls_new_commits(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    remote = str(tmp_path / "remote-update.git")
    _init_local_git_remote(remote, {"extensions/main.py": _ext("first")})

    source = GitSource(repo=remote, host="example.test", path="user/update-repo", ref=None, pinned=False)
    await pm._install_git(source, "user")
    installed_path = pm._get_git_install_path(source, "user")
    assert not os.path.isfile(os.path.join(installed_path, "extensions", "new_file.py"))

    _write(os.path.join(remote, "extensions", "new_file.py"), _ext("new"))
    subprocess.run(["git", "add", "."], cwd=remote, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "add new_file"], cwd=remote, check=True)

    await pm._update_git(source, "user")
    assert os.path.isfile(os.path.join(installed_path, "extensions", "new_file.py"))


async def test_remove_git_source_deletes_checkout(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    remote = str(tmp_path / "remote-remove.git")
    _init_local_git_remote(remote, {"extensions/main.py": _ext()})

    source = GitSource(repo=remote, host="example.test", path="user/remove-repo", ref=None, pinned=False)
    await pm._install_git(source, "user")
    installed_path = pm._get_git_install_path(source, "user")
    assert os.path.isdir(installed_path)

    await pm._remove_git(source, "user")
    assert not os.path.exists(installed_path)


async def test_install_recognizes_github_urls_without_git_prefix(tmp_path):
    """TS: "should recognize github URLs without git: prefix"."""

    class FailingCommandRunner(FakeCommandRunner):
        async def run(self, command, args, *, cwd=None):
            self.calls.append((command, args, cwd))
            raise RuntimeError("simulated git clone failure")

    fake = FailingCommandRunner()
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=fake)
    events: list[ProgressEvent] = []
    pm.set_progress_callback(events.append)
    source = "https://github.com/nonexistent/repo"

    with pytest.raises(RuntimeError, match="simulated git clone failure"):
        await pm.install(source)

    installed_path = pm._get_git_install_path(pm.parse_source(source), "user")
    assert ("git", ["clone", source, installed_path], None) in fake.calls
    assert any(e.type == "start" and e.action == "install" for e in events)


async def test_install_and_persist_then_remove_and_persist_round_trip(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "roundtrip-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext())

    await pm.install_and_persist(pkg_dir)
    assert (settings.get_global_settings().get("packages") or []) != []
    result = await pm.resolve()
    assert _is_enabled(result.extensions, "main.py")

    removed = await pm.remove_and_persist(pkg_dir)
    assert removed is True
    assert (settings.get_global_settings().get("packages") or []) == []
    result_after_remove = await pm.resolve()
    assert not _contains(result_after_remove.extensions, "roundtrip-pkg")


async def test_git_source_install_and_persist_manifest_round_trip(tmp_path):
    """Manifest/lockfile round trip for a git source: settings.json's
    `packages` array *is* the installed-package record (see module
    docstring) -- add persists it, resolve() re-clones/reads from the
    installed checkout, remove_and_persist both deletes the checkout and
    drops the settings entry.
    """
    fake = FakeCommandRunner()
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=fake)
    source = "https://github.com/roundtrip-user/roundtrip-repo"

    added = pm.add_source_to_settings(source)
    assert added is True
    assert settings.get_global_settings()["packages"] == [source]

    # Simulate a successful clone by creating the install path the real
    # `git clone` would have produced, without touching the network.
    parsed = pm.parse_source(source)
    installed_path = pm._get_git_install_path(parsed, "user")
    _write(os.path.join(installed_path, "extensions", "main.py"), _ext())

    configured = pm.list_configured_packages()
    assert configured == [
        type(configured[0])(source=source, scope="user", filtered=False, installed_path=installed_path)
    ]

    removed = await pm.remove_and_persist(source)
    assert removed is True
    assert not os.path.exists(installed_path)
    assert (settings.get_global_settings().get("packages") or []) == []


async def test_update_with_no_matching_source_raises_with_suggestion(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    settings.set_packages(["git:github.com/user/repo"])

    with pytest.raises(ValueError, match="No matching package found"):
        await pm.update("git:github.com/other/repo")


async def test_update_dispatches_to_matching_git_sources_only(tmp_path):
    fake = FakeCommandRunner()
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path, command_runner=fake)
    local_pkg = os.path.join(agent_dir, "packages", "local-pkg")
    _write(os.path.join(local_pkg, "extensions", "main.py"), _ext())
    settings.set_packages(["git:github.com/user/repo", local_pkg])

    # Fake runner: make `_get_local_git_update_target`'s upstream lookup and
    # the subsequent fetch/rev-parse calls succeed trivially.
    fake.capture_responses[("git", "rev-parse", "--abbrev-ref", "@{upstream}")] = "origin/main"
    fake.capture_responses[("git", "rev-parse", "HEAD")] = "abc123"
    fake.capture_responses[("git", "rev-parse", "FETCH_HEAD^{commit}")] = "abc123"

    # git source isn't installed on disk, so _update_git will call _install_git,
    # which will attempt `git clone` via the fake runner (recorded, not executed).
    await pm.update("git:github.com/user/repo")

    assert any(call[0] == "git" and call[1][0] == "clone" for call in fake.calls)


# ---------------------------------------------------------------------------
# check_for_available_updates
# ---------------------------------------------------------------------------


async def test_check_for_available_updates_reports_git_source_with_new_commits(tmp_path):
    pm, settings, _cwd, _agent_dir = _make_manager(tmp_path)
    remote = str(tmp_path / "remote-check-updates.git")
    _init_local_git_remote(remote, {"extensions/main.py": _ext()})
    source = GitSource(repo=remote, host="example.test", path="check/update-repo", ref=None, pinned=False)
    await pm._install_git(source, "user")

    _write(os.path.join(remote, "extensions", "new_file.py"), _ext("new"))
    subprocess.run(["git", "add", "."], cwd=remote, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "add new_file"], cwd=remote, check=True)

    settings.set_packages([f"https://{source.host}/{source.path}"])
    # Point the settings entry's identity at our fake host/path so
    # check_for_available_updates resolves the same installed_path.
    updates = await pm.check_for_available_updates()
    assert any(u.source.endswith(source.path) for u in updates)


# ---------------------------------------------------------------------------
# resolve: auto-discovery (package-manager.test.ts "resolve")
# ---------------------------------------------------------------------------


def _manager_at(cwd: str, agent_dir: str, settings: SettingsManager | None = None):
    os.makedirs(cwd, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)
    settings_manager = settings or SettingsManager.in_memory(options=SettingsManagerCreateOptions(project_trusted=True))
    return PackageManager(cwd, agent_dir, settings_manager), settings_manager


async def test_resolve_auto_discovers_root_markdown_skills_from_pi_skill_dirs(tmp_path):
    pm, _settings, _cwd, agent_dir = _make_manager(tmp_path)
    skill_file = _write(
        os.path.join(agent_dir, "skills", "single-file.md"),
        "---\nname: single-file\ndescription: A root markdown skill\n---\nContent",
    )

    result = await pm.resolve()
    assert any(r.path == skill_file and r.enabled for r in result.skills)


async def test_resolve_auto_discovers_user_prompts_with_overrides(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    prompt_path = _write(os.path.join(agent_dir, "prompts", "auto.md"), "Auto prompt")

    settings.set_prompt_template_paths(["!prompts/auto.md"])

    result = await pm.resolve()
    assert any(r.path == prompt_path and not r.enabled for r in result.prompts)


async def test_resolve_auto_discovers_project_prompts_with_overrides(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    prompt_path = _write(os.path.join(cwd, ".pi", "prompts", "is.md"), "Is prompt")

    settings.set_project_prompt_template_paths(["!prompts/is.md"])

    result = await pm.resolve()
    assert any(r.path == prompt_path and not r.enabled for r in result.prompts)


async def test_resolve_symlinked_user_and_project_resources_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pm, _settings, cwd, agent_dir = _make_manager(tmp_path)

    shared = tmp_path / "shared-resources"
    _write(str(shared / "extensions" / "shared.py"), _ext("shared"))
    _write(
        str(shared / "skills" / "shared-skill" / "SKILL.md"),
        "---\nname: shared-skill\ndescription: Shared skill\n---\nContent",
    )
    _write(str(shared / "prompts" / "shared.md"), "Shared prompt")
    _write(str(shared / "themes" / "shared.json"), json.dumps({"name": "shared-theme"}))

    os.makedirs(os.path.join(cwd, ".pi"), exist_ok=True)
    for kind in ("extensions", "skills", "prompts", "themes"):
        os.symlink(str(shared / kind), os.path.join(agent_dir, kind))
        os.symlink(str(shared / kind), os.path.join(cwd, ".pi", kind))

    result = await pm.resolve()

    assert (
        len(result.extensions),
        len(result.skills),
        len(result.prompts),
        len(result.themes),
    ) == (1, 1, 1, 1)
    # Project auto-discovered has higher precedence than user auto-discovered.
    assert result.extensions[0].metadata.scope == "project"
    assert result.skills[0].metadata.scope == "project"
    assert result.prompts[0].metadata.scope == "project"
    assert result.themes[0].metadata.scope == "project"


async def test_resolve_directory_with_pi_manifest_extensions_in_extensions_setting(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "my-extensions-pkg")
    _write(
        os.path.join(pkg_dir, "pi.json"),
        json.dumps({"name": "my-extensions-pkg", "extensions": ["./extensions/clip.py", "./extensions/cost.py"]}),
    )
    _write(os.path.join(pkg_dir, "extensions", "clip.py"), _ext("clip"))
    _write(os.path.join(pkg_dir, "extensions", "cost.py"), _ext("cost"))
    _write(os.path.join(pkg_dir, "extensions", "helper.py"), "X = 1\n")

    settings.set_extension_paths([pkg_dir])

    result = await pm.resolve()

    assert any(r.path == os.path.join(pkg_dir, "extensions", "clip.py") and r.enabled for r in result.extensions)
    assert any(r.path == os.path.join(pkg_dir, "extensions", "cost.py") and r.enabled for r in result.extensions)
    assert not _contains(result.extensions, "helper.py")


# ---------------------------------------------------------------------------
# auto-discovered skill metadata
# ---------------------------------------------------------------------------


async def test_auto_skill_metadata_agent_dir_is_base_dir_for_user_pi_skills(tmp_path):
    pm, _settings, _cwd, agent_dir = _make_manager(tmp_path)
    skill_path = _write(
        os.path.join(agent_dir, "skills", "user-pi", "SKILL.md"),
        "---\nname: user-pi\ndescription: user pi\n---\n",
    )

    result = await pm.resolve()
    skill = next(r for r in result.skills if r.path == skill_path)
    assert skill.metadata.source == "auto"
    assert skill.metadata.scope == "user"
    assert skill.metadata.base_dir == agent_dir


async def test_auto_skill_metadata_project_pi_dir_is_base_dir_for_project_skills(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    project_base_dir = os.path.join(cwd, ".pi")
    skill_path = _write(
        os.path.join(project_base_dir, "skills", "project-pi", "SKILL.md"),
        "---\nname: project-pi\ndescription: project pi\n---\n",
    )

    result = await pm.resolve()
    skill = next(r for r in result.skills if r.path == skill_path)
    assert skill.metadata.source == "auto"
    assert skill.metadata.scope == "project"
    assert skill.metadata.base_dir == project_base_dir


async def test_auto_skill_metadata_home_agents_is_base_dir_for_user_agents_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    agents_base_dir = str(tmp_path / ".agents")
    skill_path = _write(
        os.path.join(agents_base_dir, "skills", "user-agents", "SKILL.md"),
        "---\nname: user-agents\ndescription: user agents\n---\n",
    )

    result = await pm.resolve()
    skill = next(r for r in result.skills if r.path == skill_path)
    assert skill.metadata.source == "auto"
    assert skill.metadata.scope == "user"
    assert skill.metadata.base_dir == agents_base_dir


async def test_auto_skill_metadata_each_project_agents_dir_is_its_own_base_dir(tmp_path):
    repo_root = tmp_path / "repo"
    nested_cwd = repo_root / "packages" / "feature"
    nested_cwd.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    repo_agents_base = str(repo_root / ".agents")
    repo_skill = _write(
        os.path.join(repo_agents_base, "skills", "repo", "SKILL.md"),
        "---\nname: repo\ndescription: repo\n---\n",
    )
    package_agents_base = str(repo_root / "packages" / ".agents")
    package_skill = _write(
        os.path.join(package_agents_base, "skills", "package", "SKILL.md"),
        "---\nname: package\ndescription: package\n---\n",
    )

    pm, _settings = _manager_at(str(nested_cwd), str(tmp_path / "agent"))
    result = await pm.resolve()

    resolved_repo = next(r for r in result.skills if r.path == repo_skill)
    resolved_package = next(r for r in result.skills if r.path == package_skill)
    assert resolved_repo.metadata.source == "auto"
    assert resolved_repo.metadata.scope == "project"
    assert resolved_repo.metadata.base_dir == repo_agents_base
    assert resolved_package.metadata.source == "auto"
    assert resolved_package.metadata.scope == "project"
    assert resolved_package.metadata.base_dir == package_agents_base


# ---------------------------------------------------------------------------
# .agents/skills auto-discovery
# ---------------------------------------------------------------------------


async def test_agents_skills_scanned_from_cwd_up_to_git_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    nested_cwd = repo_root / "packages" / "feature"
    nested_cwd.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    above_repo_skill = _write(
        str(tmp_path / ".agents" / "skills" / "above-repo" / "SKILL.md"),
        "---\nname: above-repo\ndescription: above\n---\n",
    )
    repo_root_skill = _write(
        str(repo_root / ".agents" / "skills" / "repo-root" / "SKILL.md"),
        "---\nname: repo-root\ndescription: repo\n---\n",
    )
    nested_skill = _write(
        str(repo_root / "packages" / ".agents" / "skills" / "nested" / "SKILL.md"),
        "---\nname: nested\ndescription: nested\n---\n",
    )

    pm, _settings = _manager_at(str(nested_cwd), str(tmp_path / "agent"))
    result = await pm.resolve()

    assert any(r.path == repo_root_skill and r.enabled for r in result.skills)
    assert any(r.path == nested_skill and r.enabled for r in result.skills)
    assert not any(r.path == above_repo_skill for r in result.skills)


async def test_agents_skills_scanned_up_to_filesystem_root_when_not_in_git_repo(tmp_path):
    non_repo_root = tmp_path / "non-repo"
    nested_cwd = non_repo_root / "a" / "b"
    nested_cwd.mkdir(parents=True)

    root_skill = _write(
        str(non_repo_root / ".agents" / "skills" / "root" / "SKILL.md"),
        "---\nname: root\ndescription: root\n---\n",
    )
    middle_skill = _write(
        str(non_repo_root / "a" / ".agents" / "skills" / "middle" / "SKILL.md"),
        "---\nname: middle\ndescription: middle\n---\n",
    )

    pm, _settings = _manager_at(str(nested_cwd), str(tmp_path / "agent"))
    result = await pm.resolve()

    assert any(r.path == root_skill and r.enabled for r in result.skills)
    assert any(r.path == middle_skill and r.enabled for r in result.skills)


async def test_agents_skills_ignores_root_markdown_files(tmp_path):
    agents_skills_dir = tmp_path / ".agents" / "skills"
    root_skill = _write(
        str(agents_skills_dir / "root-file.md"),
        "---\nname: root-file\ndescription: Root markdown file\n---\n",
    )
    nested_skill = _write(
        str(agents_skills_dir / "nested-skill" / "SKILL.md"),
        "---\nname: nested-skill\ndescription: Nested skill\n---\n",
    )

    pm, _settings = _manager_at(str(tmp_path / "work"), str(tmp_path / "agent"))
    result = await pm.resolve()

    assert not any(r.path == root_skill for r in result.skills)
    assert any(r.path == nested_skill and r.enabled for r in result.skills)


async def test_home_agents_skills_stay_user_scoped_when_cwd_under_home_in_non_git_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cwd = tmp_path / "scratch" / "nested"
    local_agent_dir = tmp_path / ".pi" / "agent"
    cwd.mkdir(parents=True)
    local_agent_dir.mkdir(parents=True)

    home_skill = _write(
        str(tmp_path / ".agents" / "skills" / "home-skill" / "SKILL.md"),
        "---\nname: home-skill\ndescription: home\n---\n",
    )

    pm, _settings = _manager_at(str(cwd), str(local_agent_dir))
    result = await pm.resolve()

    matching = [r for r in result.skills if r.path == home_skill]
    assert len(matching) == 1
    assert matching[0].enabled is True
    assert matching[0].metadata.scope == "user"
    assert matching[0].metadata.source == "auto"


async def test_user_skills_deduped_when_agent_skills_symlinks_home_agents_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pm, _settings, _cwd, agent_dir = _make_manager(tmp_path)

    agents_skills_dir = tmp_path / ".agents" / "skills"
    agents_skills_dir.mkdir(parents=True)
    os.symlink(str(agents_skills_dir), os.path.join(agent_dir, "skills"))

    _write(str(agents_skills_dir / "foo" / "SKILL.md"), "---\nname: foo\ndescription: foo\n---\n")

    result = await pm.resolve()
    foo_skills = [r for r in result.skills if r.path.replace("\\", "/").endswith("foo/SKILL.md")]
    assert len(foo_skills) == 1


async def test_parent_gitignore_does_not_apply_to_pi_auto_discovery(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    _write(os.path.join(cwd, ".gitignore"), ".pi\n")
    skill_path = _write(
        os.path.join(cwd, ".pi", "skills", "auto-skill", "SKILL.md"),
        "---\nname: auto-skill\ndescription: Auto\n---\nContent",
    )

    result = await pm.resolve()
    assert any(r.path == skill_path and r.enabled for r in result.skills)


async def test_gitignore_in_skill_directories_is_respected(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    skills_dir = os.path.join(agent_dir, "skills")
    _write(os.path.join(skills_dir, ".gitignore"), "venv\n__pycache__\n")
    _write(
        os.path.join(skills_dir, "good-skill", "SKILL.md"),
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    )
    _write(
        os.path.join(skills_dir, "venv", "bad-skill", "SKILL.md"),
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    )

    settings.set_skill_paths(["skills"])

    result = await pm.resolve()
    assert _contains_enabled(result.skills, "good-skill")
    assert not _contains_enabled(result.skills, "venv")


# ---------------------------------------------------------------------------
# resolve_extension_sources
# ---------------------------------------------------------------------------


async def test_resolve_extension_sources_keeps_tilde_manifest_entries_package_relative(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "tilde-manifest-package")
    direct_extension = _write(os.path.join(pkg_dir, "~extensions", "main.py"), _ext("main"))
    slash_extension = _write(os.path.join(pkg_dir, "~", "extensions", "alt.py"), _ext("alt"))
    direct_skill = _write(
        os.path.join(pkg_dir, "~skills", "direct-skill", "SKILL.md"),
        "---\nname: direct-skill\ndescription: Direct\n---\nContent",
    )
    slash_skill = _write(
        os.path.join(pkg_dir, "~", "skills", "slash-skill", "SKILL.md"),
        "---\nname: slash-skill\ndescription: Slash\n---\nContent",
    )
    _write(
        os.path.join(pkg_dir, "pi.json"),
        json.dumps(
            {
                "name": "tilde-manifest-package",
                "extensions": ["~extensions/main.py", "~/extensions/alt.py"],
                "skills": ["~skills", "~/skills"],
            }
        ),
    )

    result = await pm.resolve_extension_sources([pkg_dir])

    assert any(r.path == direct_extension and r.enabled for r in result.extensions)
    assert any(r.path == slash_extension and r.enabled for r in result.extensions)
    assert any(r.path == direct_skill and r.enabled for r in result.skills)
    assert any(r.path == slash_skill and r.enabled for r in result.skills)


async def test_resolve_extension_sources_handles_auto_discovery_layout(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "auto-pkg")
    _write(os.path.join(pkg_dir, "extensions", "main.py"), _ext("main"))
    _write(os.path.join(pkg_dir, "themes", "dark.json"), "{}")

    result = await pm.resolve_extension_sources([pkg_dir])
    assert _is_enabled(result.extensions, "main.py")
    assert _is_enabled(result.themes, "dark.json")


async def test_resolve_extension_sources_stops_recursing_at_skill_md(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "skill-root-pkg")
    root_skill = _write(
        os.path.join(pkg_dir, "skills", "root-skill", "SKILL.md"),
        "---\nname: root-skill\ndescription: Root skill\n---\n",
    )
    nested_skill = _write(
        os.path.join(pkg_dir, "skills", "root-skill", "nested-skill", "SKILL.md"),
        "---\nname: nested-skill\ndescription: Nested skill\n---\n",
    )

    result = await pm.resolve_extension_sources([pkg_dir])
    assert any(r.path == root_skill and r.enabled for r in result.skills)
    assert not any(r.path == nested_skill for r in result.skills)


async def test_resolve_extension_sources_handles_mixed_top_level_files_and_subdirectories(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "mixed-pkg")
    _write(os.path.join(pkg_dir, "extensions", "simple.py"), _ext("simple"))
    _write(os.path.join(pkg_dir, "extensions", "complex", "__init__.py"), _ext("complex"))
    _write(os.path.join(pkg_dir, "extensions", "complex", "a.py"), "A = 1\n")
    _write(os.path.join(pkg_dir, "extensions", "complex", "b.py"), "B = 2\n")

    result = await pm.resolve_extension_sources([pkg_dir])

    assert _is_enabled(result.extensions, "simple.py")
    assert _is_enabled(result.extensions, "complex/__init__.py")
    assert not _contains(result.extensions, "complex/a.py")
    assert not _contains(result.extensions, "complex/b.py")
    assert len([r for r in result.extensions if r.enabled]) == 2


# ---------------------------------------------------------------------------
# progress callback / command spawning
# ---------------------------------------------------------------------------


async def test_progress_callback_emits_no_events_for_local_paths(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    events: list[ProgressEvent] = []
    pm.set_progress_callback(events.append)

    ext_path = _write(os.path.join(cwd, "ext.py"), _ext())
    await pm.resolve_extension_sources([ext_path])

    assert len(events) == 0


async def test_command_runner_preserves_argv_entries_containing_spaces(tmp_path):
    """Ported from 'command spawning > should preserve argv entries containing spaces'.

    TypeScript spawns `process.execPath`; the Python equivalent is `sys.executable`.
    """
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    value_with_space = "C:\\Users\\A B\\.pi\\npm"
    output = await pm._runner.run_capture(
        sys.executable,
        ["-c", "import sys; print(sys.argv[1])", value_with_space],
    )
    assert output == value_with_space


# ---------------------------------------------------------------------------
# force-include patterns
# ---------------------------------------------------------------------------


async def test_force_include_overrides_exclude_in_package_filters(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "force-pkg")
    for name in ("alpha", "beta", "gamma"):
        _write(os.path.join(pkg_dir, "extensions", f"{name}.py"), _ext(name))

    settings.set_packages(
        [
            {
                "source": pkg_dir,
                "extensions": ["!**/*.py", "+extensions/beta.py"],
                "skills": [],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await pm.resolve()
    assert _is_disabled(result.extensions, "alpha.py")
    assert _is_enabled(result.extensions, "beta.py")
    assert _is_disabled(result.extensions, "gamma.py")


async def test_force_include_multiple_resources(tmp_path):
    pm, settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "multi-force-pkg")
    for name in ("skill-a", "skill-b", "skill-c"):
        _write(
            os.path.join(pkg_dir, "skills", name, "SKILL.md"),
            f"---\nname: {name}\ndescription: {name}\n---\nContent",
        )

    settings.set_packages(
        [
            {
                "source": pkg_dir,
                "extensions": [],
                "skills": ["!**/*", "+skills/skill-a", "+skills/skill-c"],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await pm.resolve()
    assert _contains_enabled(result.skills, "skill-a")
    assert any("skill-b" in r.path.replace("\\", "/") and not r.enabled for r in result.skills)
    assert _contains_enabled(result.skills, "skill-c")


async def test_force_include_after_specific_exclusion(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    _write(os.path.join(agent_dir, "extensions", "a.py"), _ext("a"))
    _write(os.path.join(agent_dir, "extensions", "b.py"), _ext("b"))

    settings.set_extension_paths(["extensions", "!extensions/b.py", "+extensions/b.py"])

    result = await pm.resolve()
    assert _is_enabled(result.extensions, "a.py")
    assert _is_enabled(result.extensions, "b.py")


async def test_force_include_in_manifest_patterns(tmp_path):
    pm, _settings, cwd, _agent_dir = _make_manager(tmp_path)
    pkg_dir = os.path.join(cwd, "manifest-force-pkg")
    for name in ("one", "two", "three"):
        _write(os.path.join(pkg_dir, "extensions", f"{name}.py"), _ext(name))
    _write(
        os.path.join(pkg_dir, "pi.json"),
        json.dumps({"name": "manifest-force-pkg", "extensions": ["extensions", "!**/two.py", "+extensions/two.py"]}),
    )

    result = await pm.resolve_extension_sources([pkg_dir])
    assert _is_enabled(result.extensions, "one.py")
    assert _is_enabled(result.extensions, "two.py")
    assert _is_enabled(result.extensions, "three.py")


async def test_force_include_themes(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    for name in ("dark", "light", "special"):
        _write(os.path.join(agent_dir, "themes", f"{name}.json"), "{}")

    settings.set_theme_paths(["themes", "!themes/*.json", "+themes/special.json"])

    result = await pm.resolve()
    assert _is_disabled(result.themes, "dark.json")
    assert _is_disabled(result.themes, "light.json")
    assert _is_enabled(result.themes, "special.json")


async def test_force_include_prompts(tmp_path):
    pm, settings, _cwd, agent_dir = _make_manager(tmp_path)
    for name in ("review", "explain", "debug"):
        _write(os.path.join(agent_dir, "prompts", f"{name}.md"), name.title())

    settings.set_prompt_template_paths(["prompts", "!prompts/*.md", "+prompts/debug.md"])

    result = await pm.resolve()
    assert _is_disabled(result.prompts, "review.md")
    assert _is_disabled(result.prompts, "explain.md")
    assert _is_enabled(result.prompts, "debug.md")


# ---------------------------------------------------------------------------
# Offline mode (`PI_OFFLINE`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("Yes", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("", False),
        ("maybe", False),
    ],
)
def test_offline_mode_is_enabled_only_by_the_documented_values(monkeypatch, value, expected):
    monkeypatch.setenv("PI_OFFLINE", value)
    assert _is_offline_mode_enabled() is expected


def test_offline_mode_is_disabled_when_pi_offline_is_unset(monkeypatch):
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    assert _is_offline_mode_enabled() is False


async def test_skips_installing_missing_package_sources_when_offline(tmp_path, monkeypatch):
    """Port of "should skip installing missing package sources when offline".

    TS configures both an `npm:` and a `git:` missing source and spies on
    `installParsedSource`. `npm:` sources raise in this port (see the module
    docstring), so only the git half is configured; the assertion that no
    resource carries `origin == "package"` and that nothing was spawned is
    unchanged.
    """
    monkeypatch.setenv("PI_OFFLINE", "1")
    runner = FakeCommandRunner()
    pm, settings_manager, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=runner)
    settings_manager.set_project_packages(["git:github.com/example/missing-repo"])

    result = await pm.resolve()

    all_resources = [*result.extensions, *result.skills, *result.prompts, *result.themes]
    assert not any(r.metadata.origin == "package" for r in all_resources)
    # TS asserts `installParsedSource` was never called; the observable
    # equivalent here is that no git subprocess was ever spawned.
    assert runner.calls == []


async def test_installs_missing_package_sources_when_not_offline(tmp_path, monkeypatch):
    """Control for the case above: the offline guard, not a missing checkout,
    is what suppresses the install."""
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    runner = FakeCommandRunner()
    pm, settings_manager, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=runner)
    settings_manager.set_project_packages(["git:github.com/example/missing-repo"])

    await pm.resolve()

    assert any(command == "git" and args[0] == "clone" for command, args, _cwd in runner.calls)


async def test_skips_refreshing_temporary_git_sources_when_offline(tmp_path, monkeypatch):
    """Port of "should skip refreshing temporary git sources when offline"."""
    monkeypatch.setenv("PI_OFFLINE", "1")
    runner = FakeCommandRunner()
    pm, _settings_manager, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=runner)
    git_source = "git:github.com/example/repo"
    parsed = pm.parse_source(git_source)
    installed_path = pm._get_git_install_path(parsed, "temporary")
    _write(os.path.join(installed_path, "extensions", "__init__.py"), _ext())

    result = await pm.resolve_extension_sources([git_source], temporary=True)

    assert _is_enabled(result.extensions, "extensions/__init__.py")
    # TS asserts `refreshTemporaryGitSource` was never called; the observable
    # equivalent is that no git subprocess ran against the existing checkout.
    assert runner.calls == []


async def test_refreshes_temporary_git_sources_when_not_offline(tmp_path, monkeypatch):
    """Control for the case above."""
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    runner = FakeCommandRunner()
    pm, _settings_manager, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=runner)
    git_source = "git:github.com/example/repo"
    installed_path = pm._get_git_install_path(pm.parse_source(git_source), "temporary")
    _write(os.path.join(installed_path, "extensions", "__init__.py"), _ext())

    await pm.resolve_extension_sources([git_source], temporary=True)

    assert runner.calls != []


async def test_does_not_check_package_updates_when_offline(tmp_path, monkeypatch):
    """Port of "should not check package updates when offline"."""
    monkeypatch.setenv("PI_OFFLINE", "1")
    remote = str(tmp_path / "remote")
    _init_local_git_remote(remote, {"extensions/__init__.py": _ext()})
    runner = FakeCommandRunner()
    pm, settings_manager, _cwd, _agent_dir = _make_manager(tmp_path, command_runner=runner)
    settings_manager.set_project_packages([f"git:{remote}"])

    updates = await pm.check_for_available_updates()

    assert updates == []
    # TS asserts `runCommandCapture` was never called.
    assert runner.calls == []


def test_temporary_install_paths_stay_under_the_agent_temp_extension_folder(tmp_path):
    """Port of "should place temporary npm packages under the agent temp extension folder".

    TS exercises `getNpmInstallPath(source, "temporary")`; this port has no npm
    sources, so the same claim is made about the git temporary path, which uses
    the identical `getExtensionTempFolder` root.
    """
    pm, _settings_manager, _cwd, agent_dir = _make_manager(tmp_path)
    source = pm.parse_source("git:github.com/example/repo")

    install_path = pm._get_git_install_path(source, "temporary")
    temp_root = os.path.join(agent_dir, "tmp", "extensions")

    assert not os.path.relpath(install_path, temp_root).startswith("..")
    assert not install_path.startswith(tempfile.gettempdir() + os.sep + "pi-extensions")
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(temp_root).st_mode) == 0o700


# ---------------------------------------------------------------------------
# Git checkout reconciliation and clone-failure cleanup
# ---------------------------------------------------------------------------


class ScriptedGitRunner(CommandRunner):
    """A `CommandRunner` that records every call and can script side effects.

    TypeScript's equivalent cases `vi.spyOn` the manager's own private
    `runCommand`/`runCommandCapture`. Overriding the real `CommandRunner`
    seam instead keeps the manager itself entirely unmocked, so the
    ordering and arguments asserted below are the ones production issues.
    """

    def __init__(self, *, capture=None, on_run=None):
        self.calls: list[tuple[str, list[str], str | None]] = []
        self._capture = capture or (lambda command, args, cwd: "")
        self._on_run = on_run

    async def run(self, command, args, *, cwd=None):
        self.calls.append((command, args, cwd))
        if self._on_run is not None:
            self._on_run(command, args, cwd)

    async def run_capture(self, command, args, *, cwd=None, timeout=None, env=None):
        self.calls.append((command, args, cwd))
        return self._capture(command, args, cwd)

    def git_args(self) -> list[list[str]]:
        return [args for command, args, _cwd in self.calls if command == "git"]


async def test_removes_a_newly_created_checkout_when_git_clone_fails(tmp_path):
    """Port of "should remove a newly created checkout when git clone fails"."""
    source = "git:github.com/user/repo"
    pm, _settings_manager, _cwd, agent_dir = _make_manager(tmp_path)
    target_dir = os.path.join(agent_dir, "git", "github.com", "user", "repo")

    def _on_run(command, args, cwd):
        if command == "git" and args[0] == "clone":
            os.makedirs(target_dir, exist_ok=True)
            raise RuntimeError("simulated git clone failure")

    pm._runner = ScriptedGitRunner(on_run=_on_run)

    with pytest.raises(RuntimeError, match="simulated git clone failure"):
        await pm.install(source)

    assert not os.path.exists(target_dir)


async def test_reconciles_an_existing_git_checkout_to_a_pinned_ref_during_install(tmp_path):
    """Port of "should reconcile an existing git checkout to a pinned ref during install"."""
    source = "git:github.com/user/repo@v2"
    pm, _settings_manager, _cwd, agent_dir = _make_manager(tmp_path)
    target_dir = os.path.join(agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(target_dir, exist_ok=True)

    def _capture(command, args, cwd):
        if args[:2] == ["rev-parse", "HEAD"]:
            return "old-head"
        if args[:2] == ["rev-parse", "FETCH_HEAD^{commit}"]:
            return "new-head"
        raise AssertionError(f"Unexpected run_capture args: {' '.join(args)}")

    runner = ScriptedGitRunner(capture=_capture)
    pm._runner = runner

    await pm.install(source)

    assert ["fetch", "origin", "v2"] in runner.git_args()
    assert ["reset", "--hard", "FETCH_HEAD^{commit}"] in runner.git_args()
    assert ["clean", "-fdx"] in runner.git_args()
    assert all(cwd == target_dir for command, args, cwd in runner.calls if args[0] != "clone")
    # TS additionally asserts `npm install --omit=dev` runs in the checkout.
    # Git packages carry no installable dependency manifest in this port (see
    # the module docstring), so no dependency install step exists.
    assert not any(command == "npm" for command, _args, _cwd in runner.calls)


async def test_reconciles_an_existing_checkout_to_its_update_target_without_a_ref(tmp_path):
    """Port of "should reconcile an existing git checkout to its update target when installing without a ref".

    TS mocks `getLocalGitUpdateTarget` outright; here the real method runs and
    the upstream lookup it performs is scripted through the command runner, so
    the derived `fetchArgs` are production's, not the test's.
    """
    source = "git:github.com/user/repo"
    pm, _settings_manager, _cwd, agent_dir = _make_manager(tmp_path)
    target_dir = os.path.join(agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(target_dir, exist_ok=True)

    def _capture(command, args, cwd):
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            raise RuntimeError("no upstream configured")
        if args[0] == "symbolic-ref":
            return "refs/remotes/origin/main\n"
        if args[:2] == ["rev-parse", "HEAD"]:
            return "old-head"
        if args[:2] == ["rev-parse", "origin/HEAD^{commit}"]:
            return "new-head"
        raise AssertionError(f"Unexpected run_capture args: {' '.join(args)}")

    runner = ScriptedGitRunner(capture=_capture)
    pm._runner = runner

    await pm.install(source)

    assert ["fetch", "--prune", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"] in runner.git_args()
    assert ["reset", "--hard", "origin/HEAD^{commit}"] in runner.git_args()
    assert ["clean", "-fdx"] in runner.git_args()


async def test_reconciliation_is_skipped_when_the_checkout_already_matches_the_target(tmp_path):
    """The other half of `_ensure_git_ref`: matching heads must not reset or clean."""
    pm, _settings_manager, _cwd, agent_dir = _make_manager(tmp_path)
    target_dir = os.path.join(agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(target_dir, exist_ok=True)

    runner = ScriptedGitRunner(capture=lambda command, args, cwd: "same-head")
    pm._runner = runner

    await pm.install("git:github.com/user/repo@v2")

    assert ["fetch", "origin", "v2"] in runner.git_args()
    assert not any(args[0] in ("reset", "clean") for args in runner.git_args())


async def test_progress_callback_emits_start_and_error_events_on_a_failed_install(tmp_path):
    """Port of "should emit progress events on install attempt".

    TS drives an `npm:` source; `npm:` raises at parse time here (module
    docstring), so a git source is used. The claim -- a `start`/`install`
    event and an `error` event around a failing install -- is unchanged.
    """
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    pm._runner = ScriptedGitRunner(
        on_run=lambda command, args, cwd: (_ for _ in ()).throw(RuntimeError("simulated git clone failure"))
    )
    events: list[ProgressEvent] = []
    pm.set_progress_callback(events.append)

    with pytest.raises(RuntimeError, match="simulated git clone failure"):
        await pm.install("git:github.com/nonexistent/repo")

    assert any(e.type == "start" and e.action == "install" for e in events)
    assert any(e.type == "error" for e in events)


async def test_run_capture_returns_the_complete_stdout_of_a_process_that_outlives_its_exit(tmp_path):
    """Port of "should wait for close before resolving captured stdout".

    TypeScript pins a Node-specific hazard: `child.on("exit")` fires before the
    stdout pipe has drained, so an implementation that resolves on `exit`
    silently truncates output. `SubprocessCommandRunner.run_capture` uses
    `Process.communicate()`, which reads both pipes to EOF *before* reaping the
    process, so the race cannot occur -- but the observable claim is testable:
    output larger than a pipe buffer (64 KiB) must come back whole, which is
    exactly what an exit-race implementation would truncate.
    """
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)
    line_count = 20000

    output = await pm._runner.run_capture(
        sys.executable,
        ["-c", f"import sys\nfor i in range({line_count}): sys.stdout.write(f'line-{{i}}\\n')"],
    )

    lines = output.split("\n")
    assert len(lines) == line_count
    assert lines[0] == "line-0"
    assert lines[-1] == f"line-{line_count - 1}"


async def test_run_capture_raises_with_the_child_stderr_when_the_command_fails(tmp_path):
    pm, _settings, _cwd, _agent_dir = _make_manager(tmp_path)

    with pytest.raises(RuntimeError, match="failed with code 3: boom"):
        await pm._runner.run_capture(
            sys.executable,
            ["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        )


# ---------------------------------------------------------------------------
# Bounded-parallel git updates and update checks
# ---------------------------------------------------------------------------


class _ConcurrencyProbe:
    """Records how many operations are in flight at once, without sleeping.

    Each operation parks on its own `asyncio.Event`; once `expected` of them
    are parked simultaneously the probe releases them all. If the code under
    test were sequential the second operation would never start, so
    `wait_for_peak()` would hang -- hence the timeout rather than a sleep.
    """

    def __init__(self, expected: int):
        self.active = 0
        self.peak = 0
        self._expected = expected
        self._reached = asyncio.Event()
        self._release = asyncio.Event()

    async def enter(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active >= self._expected:
            self._reached.set()
            self._release.set()
        await self._release.wait()
        self.active -= 1

    async def wait_for_peak(self, timeout: float = 5.0) -> None:
        await asyncio.wait_for(self._reached.wait(), timeout=timeout)


async def test_update_runs_git_updates_in_parallel(tmp_path):
    """Port of the git half of "should batch npm updates per scope and run git
    updates in parallel while skipping pinned npm and current packages".

    TS asserts `maxConcurrentGitUpdates > 1`. The npm half of that case has no
    counterpart here (module docstring), and `updateGit` is not stubbed out --
    the real `_update_git` runs, with only the command runner scripted.
    """
    probe = _ConcurrencyProbe(expected=3)
    pm, settings_manager, _cwd, _agent_dir = _make_manager(tmp_path)

    class _ParkingRunner(CommandRunner):
        async def run(self, command, args, *, cwd=None):
            await probe.enter()

        async def run_capture(self, command, args, *, cwd=None, timeout=None, env=None):
            return "same-head"

    pm._runner = _ParkingRunner()
    sources = [
        "git:github.com/example/repo-a",
        "git:github.com/example/repo-b",
        "git:github.com/example/repo-pinned@v1",
    ]
    for source in sources:
        os.makedirs(pm._get_git_install_path(pm.parse_source(source), "user"), exist_ok=True)
    settings_manager.set_packages(sources)

    await asyncio.gather(pm.update(), probe.wait_for_peak())

    assert probe.peak > 1
    # TS includes pinned git refs in the update set ("configured checkout
    # targets"), unlike pinned npm versions; all three sources ran.
    assert probe.peak == 3


async def test_check_for_available_updates_runs_checks_in_parallel(tmp_path):
    """`checkForAvailableUpdates` uses `runWithConcurrency(UPDATE_CHECK_CONCURRENCY)`."""
    probe = _ConcurrencyProbe(expected=3)
    pm, settings_manager, _cwd, _agent_dir = _make_manager(tmp_path)

    class _ParkingRunner(CommandRunner):
        async def run(self, command, args, *, cwd=None):
            return None

        async def run_capture(self, command, args, *, cwd=None, timeout=None, env=None):
            if args[:2] == ["rev-parse", "HEAD"]:
                await probe.enter()
                return "a" * 40
            if args[0] == "ls-remote":
                return f"{'b' * 40}\tHEAD\n"
            raise RuntimeError("no upstream configured")

    pm._runner = _ParkingRunner()
    sources = [
        "git:github.com/example/repo-a",
        "git:github.com/example/repo-b",
        "git:github.com/example/repo-c",
    ]
    for source in sources:
        os.makedirs(pm._get_git_install_path(pm.parse_source(source), "user"), exist_ok=True)
    settings_manager.set_packages(sources)

    updates, _ = await asyncio.gather(pm.check_for_available_updates(), probe.wait_for_peak())

    assert probe.peak > 1
    # Results keep the order of the configured sources, not completion order.
    assert [u.source for u in updates] == sources


async def test_check_for_available_updates_skips_pinned_and_missing_checkouts(tmp_path):
    pm, settings_manager, _cwd, _agent_dir = _make_manager(tmp_path)

    class _AlwaysBehindRunner(CommandRunner):
        async def run(self, command, args, *, cwd=None):
            return None

        async def run_capture(self, command, args, *, cwd=None, timeout=None, env=None):
            if args[:2] == ["rev-parse", "HEAD"]:
                return "a" * 40
            if args[0] == "ls-remote":
                return f"{'b' * 40}\tHEAD\n"
            raise RuntimeError("no upstream configured")

    pm._runner = _AlwaysBehindRunner()
    installed = "git:github.com/example/installed"
    pinned = "git:github.com/example/pinned@v1"
    missing = "git:github.com/example/missing"
    os.makedirs(pm._get_git_install_path(pm.parse_source(installed), "user"), exist_ok=True)
    os.makedirs(pm._get_git_install_path(pm.parse_source(pinned), "user"), exist_ok=True)
    settings_manager.set_packages([installed, pinned, missing])

    updates = await pm.check_for_available_updates()

    assert [u.source for u in updates] == [installed]
    assert updates[0].display_name == "github.com/example/installed"
    assert updates[0].type == "git"
    assert updates[0].scope == "user"
