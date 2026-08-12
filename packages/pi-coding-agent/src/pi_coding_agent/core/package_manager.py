"""Resource package manager: install/update/remove/list/resolve packages.

Python port of `packages/coding-agent/src/core/package-manager.ts` (2677
lines): discovers and resolves extension/skill/prompt/theme resources from
git repositories, local paths, and the auto-discovered project/user
directories `resource_loader.py` and `core/extensions/loader.py` also read,
applies the enable/disable override patterns (`!exclude`, `+force-include`,
`-force-exclude`) those two modules' `ResourceLoader`/`discover_and_load_extensions`
consume, and persists installed sources into `SettingsManager`'s `packages`
list (the "manifest" of what's installed -- there is no separate lockfile;
the settings.json `packages` array *is* the installed-package record, exactly
as in the TypeScript).

**npm omission.** The TypeScript source supports three kinds of sources:
`npm:<spec>` (installed via `npm`/`bun`/`pnpm` into a managed
`node_modules`), `git:<url>`/bare protocol URLs (cloned with `git`), and
local filesystem paths. There is no Python equivalent of an npm registry or
a JavaScript package manager, and faking one (e.g. treating `npm:foo` as a
PyPI package) would silently do something entirely different from what the
TypeScript does. This port therefore only implements the **git** and
**local path** sources faithfully; `parse_source()` raises `ValueError` for
`npm:`-prefixed input with a message explaining the omission, and
`install()`/`update()`/`remove()` surface that error rather than silently
no-op-ing.

**No dependency-install step for git sources.** The TypeScript clones a git
extension and then runs `npm install` inside it if it has a `package.json`
(`getGitDependencyInstallArgs`, `repairMissingGitDependencies`,
`cleanAndInstallGitDependencies`). Python extensions (see
`core/extensions/loader.py`'s module docstring) are plain `.py` files with no
third-party dependency-manifest convention to install from, so this port
drops that entire dependency-install/repair subsystem; `git clone`/`fetch`/
`reset --hard`/`clean -fdx` are still ported faithfully since they matter for
keeping a checkout pristine and reconciling it to a ref.

**package.json manifest -> pi.json.** See `core/pi_manifest.py`'s module
docstring: the per-package resource manifest (`extensions`/`skills`/
`prompts`/`themes` arrays) is read from a top-level `pi.json` file instead of
a nested `"pi"` field inside a JavaScript `package.json`.

**Extension file convention.** Resource collection for the "extensions" type
uses the `.py` file / `subdir/__init__.py` convention `core/extensions/
loader.py` already establishes (its `discover_extensions_in_dir`), not
TypeScript's `.ts`/`.js` / `index.ts`/`index.js`.

**Self-update / npm registry checks omitted.** `checkForAvailableUpdates()`'s
npm-registry-version-lookup half and the entire `update --self`/`update
--models` machinery in `package-manager-cli.ts` (self-update via the package
manager that installed `pi`, model catalog refresh) have no meaning here --
see `config.py`'s module docstring for the same "no self-update story"
scope decision. `update()` here only reconciles git-sourced packages.

**Trust gating.** Exactly like `resource_loader.py` and `core/extensions/
loader.py`: project-local package storage (`<cwd>/.pi/{git,extensions,
skills,prompts,themes}`) and the project `packages` settings entries are
only ever touched when `settings_manager.is_project_trusted()` is true.
`install()`/`remove()` raise immediately for `scope="project"` on an
untrusted project (`assert_project_trusted_for_scope`), matching
`DefaultPackageManager.assertProjectTrustedForScope` exactly -- this is the
one piece of trust enforcement that guards against installing executable
extension code into project-local storage without the user having trusted
the project first.

**Subprocess injection.** All `git` invocations go through a `CommandRunner`
(default: `asyncio.create_subprocess_exec`), injectable via the
`command_runner` constructor parameter, so tests can substitute a fake
runner instead of shelling out. All root directories (`cwd`, `agent_dir`)
are constructor parameters with no hidden real-`$HOME` fallback, so tests
never touch the real user config directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pi_coding_agent.core.config import CONFIG_DIR_NAME
from pi_coding_agent.core.pi_manifest import read_pi_manifest
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.tools.gitignore import GitignoreMatcher
from pi_coding_agent.utils.git import GitSource, parse_git_url
from pi_coding_agent.utils.paths import resolve_path

NETWORK_TIMEOUT_S = 10.0

# Mirrors `UPDATE_CHECK_CONCURRENCY` / `GIT_UPDATE_CONCURRENCY` in
# `core/package-manager.ts`: update checks and git updates run bounded-parallel,
# not one after another.
UPDATE_CHECK_CONCURRENCY = 4
GIT_UPDATE_CONCURRENCY = 4

SourceScope = Literal["user", "project", "temporary"]
SourceOrigin = Literal["package", "top-level"]
ResourceType = Literal["extensions", "skills", "prompts", "themes"]
MissingSourceAction = Literal["install", "skip", "error"]

RESOURCE_TYPES: tuple[ResourceType, ...] = ("extensions", "skills", "prompts", "themes")

_IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")


# --------------------------------------------------------------------------
# Public data shapes
# --------------------------------------------------------------------------


_T = TypeVar("_T")


async def _run_with_concurrency(tasks: list[Callable[[], Awaitable[_T]]], limit: int) -> list[_T]:
    """Run `tasks` with at most `limit` in flight, preserving result order.

    Port of `runWithConcurrency` in `core/package-manager.ts`.
    """
    if not tasks:
        return []

    results: list[_T] = [None] * len(tasks)  # type: ignore[list-item]
    next_index = 0
    worker_count = max(1, min(limit, len(tasks)))

    async def worker() -> None:
        nonlocal next_index
        while True:
            index = next_index
            next_index += 1
            if index >= len(tasks):
                return
            results[index] = await tasks[index]()

    await asyncio.gather(*(worker() for _ in range(worker_count)))
    return results


def _is_offline_mode_enabled() -> bool:
    """Whether startup network operations are disabled.

    Mirrors the TypeScript `isOfflineModeEnabled`: only "1", "true" or "yes"
    (case-insensitive) enable it, so an empty or unrelated value does not.
    """
    value = os.environ.get("PI_OFFLINE")
    if not value:
        return False
    return value == "1" or value.lower() in ("true", "yes")


@dataclass
class PathMetadata:
    source: str
    scope: SourceScope
    origin: SourceOrigin
    base_dir: str | None = None


@dataclass
class ResolvedResource:
    path: str
    enabled: bool
    metadata: PathMetadata


@dataclass
class ResolvedPaths:
    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)


@dataclass
class ProgressEvent:
    type: Literal["start", "progress", "complete", "error"]
    action: Literal["install", "remove", "update", "clone", "pull"]
    source: str
    message: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class PackageUpdate:
    source: str
    display_name: str
    type: Literal["git"] = "git"
    scope: Literal["user", "project"] = "user"


@dataclass
class ConfiguredPackage:
    source: str
    scope: Literal["user", "project"]
    filtered: bool
    installed_path: str | None = None


@dataclass
class LocalSource:
    path: str
    type: str = "local"


ParsedSource = GitSource | LocalSource


@dataclass
class PackageFilter:
    autoload: bool | None = None
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None

    def get(self, resource_type: ResourceType) -> list[str] | None:
        return getattr(self, resource_type)


# A settings `packages` entry is either a bare source string or a dict with a
# `source` key plus optional filter fields (matches TypeScript's
# `PackageSource = string | { source, autoload?, extensions?, ... }`).
PackageSourceEntry = str | dict[str, Any]


def _package_source_string(pkg: PackageSourceEntry) -> str:
    return pkg if isinstance(pkg, str) else str(pkg["source"])


def _package_filter(pkg: PackageSourceEntry) -> PackageFilter | None:
    if isinstance(pkg, str):
        return None
    return PackageFilter(
        autoload=pkg.get("autoload"),
        extensions=pkg.get("extensions"),
        skills=pkg.get("skills"),
        prompts=pkg.get("prompts"),
        themes=pkg.get("themes"),
    )


# --------------------------------------------------------------------------
# Command execution (injectable)
# --------------------------------------------------------------------------


class CommandRunner:
    """Runs external commands. Overridden by tests with a fake runner."""

    async def run(self, command: str, args: list[str], *, cwd: str | None = None) -> None:
        raise NotImplementedError

    async def run_capture(
        self,
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        raise NotImplementedError


class SubprocessCommandRunner(CommandRunner):
    """Default `CommandRunner`: shells out via `asyncio.create_subprocess_exec`."""

    async def run(self, command: str, args: list[str], *, cwd: str | None = None) -> None:
        process = await asyncio.create_subprocess_exec(command, *args, cwd=cwd)
        returncode = await process.wait()
        if returncode != 0:
            raise RuntimeError(f"{command} {' '.join(args)} failed with code {returncode}")

    async def run_capture(
        self,
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        run_env = {**os.environ, **env} if env else None
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env=run_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"{command} {' '.join(args)} timed out after {timeout}s") from None

        if process.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", "replace")
            raise RuntimeError(f"{command} {' '.join(args)} failed with code {process.returncode}: {detail}")
        return stdout.decode("utf-8", "replace").strip()


def get_extension_temp_folder(agent_dir: str) -> str:
    temp_folder = os.path.join(agent_dir, "tmp", "extensions")
    os.makedirs(temp_folder, mode=0o700, exist_ok=True)
    os.chmod(temp_folder, 0o700)
    return temp_folder


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------


def _to_posix(p: str) -> str:
    return p.replace(os.sep, "/")


def _canonicalize_path(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def _get_home_dir() -> str:
    return os.environ.get("HOME") or os.path.expanduser("~")


# --------------------------------------------------------------------------
# minimatch-equivalent glob matching
# --------------------------------------------------------------------------


def _expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b}` alternations, as minimatch does by default.

    Without this a disable pattern such as `!{foo,bar}.md` can never match, so
    the resource stays enabled — a fail-open outcome for patterns that gate
    executable extension code.
    """
    start = -1
    depth = 0
    for index, char in enumerate(pattern):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                prefix, body, suffix = pattern[:start], pattern[start + 1 : index], pattern[index + 1 :]
                options: list[str] = []
                inner = 0
                current = ""
                for piece in body:
                    if piece == "," and inner == 0:
                        options.append(current)
                        current = ""
                        continue
                    if piece == "{":
                        inner += 1
                    elif piece == "}":
                        inner -= 1
                    current += piece
                options.append(current)
                expanded: list[str] = []
                for option in options:
                    expanded.extend(_expand_braces(prefix + option + suffix))
                return expanded
    return [pattern]


def _glob_segment_to_regex(pattern: str) -> str:
    """Compile one already-brace-expanded glob into an unanchored regex body."""
    i, n = 0, len(pattern)
    out: list[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                # A globstar matches zero or more path segments, so consume the
                # following separator: `a/**/b` must match `a/b`.
                if pattern[i : i + 3] == "**/":
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Anchored full-match regex for a `minimatch`-style glob.

    Supports `*`, `**` (including the zero-segment case), `?`, `[...]` and
    `{a,b}` brace expansion.
    """
    alternatives = [_glob_segment_to_regex(expanded) for expanded in _expand_braces(pattern)]
    if len(alternatives) == 1:
        return re.compile("^" + alternatives[0] + "$")
    return re.compile("^(?:" + "|".join(alternatives) + ")$")


def _minimatch(value: str, pattern: str) -> bool:
    return _glob_to_regex(pattern).match(value) is not None


def is_pattern(s: str) -> bool:
    return s.startswith(("!", "+", "-")) or "*" in s or "?" in s


def is_override_pattern(s: str) -> bool:
    return s.startswith(("!", "+", "-"))


def has_glob_pattern(s: str) -> bool:
    return "*" in s or "?" in s


def split_patterns(entries: list[str]) -> tuple[list[str], list[str]]:
    plain, patterns = [], []
    for entry in entries:
        (patterns if is_pattern(entry) else plain).append(entry)
    return plain, patterns


def _skill_parent_parts(file_path: str, base_dir: str) -> tuple[str, str, str] | None:
    if os.path.basename(file_path) != "SKILL.md":
        return None
    parent_dir = os.path.dirname(file_path)
    parent_rel = _to_posix(os.path.relpath(parent_dir, base_dir))
    parent_name = os.path.basename(parent_dir)
    parent_dir_posix = _to_posix(parent_dir)
    return parent_rel, parent_name, parent_dir_posix


def matches_any_pattern(file_path: str, patterns: list[str], base_dir: str) -> bool:
    rel = _to_posix(os.path.relpath(file_path, base_dir))
    name = os.path.basename(file_path)
    file_path_posix = _to_posix(file_path)
    skill_parts = _skill_parent_parts(file_path, base_dir)

    for pattern in patterns:
        normalized_pattern = _to_posix(pattern)
        if (
            _minimatch(rel, normalized_pattern)
            or _minimatch(name, normalized_pattern)
            or _minimatch(file_path_posix, normalized_pattern)
        ):
            return True
        if skill_parts is None:
            continue
        parent_rel, parent_name, parent_dir_posix = skill_parts
        if (
            _minimatch(parent_rel, normalized_pattern)
            or _minimatch(parent_name, normalized_pattern)
            or _minimatch(parent_dir_posix, normalized_pattern)
        ):
            return True
    return False


def _normalize_exact_pattern(pattern: str) -> str:
    normalized = pattern[2:] if pattern.startswith(("./", ".\\")) else pattern
    return _to_posix(normalized)


def matches_any_exact_pattern(file_path: str, patterns: list[str], base_dir: str) -> bool:
    if not patterns:
        return False
    rel = _to_posix(os.path.relpath(file_path, base_dir))
    file_path_posix = _to_posix(file_path)
    skill_parts = _skill_parent_parts(file_path, base_dir)

    for pattern in patterns:
        normalized = _normalize_exact_pattern(pattern)
        if normalized in (rel, file_path_posix):
            return True
        if skill_parts is None:
            continue
        parent_rel, _parent_name, parent_dir_posix = skill_parts
        if normalized in (parent_rel, parent_dir_posix):
            return True
    return False


def get_override_patterns(entries: list[str]) -> list[str]:
    return [p for p in entries if p.startswith(("!", "+", "-"))]


def is_enabled_by_overrides(file_path: str, patterns: list[str], base_dir: str) -> bool:
    overrides = get_override_patterns(patterns)
    excludes = [p[1:] for p in overrides if p.startswith("!")]
    force_includes = [p[1:] for p in overrides if p.startswith("+")]
    force_excludes = [p[1:] for p in overrides if p.startswith("-")]

    enabled = True
    if excludes and matches_any_pattern(file_path, excludes, base_dir):
        enabled = False
    if force_includes and matches_any_exact_pattern(file_path, force_includes, base_dir):
        enabled = True
    if force_excludes and matches_any_exact_pattern(file_path, force_excludes, base_dir):
        enabled = False
    return enabled


def apply_patterns(all_paths: list[str], patterns: list[str], base_dir: str) -> set[str]:
    includes: list[str] = []
    excludes: list[str] = []
    force_includes: list[str] = []
    force_excludes: list[str] = []

    for p in patterns:
        if p.startswith("+"):
            force_includes.append(p[1:])
        elif p.startswith("-"):
            force_excludes.append(p[1:])
        elif p.startswith("!"):
            excludes.append(p[1:])
        else:
            includes.append(p)

    result = list(all_paths) if not includes else [f for f in all_paths if matches_any_pattern(f, includes, base_dir)]

    if excludes:
        result = [f for f in result if not matches_any_pattern(f, excludes, base_dir)]

    if force_includes:
        result_set = set(result)
        for f in all_paths:
            if f not in result_set and matches_any_exact_pattern(f, force_includes, base_dir):
                result.append(f)
                result_set.add(f)

    if force_excludes:
        result = [f for f in result if not matches_any_exact_pattern(f, force_excludes, base_dir)]

    return set(result)


def apply_autoload_disabled_patterns(all_paths: list[str], patterns: list[str], base_dir: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for pattern in patterns:
        has_prefix = pattern.startswith(("+", "-", "!"))
        target = pattern[1:] if has_prefix else pattern
        enabled = not pattern.startswith("-") and not pattern.startswith("!")
        exact = pattern.startswith(("+", "-"))
        for f in all_paths:
            matched = (
                matches_any_exact_pattern(f, [target], base_dir)
                if exact
                else matches_any_pattern(f, [target], base_dir)
            )
            if matched:
                result[f] = enabled
    return result


# --------------------------------------------------------------------------
# Ignore-file handling (reuses GitignoreMatcher; `.gitignore`/`.ignore`/`.fdignore`)
# --------------------------------------------------------------------------


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed or (trimmed.startswith("#") and not trimmed.startswith("\\#")):
        return None

    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]

    if pattern.startswith("/"):
        pattern = pattern[1:]

    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _add_ignore_rules(matcher: GitignoreMatcher, directory: str, root_dir: str) -> None:
    relative_dir = os.path.relpath(directory, root_dir)
    prefix = f"{_to_posix(relative_dir)}/" if relative_dir != "." else ""

    for filename in _IGNORE_FILE_NAMES:
        ignore_path = os.path.join(directory, filename)
        if not os.path.exists(ignore_path):
            continue
        try:
            with open(ignore_path, encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue
        patterns = [
            p for line in re.split(r"\r?\n", content) if (p := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            matcher.add(patterns)


# --------------------------------------------------------------------------
# Resource discovery
# --------------------------------------------------------------------------


def collect_files(
    directory: str,
    file_pattern: re.Pattern[str],
    skip_node_modules: bool = True,
    ignore_matcher: GitignoreMatcher | None = None,
    root_dir: str | None = None,
) -> list[str]:
    files: list[str] = []
    if not os.path.exists(directory):
        return files

    root = root_dir or directory
    matcher = ignore_matcher if ignore_matcher is not None else GitignoreMatcher()
    _add_ignore_rules(matcher, directory, root)

    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return files

    for entry in entries:
        if entry.name.startswith(".") or (skip_node_modules and entry.name == "node_modules"):
            continue

        full_path = os.path.join(directory, entry.name)
        try:
            is_dir = entry.is_dir(follow_symlinks=True)
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue

        rel_path = _to_posix(os.path.relpath(full_path, root))
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if matcher.is_ignored(ignore_path, is_dir):
            continue

        if is_dir:
            files.extend(collect_files(full_path, file_pattern, skip_node_modules, matcher, root))
        elif is_file and file_pattern.search(entry.name):
            files.append(full_path)

    return files


def collect_skill_entries(
    directory: str,
    mode: Literal["pi", "agents"],
    ignore_matcher: GitignoreMatcher | None = None,
    root_dir: str | None = None,
) -> list[str]:
    entries: list[str] = []
    if not os.path.exists(directory):
        return entries

    root = root_dir or directory
    matcher = ignore_matcher if ignore_matcher is not None else GitignoreMatcher()
    _add_ignore_rules(matcher, directory, root)

    try:
        dir_entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return entries

    for entry in dir_entries:
        if entry.name != "SKILL.md":
            continue
        full_path = os.path.join(directory, entry.name)
        try:
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue
        rel_path = _to_posix(os.path.relpath(full_path, root))
        if is_file and not matcher.is_ignored(rel_path, False):
            entries.append(full_path)
            return entries

    for entry in dir_entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue

        full_path = os.path.join(directory, entry.name)
        try:
            is_dir = entry.is_dir(follow_symlinks=True)
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue

        rel_path = _to_posix(os.path.relpath(full_path, root))
        if mode == "pi" and directory == root and is_file and entry.name.endswith(".md"):
            if not matcher.is_ignored(rel_path, False):
                entries.append(full_path)
            continue

        if not is_dir:
            continue
        if matcher.is_ignored(f"{rel_path}/", True):
            continue

        entries.extend(collect_skill_entries(full_path, mode, matcher, root))

    return entries


def collect_auto_skill_entries(directory: str, mode: Literal["pi", "agents"]) -> list[str]:
    return collect_skill_entries(directory, mode)


def find_git_repo_root(start_dir: str) -> str | None:
    directory = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(directory, ".git")):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def collect_ancestor_agents_skill_dirs(start_dir: str) -> list[str]:
    skill_dirs: list[str] = []
    resolved_start_dir = os.path.abspath(start_dir)
    git_repo_root = find_git_repo_root(resolved_start_dir)

    directory = resolved_start_dir
    while True:
        skill_dirs.append(os.path.join(directory, ".agents", "skills"))
        if git_repo_root and directory == git_repo_root:
            break
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent

    return skill_dirs


def _collect_auto_flat_entries(directory: str, suffix: str) -> list[str]:
    """Shared logic for `collect_auto_prompt_entries`/`collect_auto_theme_entries`."""
    entries: list[str] = []
    if not os.path.exists(directory):
        return entries

    matcher = GitignoreMatcher()
    _add_ignore_rules(matcher, directory, directory)

    try:
        dir_entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return entries

    for entry in dir_entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        full_path = os.path.join(directory, entry.name)
        try:
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue

        rel_path = _to_posix(os.path.relpath(full_path, directory))
        if matcher.is_ignored(rel_path, False):
            continue
        if is_file and entry.name.endswith(suffix):
            entries.append(full_path)

    return entries


def collect_auto_prompt_entries(directory: str) -> list[str]:
    return _collect_auto_flat_entries(directory, ".md")


def collect_auto_theme_entries(directory: str) -> list[str]:
    return _collect_auto_flat_entries(directory, ".json")


def resolve_extension_entries(directory: str) -> list[str] | None:
    """Resolve a directory's extension entry points: `pi.json` manifest, or `__init__.py`."""
    manifest_path = os.path.join(directory, "pi.json")
    manifest = read_pi_manifest(manifest_path)
    if manifest and manifest.extensions:
        resolved = [
            resolved_path
            for ext_path in manifest.extensions
            if os.path.exists(resolved_path := os.path.abspath(os.path.join(directory, ext_path)))
        ]
        if resolved:
            return resolved

    init_path = os.path.join(directory, "__init__.py")
    if os.path.isfile(init_path):
        return [init_path]
    return None


def collect_auto_extension_entries(directory: str) -> list[str]:
    if not os.path.exists(directory):
        return []

    root_entries = resolve_extension_entries(directory)
    if root_entries:
        return root_entries

    entries: list[str] = []
    matcher = GitignoreMatcher()
    _add_ignore_rules(matcher, directory, directory)

    try:
        dir_entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return entries

    for entry in dir_entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue

        full_path = os.path.join(directory, entry.name)
        try:
            is_dir = entry.is_dir(follow_symlinks=True)
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue

        rel_path = _to_posix(os.path.relpath(full_path, directory))
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if matcher.is_ignored(ignore_path, is_dir):
            continue

        if is_file and entry.name.endswith(".py"):
            entries.append(full_path)
        elif is_dir:
            resolved = resolve_extension_entries(full_path)
            if resolved:
                entries.extend(resolved)

    return entries


_FILE_SUFFIX_PATTERNS: dict[ResourceType, re.Pattern[str]] = {
    "extensions": re.compile(r"\.py$"),
    "skills": re.compile(r"\.md$"),
    "prompts": re.compile(r"\.md$"),
    "themes": re.compile(r"\.json$"),
}


def collect_resource_files(directory: str, resource_type: ResourceType) -> list[str]:
    if resource_type == "skills":
        return collect_skill_entries(directory, "pi")
    if resource_type == "extensions":
        return collect_auto_extension_entries(directory)
    return collect_files(directory, _FILE_SUFFIX_PATTERNS[resource_type])


def _resource_precedence_rank(m: PathMetadata) -> int:
    """Lower rank = higher precedence for name-collision resolution ("first wins")."""
    if m.origin == "package":
        return 4
    scope_base = 0 if m.scope == "project" else 2
    return scope_base + (0 if m.source == "local" else 1)


# --------------------------------------------------------------------------
# Package manager
# --------------------------------------------------------------------------

_ResourceMapEntry = tuple[PathMetadata, bool]
_ResourceMap = dict[str, _ResourceMapEntry]


@dataclass
class _ResourceAccumulator:
    extensions: _ResourceMap = field(default_factory=dict)
    skills: _ResourceMap = field(default_factory=dict)
    prompts: _ResourceMap = field(default_factory=dict)
    themes: _ResourceMap = field(default_factory=dict)


class PackageManager:
    """Installs, updates, removes, lists, and resolves resource packages.

    Port of `DefaultPackageManager`. See the module docstring for the npm,
    dependency-install, and manifest-format scope deviations.
    """

    def __init__(
        self,
        cwd: str,
        agent_dir: str,
        settings_manager: SettingsManager,
        *,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._cwd = resolve_path(cwd)
        self._agent_dir = resolve_path(agent_dir)
        self._settings_manager = settings_manager
        self._runner = command_runner or SubprocessCommandRunner()
        self._progress_callback: ProgressCallback | None = None

    # -- progress -----------------------------------------------------------

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self._progress_callback = callback

    def _emit_progress(self, event: ProgressEvent) -> None:
        if self._progress_callback:
            self._progress_callback(event)

    async def _with_progress(
        self,
        action: Literal["install", "remove", "update", "clone", "pull"],
        source: str,
        message: str,
        operation: Callable[[], Any],
    ) -> None:
        self._emit_progress(ProgressEvent("start", action, source, message))
        try:
            result = operation()
            if hasattr(result, "__await__"):
                await result
            self._emit_progress(ProgressEvent("complete", action, source))
        except Exception as error:
            self._emit_progress(ProgressEvent("error", action, source, str(error)))
            raise

    # -- settings persistence ------------------------------------------------

    def add_source_to_settings(self, source: str, *, local: bool = False) -> bool:
        scope: SourceScope = "project" if local else "user"
        current_settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        current_packages: list[PackageSourceEntry] = list(current_settings.get("packages") or [])
        normalized_source = self._normalize_package_source_for_settings(source, scope)
        match_index = next(
            (i for i, existing in enumerate(current_packages) if self._package_sources_match(existing, source, scope)),
            None,
        )
        if match_index is not None:
            existing = current_packages[match_index]
            if _package_source_string(existing) == normalized_source:
                return False
            next_packages = list(current_packages)
            if isinstance(existing, str):
                next_packages[match_index] = normalized_source
            else:
                next_packages[match_index] = {**existing, "source": normalized_source}
            self._set_packages_for_scope(scope, next_packages)
            return True

        next_packages = [*current_packages, normalized_source]
        self._set_packages_for_scope(scope, next_packages)
        return True

    def remove_source_from_settings(self, source: str, *, local: bool = False) -> bool:
        scope: SourceScope = "project" if local else "user"
        current_settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        current_packages: list[PackageSourceEntry] = list(current_settings.get("packages") or [])
        next_packages = [pkg for pkg in current_packages if not self._package_sources_match(pkg, source, scope)]
        if len(next_packages) == len(current_packages):
            return False
        self._set_packages_for_scope(scope, next_packages)
        return True

    def _set_packages_for_scope(self, scope: SourceScope, packages: list[PackageSourceEntry]) -> None:
        if scope == "project":
            self._settings_manager.set_project_packages(packages)
        else:
            self._settings_manager.set_packages(packages)

    def get_installed_path(self, source: str, scope: Literal["user", "project"]) -> str | None:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            path = self._get_git_install_path(parsed, scope)
            return path if os.path.exists(path) else None
        base_dir = self._get_base_dir_for_scope(scope)
        path = self._resolve_path_from_base(parsed.path, base_dir)
        return path if os.path.exists(path) else None

    def list_configured_packages(self) -> list[ConfiguredPackage]:
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        configured: list[ConfiguredPackage] = []

        for pkg in global_settings.get("packages") or []:
            source = _package_source_string(pkg)
            configured.append(
                ConfiguredPackage(
                    source=source,
                    scope="user",
                    filtered=isinstance(pkg, dict),
                    installed_path=self.get_installed_path(source, "user"),
                )
            )
        for pkg in project_settings.get("packages") or []:
            source = _package_source_string(pkg)
            configured.append(
                ConfiguredPackage(
                    source=source,
                    scope="project",
                    filtered=isinstance(pkg, dict),
                    installed_path=self.get_installed_path(source, "project"),
                )
            )
        return configured

    # -- install / remove / update -------------------------------------------

    async def install(self, source: str, *, local: bool = False) -> None:
        parsed = self.parse_source(source)
        scope: SourceScope = "project" if local else "user"
        self._assert_project_trusted_for_scope(scope)

        async def do_install() -> None:
            if isinstance(parsed, GitSource):
                await self._install_git(parsed, scope)
                return
            resolved = self._resolve_path(parsed.path)
            if not os.path.exists(resolved):
                raise ValueError(f"Path does not exist: {resolved}")

        await self._with_progress("install", source, f"Installing {source}...", do_install)

    async def install_and_persist(self, source: str, *, local: bool = False) -> None:
        await self.install(source, local=local)
        self.add_source_to_settings(source, local=local)

    async def remove(self, source: str, *, local: bool = False) -> None:
        parsed = self.parse_source(source)
        scope: SourceScope = "project" if local else "user"
        self._assert_project_trusted_for_scope(scope)

        async def do_remove() -> None:
            if isinstance(parsed, GitSource):
                await self._remove_git(parsed, scope)

        await self._with_progress("remove", source, f"Removing {source}...", do_remove)

    async def remove_and_persist(self, source: str, *, local: bool = False) -> bool:
        await self.remove(source, local=local)
        return self.remove_source_from_settings(source, local=local)

    async def update(self, source: str | None = None) -> None:
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        identity = self._get_package_identity(source) if source else None
        matched = False
        update_sources: list[tuple[str, Literal["user", "project"]]] = []

        for pkg in global_settings.get("packages") or []:
            source_str = _package_source_string(pkg)
            if identity and self._get_package_identity(source_str, "user") != identity:
                continue
            matched = True
            update_sources.append((source_str, "user"))
        for pkg in project_settings.get("packages") or []:
            source_str = _package_source_string(pkg)
            if identity and self._get_package_identity(source_str, "project") != identity:
                continue
            matched = True
            update_sources.append((source_str, "project"))

        if source and not matched:
            all_packages = list(global_settings.get("packages") or []) + list(project_settings.get("packages") or [])
            raise ValueError(self._build_no_matching_package_message(source, all_packages))

        # Matches the TypeScript's `updateConfiguredSources` guard: offline mode
        # disables every network update.
        if _is_offline_mode_enabled() or not update_sources:
            return

        git_tasks: list[Callable[[], Awaitable[None]]] = []
        for source_str, scope in update_sources:
            parsed = self.parse_source(source_str)
            if not isinstance(parsed, GitSource):
                continue
            git_tasks.append(
                lambda p=parsed, s=scope, src=source_str: self._with_progress(
                    "update", src, f"Updating {src}...", lambda p=p, s=s: self._update_git(p, s)
                )
            )
        await _run_with_concurrency(git_tasks, GIT_UPDATE_CONCURRENCY)

    # -- resolve --------------------------------------------------------------

    async def resolve(self, on_missing: Callable[[str], Any] | None = None) -> ResolvedPaths:
        accumulator = _ResourceAccumulator()
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()

        all_packages: list[tuple[PackageSourceEntry, SourceScope]] = []
        for pkg in project_settings.get("packages") or []:
            all_packages.append((pkg, "project"))
        for pkg in global_settings.get("packages") or []:
            all_packages.append((pkg, "user"))

        package_sources = self._dedupe_packages(all_packages)
        await self._resolve_package_sources(package_sources, accumulator, on_missing)

        global_base_dir = self._agent_dir
        project_base_dir = os.path.join(self._cwd, CONFIG_DIR_NAME)

        for resource_type in RESOURCE_TYPES:
            target = self._get_target_map(accumulator, resource_type)
            global_entries = list(global_settings.get(resource_type) or [])
            project_entries = list(project_settings.get(resource_type) or [])
            self._resolve_local_entries(
                project_entries,
                resource_type,
                target,
                PathMetadata(source="local", scope="project", origin="top-level"),
                project_base_dir,
            )
            self._resolve_local_entries(
                global_entries,
                resource_type,
                target,
                PathMetadata(source="local", scope="user", origin="top-level"),
                global_base_dir,
            )

        self._add_auto_discovered_resources(
            accumulator, global_settings, project_settings, global_base_dir, project_base_dir
        )

        return self._to_resolved_paths(accumulator)

    async def resolve_extension_sources(
        self, sources: list[str], *, local: bool = False, temporary: bool = False
    ) -> ResolvedPaths:
        accumulator = _ResourceAccumulator()
        scope: SourceScope = "temporary" if temporary else ("project" if local else "user")
        package_sources: list[tuple[PackageSourceEntry, SourceScope]] = [(source, scope) for source in sources]
        await self._resolve_package_sources(package_sources, accumulator, None)
        return self._to_resolved_paths(accumulator)

    # -- source parsing ---------------------------------------------------

    def parse_source(self, source: str) -> ParsedSource:
        if source.startswith("npm:"):
            raise ValueError(
                "npm package sources are not supported by this Python port (no JavaScript package "
                "manager equivalent); use a git or local path source instead."
            )
        if self._is_local_path(source):
            return LocalSource(path=source)
        git_parsed = parse_git_url(source)
        if git_parsed:
            return git_parsed
        return LocalSource(path=source)

    @staticmethod
    def _is_local_path(value: str) -> bool:
        trimmed = value.strip()
        return not trimmed.startswith(("npm:", "git:", "github:", "http:", "https:", "ssh:"))

    def _get_package_identity(self, source: str, scope: SourceScope | None = None) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        if scope:
            base_dir = self._get_base_dir_for_scope(scope)
            return f"local:{self._resolve_path_from_base(parsed.path, base_dir)}"
        return f"local:{self._resolve_path(parsed.path)}"

    def _package_sources_match(self, existing: PackageSourceEntry, input_source: str, scope: SourceScope) -> bool:
        left = self._get_source_match_key_for_settings(_package_source_string(existing), scope)
        right = self._get_source_match_key_for_input(input_source)
        return left == right

    def _get_source_match_key_for_input(self, source: str) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        return f"local:{self._resolve_path(parsed.path)}"

    def _get_source_match_key_for_settings(self, source: str, scope: SourceScope) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        base_dir = self._get_base_dir_for_scope(scope)
        return f"local:{self._resolve_path_from_base(parsed.path, base_dir)}"

    def _normalize_package_source_for_settings(self, source: str, scope: SourceScope) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return source
        base_dir = self._get_base_dir_for_scope(scope)
        resolved = self._resolve_path(parsed.path)
        rel = os.path.relpath(resolved, base_dir)
        return rel or "."

    def _build_no_matching_package_message(self, source: str, configured_packages: list[PackageSourceEntry]) -> str:
        suggestion = self._find_suggested_configured_source(source, configured_packages)
        if not suggestion:
            return f"No matching package found for {source}"
        return f"No matching package found for {source}. Did you mean {suggestion}?"

    def _find_suggested_configured_source(
        self, source: str, configured_packages: list[PackageSourceEntry]
    ) -> str | None:
        trimmed_source = source.strip()
        for pkg in configured_packages:
            source_str = _package_source_string(pkg)
            parsed = self.parse_source(source_str)
            if isinstance(parsed, GitSource):
                shorthand = f"{parsed.host}/{parsed.path}"
                shorthand_with_ref = f"{shorthand}@{parsed.ref}" if parsed.ref else None
                if trimmed_source == shorthand or (shorthand_with_ref and trimmed_source == shorthand_with_ref):
                    return source_str
        return None

    def _get_package_identity_for_entry(self, pkg: PackageSourceEntry, scope: SourceScope) -> str:
        return self._get_package_identity(_package_source_string(pkg), scope)

    def _dedupe_packages(
        self, packages: list[tuple[PackageSourceEntry, SourceScope]]
    ) -> list[tuple[PackageSourceEntry, SourceScope]]:
        """Dedupe by identity; project wins over user unless it's an autoload=false delta."""
        result: list[tuple[PackageSourceEntry, SourceScope]] = []
        seen: dict[str, int] = {}
        for pkg, scope in packages:
            identity = self._get_package_identity_for_entry(pkg, scope)
            index = seen.get(identity)
            if index is None:
                seen[identity] = len(result)
                result.append((pkg, scope))
                continue
            existing_pkg, existing_scope = result[index]
            if existing_scope == "project" and scope == "user":
                existing_filter = _package_filter(existing_pkg)
                if existing_filter is not None and existing_filter.autoload is False:
                    result.append((pkg, scope))
            elif scope == "project":
                result[index] = (pkg, scope)
        return result

    def _find_autoload_delta_base(
        self,
        pkg: PackageSourceEntry,
        scope: SourceScope,
        sources: list[tuple[PackageSourceEntry, SourceScope]],
    ) -> tuple[str, SourceScope] | None:
        if scope != "project":
            return None
        pkg_filter = _package_filter(pkg)
        if pkg_filter is None or pkg_filter.autoload is not False:
            return None
        identity = self._get_package_identity(_package_source_string(pkg), scope)
        for entry_pkg, entry_scope in sources:
            if entry_scope != "user":
                continue
            if self._get_package_identity(_package_source_string(entry_pkg), "user") == identity:
                return _package_source_string(entry_pkg), "user"
        return None

    # -- resource-source resolution ----------------------------------------

    async def _resolve_package_sources(
        self,
        sources: list[tuple[PackageSourceEntry, SourceScope]],
        accumulator: _ResourceAccumulator,
        on_missing: Callable[[str], Any] | None,
    ) -> None:
        for pkg, scope in sources:
            source_str = _package_source_string(pkg)
            pkg_filter = _package_filter(pkg)
            delta_base = self._find_autoload_delta_base(pkg, scope, sources)
            resolved_source = delta_base[0] if delta_base else source_str
            resolved_scope = delta_base[1] if delta_base else scope
            parsed = self.parse_source(resolved_source)
            metadata = PathMetadata(source=source_str, scope=scope, origin="package")

            if not isinstance(parsed, GitSource):
                base_dir = self._get_base_dir_for_scope(resolved_scope)
                self._resolve_local_extension_source(parsed, accumulator, pkg_filter, metadata, base_dir)
                continue

            installed_path = self._get_git_install_path(parsed, resolved_scope)
            if not os.path.exists(installed_path):
                installed = await self._install_missing(parsed, resolved_scope, resolved_source, on_missing)
                if not installed:
                    continue
            elif resolved_scope == "temporary" and not parsed.pinned and not _is_offline_mode_enabled():
                await self._refresh_temporary_git_source(parsed, resolved_source)
            metadata.base_dir = installed_path
            self._collect_package_resources(installed_path, accumulator, pkg_filter, metadata)

    async def _install_missing(
        self,
        parsed: GitSource,
        scope: SourceScope,
        resolved_source: str,
        on_missing: Callable[[str], Any] | None,
    ) -> bool:
        if _is_offline_mode_enabled():
            return False
        if not on_missing:
            await self._install_git(parsed, scope)
            return True
        action = on_missing(resolved_source)
        if hasattr(action, "__await__"):
            action = await action
        if action == "skip":
            return False
        if action == "error":
            raise ValueError(f"Missing source: {resolved_source}")
        await self._install_git(parsed, scope)
        return True

    def _resolve_local_extension_source(
        self,
        source: LocalSource,
        accumulator: _ResourceAccumulator,
        pkg_filter: PackageFilter | None,
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        resolved = self._resolve_path_from_base(source.path, base_dir)
        if not os.path.exists(resolved):
            return
        if os.path.isfile(resolved):
            metadata.base_dir = os.path.dirname(resolved)
            self._add_resource(accumulator.extensions, resolved, metadata, True)
            return
        if os.path.isdir(resolved):
            metadata.base_dir = resolved
            handled = self._collect_package_resources(resolved, accumulator, pkg_filter, metadata)
            if not handled:
                self._add_resource(accumulator.extensions, resolved, metadata, True)

    # -- resource collection within a package root --------------------------

    def _collect_package_resources(
        self,
        package_root: str,
        accumulator: _ResourceAccumulator,
        pkg_filter: PackageFilter | None,
        metadata: PathMetadata,
    ) -> bool:
        if pkg_filter is not None:
            for resource_type in RESOURCE_TYPES:
                patterns = pkg_filter.get(resource_type)
                target = self._get_target_map(accumulator, resource_type)
                if pkg_filter.autoload is False:
                    self._apply_package_delta_filter(package_root, patterns or [], resource_type, target, metadata)
                elif patterns is not None:
                    self._apply_package_filter(package_root, patterns, resource_type, target, metadata)
                else:
                    self._collect_default_resources(package_root, resource_type, target, metadata)
            return True

        manifest = read_pi_manifest(os.path.join(package_root, "pi.json"))
        if manifest:
            for resource_type in RESOURCE_TYPES:
                entries = manifest.get(resource_type)
                self._add_manifest_entries(
                    entries, package_root, resource_type, self._get_target_map(accumulator, resource_type), metadata
                )
            return True

        has_any_dir = False
        for resource_type in RESOURCE_TYPES:
            directory = os.path.join(package_root, resource_type)
            if os.path.exists(directory):
                for f in collect_resource_files(directory, resource_type):
                    self._add_resource(self._get_target_map(accumulator, resource_type), f, metadata, True)
                has_any_dir = True
        return has_any_dir

    def _collect_default_resources(
        self, package_root: str, resource_type: ResourceType, target: _ResourceMap, metadata: PathMetadata
    ) -> None:
        manifest = read_pi_manifest(os.path.join(package_root, "pi.json"))
        entries = manifest.get(resource_type) if manifest else None
        if entries is not None:
            self._add_manifest_entries(entries, package_root, resource_type, target, metadata)
            return
        directory = os.path.join(package_root, resource_type)
        if os.path.exists(directory):
            for f in collect_resource_files(directory, resource_type):
                self._add_resource(target, f, metadata, True)

    def _apply_package_filter(
        self,
        package_root: str,
        user_patterns: list[str],
        resource_type: ResourceType,
        target: _ResourceMap,
        metadata: PathMetadata,
    ) -> None:
        all_files, _enabled_by_manifest = self._collect_manifest_files(package_root, resource_type)

        if not user_patterns:
            for f in all_files:
                self._add_resource(target, f, metadata, False)
            return

        enabled_by_user = apply_patterns(all_files, user_patterns, package_root)
        for f in all_files:
            self._add_resource(target, f, metadata, f in enabled_by_user)

    def _apply_package_delta_filter(
        self,
        package_root: str,
        user_patterns: list[str],
        resource_type: ResourceType,
        target: _ResourceMap,
        metadata: PathMetadata,
    ) -> None:
        if not user_patterns:
            return
        all_files, _enabled_by_manifest = self._collect_manifest_files(package_root, resource_type)
        enabled_by_user = apply_autoload_disabled_patterns(all_files, user_patterns, package_root)
        for file_path, enabled in enabled_by_user.items():
            self._add_resource(target, file_path, metadata, enabled)

    def _collect_manifest_files(self, package_root: str, resource_type: ResourceType) -> tuple[list[str], set[str]]:
        manifest = read_pi_manifest(os.path.join(package_root, "pi.json"))
        entries = manifest.get(resource_type) if manifest else None
        if entries:
            all_files = self._collect_files_from_manifest_entries(entries, package_root, resource_type)
            manifest_patterns = [e for e in entries if is_override_pattern(e)]
            enabled_by_manifest = (
                apply_patterns(all_files, manifest_patterns, package_root) if manifest_patterns else set(all_files)
            )
            return list(enabled_by_manifest), enabled_by_manifest

        convention_dir = os.path.join(package_root, resource_type)
        if not os.path.exists(convention_dir):
            return [], set()
        all_files = collect_resource_files(convention_dir, resource_type)
        return all_files, set(all_files)

    def _add_manifest_entries(
        self,
        entries: list[str] | None,
        root: str,
        resource_type: ResourceType,
        target: _ResourceMap,
        metadata: PathMetadata,
    ) -> None:
        if entries is None:
            return
        all_files = self._collect_files_from_manifest_entries(entries, root, resource_type)
        patterns = [e for e in entries if is_override_pattern(e)]
        enabled_paths = apply_patterns(all_files, patterns, root)
        for f in all_files:
            if f in enabled_paths:
                self._add_resource(target, f, metadata, True)

    def _collect_files_from_manifest_entries(
        self, entries: list[str], root: str, resource_type: ResourceType
    ) -> list[str]:
        import glob as glob_module

        source_entries = [e for e in entries if not is_override_pattern(e)]
        resolved: list[str] = []
        for entry in source_entries:
            if not has_glob_pattern(entry):
                resolved.append(os.path.abspath(os.path.join(root, entry)))
                continue
            matches = glob_module.glob(os.path.join(root, entry), recursive=True)
            resolved.extend(os.path.abspath(m) for m in matches)
        return self._collect_files_from_paths(resolved, resource_type)

    def _resolve_local_entries(
        self,
        entries: list[str],
        resource_type: ResourceType,
        target: _ResourceMap,
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        if not entries:
            return
        plain, patterns = split_patterns(entries)
        resolved_plain = [self._resolve_path_from_base(p, base_dir) for p in plain]
        all_files = self._collect_files_from_paths(resolved_plain, resource_type)
        enabled_paths = apply_patterns(all_files, patterns, base_dir)
        for f in all_files:
            self._add_resource(target, f, metadata, f in enabled_paths)

    def _add_auto_discovered_resources(
        self,
        accumulator: _ResourceAccumulator,
        global_settings: dict[str, Any],
        project_settings: dict[str, Any],
        global_base_dir: str,
        project_base_dir: str,
    ) -> None:
        user_metadata = PathMetadata(source="auto", scope="user", origin="top-level", base_dir=global_base_dir)
        project_metadata = PathMetadata(source="auto", scope="project", origin="top-level", base_dir=project_base_dir)

        user_overrides = {rt: list(global_settings.get(rt) or []) for rt in RESOURCE_TYPES}
        project_overrides = {rt: list(project_settings.get(rt) or []) for rt in RESOURCE_TYPES}

        user_dirs = {rt: os.path.join(global_base_dir, rt) for rt in RESOURCE_TYPES}
        project_dirs = {rt: os.path.join(project_base_dir, rt) for rt in RESOURCE_TYPES}

        user_agents_skills_dir = os.path.join(_get_home_dir(), ".agents", "skills")
        project_trusted = self._settings_manager.is_project_trusted()
        project_agents_skill_dirs = (
            [
                d
                for d in collect_ancestor_agents_skill_dirs(self._cwd)
                if os.path.abspath(d) != os.path.abspath(user_agents_skills_dir)
            ]
            if project_trusted
            else []
        )

        def add_resources(
            resource_type: ResourceType, paths: list[str], metadata: PathMetadata, overrides: list[str], base_dir: str
        ) -> None:
            target = self._get_target_map(accumulator, resource_type)
            for path in paths:
                enabled = is_enabled_by_overrides(path, overrides, base_dir)
                self._add_resource(target, path, metadata, enabled)

        if project_trusted:
            add_resources(
                "extensions",
                collect_auto_extension_entries(project_dirs["extensions"]),
                project_metadata,
                project_overrides["extensions"],
                project_base_dir,
            )
            add_resources(
                "skills",
                collect_auto_skill_entries(project_dirs["skills"], "pi"),
                project_metadata,
                project_overrides["skills"],
                project_base_dir,
            )

        for agents_skills_dir in project_agents_skill_dirs:
            agents_base_dir = os.path.dirname(agents_skills_dir)
            agents_metadata = PathMetadata(
                source=project_metadata.source,
                scope=project_metadata.scope,
                origin=project_metadata.origin,
                base_dir=agents_base_dir,
            )
            add_resources(
                "skills",
                collect_auto_skill_entries(agents_skills_dir, "agents"),
                agents_metadata,
                project_overrides["skills"],
                agents_base_dir,
            )

        if project_trusted:
            add_resources(
                "prompts",
                collect_auto_prompt_entries(project_dirs["prompts"]),
                project_metadata,
                project_overrides["prompts"],
                project_base_dir,
            )
            add_resources(
                "themes",
                collect_auto_theme_entries(project_dirs["themes"]),
                project_metadata,
                project_overrides["themes"],
                project_base_dir,
            )

        add_resources(
            "extensions",
            collect_auto_extension_entries(user_dirs["extensions"]),
            user_metadata,
            user_overrides["extensions"],
            global_base_dir,
        )
        add_resources(
            "skills",
            collect_auto_skill_entries(user_dirs["skills"], "pi"),
            user_metadata,
            user_overrides["skills"],
            global_base_dir,
        )

        user_agents_base_dir = os.path.dirname(user_agents_skills_dir)
        user_agents_metadata = PathMetadata(
            source=user_metadata.source,
            scope=user_metadata.scope,
            origin=user_metadata.origin,
            base_dir=user_agents_base_dir,
        )
        add_resources(
            "skills",
            collect_auto_skill_entries(user_agents_skills_dir, "agents"),
            user_agents_metadata,
            user_overrides["skills"],
            user_agents_base_dir,
        )

        add_resources(
            "prompts",
            collect_auto_prompt_entries(user_dirs["prompts"]),
            user_metadata,
            user_overrides["prompts"],
            global_base_dir,
        )
        add_resources(
            "themes",
            collect_auto_theme_entries(user_dirs["themes"]),
            user_metadata,
            user_overrides["themes"],
            global_base_dir,
        )

    def _collect_files_from_paths(self, paths: list[str], resource_type: ResourceType) -> list[str]:
        files: list[str] = []
        for p in paths:
            if not os.path.exists(p):
                continue
            if os.path.isfile(p):
                files.append(p)
            elif os.path.isdir(p):
                files.extend(collect_resource_files(p, resource_type))
        return files

    def _get_target_map(self, accumulator: _ResourceAccumulator, resource_type: ResourceType) -> _ResourceMap:
        return getattr(accumulator, resource_type)

    def _add_resource(self, target: _ResourceMap, path: str, metadata: PathMetadata, enabled: bool) -> None:
        if not path:
            return
        if path not in target:
            target[path] = (metadata, enabled)

    def _to_resolved_paths(self, accumulator: _ResourceAccumulator) -> ResolvedPaths:
        def map_to_resolved(entries: _ResourceMap) -> list[ResolvedResource]:
            resolved = [
                ResolvedResource(path=path, enabled=enabled, metadata=meta) for path, (meta, enabled) in entries.items()
            ]
            resolved.sort(key=lambda r: _resource_precedence_rank(r.metadata))
            seen: set[str] = set()
            deduped: list[ResolvedResource] = []
            for entry in resolved:
                canonical = _canonicalize_path(entry.path)
                if canonical in seen:
                    continue
                seen.add(canonical)
                deduped.append(entry)
            return deduped

        return ResolvedPaths(
            extensions=map_to_resolved(accumulator.extensions),
            skills=map_to_resolved(accumulator.skills),
            prompts=map_to_resolved(accumulator.prompts),
            themes=map_to_resolved(accumulator.themes),
        )

    # -- path/scope helpers ---------------------------------------------------

    def _assert_project_trusted_for_scope(self, scope: SourceScope) -> None:
        if scope == "project" and not self._settings_manager.is_project_trusted():
            raise ValueError("Project is not trusted; refusing to access project package storage")

    def _get_base_dir_for_scope(self, scope: SourceScope) -> str:
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return os.path.join(self._cwd, CONFIG_DIR_NAME)
        if scope == "user":
            return self._agent_dir
        return self._cwd

    def _resolve_path(self, input_path: str) -> str:
        return resolve_path(input_path, self._cwd)

    def _resolve_path_from_base(self, input_path: str, base_dir: str) -> str:
        return resolve_path(input_path, base_dir)

    def _resolve_managed_path(self, root: str, *parts: str) -> str:
        resolved_root = os.path.abspath(root)
        resolved_path = os.path.abspath(os.path.join(resolved_root, *parts))
        if resolved_path != resolved_root and not resolved_path.startswith(resolved_root + os.sep):
            raise ValueError(f"Refusing to use path outside package install root: {resolved_path}")
        return resolved_path

    def _get_temporary_dir(self, prefix: str, suffix: str = "") -> str:
        root = self._resolve_managed_path(get_extension_temp_folder(self._agent_dir), prefix)
        digest = hashlib.sha256(f"{prefix}-{suffix}".encode()).hexdigest()[:8]
        return self._resolve_managed_path(root, digest, suffix)

    def _get_git_install_root(self, scope: SourceScope) -> str | None:
        if scope == "temporary":
            return None
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return os.path.join(self._cwd, CONFIG_DIR_NAME, "git")
        return os.path.join(self._agent_dir, "git")

    def _get_git_install_path(self, source: GitSource, scope: SourceScope) -> str:
        if scope == "temporary":
            return self._get_temporary_dir(f"git-{source.host}", source.path)
        install_root = self._get_git_install_root(scope)
        if not install_root:
            raise ValueError("Missing git install root")
        return self._resolve_managed_path(install_root, source.host, source.path)

    def _get_git_update_marker_path(self, target_dir: str) -> str:
        return os.path.join(os.path.dirname(target_dir), f".{os.path.basename(target_dir)}.pi-update-incomplete")

    def _prune_empty_git_parents(self, target_dir: str, install_root: str | None) -> None:
        if not install_root:
            return
        resolved_root = os.path.abspath(install_root)
        current = os.path.dirname(target_dir)
        while current.startswith(resolved_root) and current != resolved_root:
            if not os.path.exists(current):
                current = os.path.dirname(current)
                continue
            if os.listdir(current):
                break
            try:
                _rmtree(current)
            except OSError:
                break
            current = os.path.dirname(current)

    def _ensure_git_ignore(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        ignore_path = os.path.join(directory, ".gitignore")
        if not os.path.exists(ignore_path):
            with open(ignore_path, "w", encoding="utf-8") as fh:
                fh.write("*\n!.gitignore\n")

    # -- git operations -------------------------------------------------------

    async def _install_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if os.path.exists(target_dir):
            if source.ref:
                await self._ensure_git_ref(target_dir, ["fetch", "origin", source.ref], "FETCH_HEAD")
                return
            target = await self._get_local_git_update_target(target_dir)
            await self._ensure_git_ref(target_dir, target["fetch_args"], target["ref"])
            return

        git_root = self._get_git_install_root(scope)
        if git_root:
            self._ensure_git_ignore(git_root)
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        _remove_if_exists(self._get_git_update_marker_path(target_dir))

        try:
            await self._runner.run("git", ["clone", source.repo, target_dir])
            if source.ref:
                await self._runner.run("git", ["checkout", source.ref], cwd=target_dir)
        except Exception:
            _rmtree(target_dir)
            self._prune_empty_git_parents(target_dir, git_root)
            raise

    async def _update_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if not os.path.exists(target_dir):
            await self._install_git(source, scope)
            return
        if source.ref:
            await self._ensure_git_ref(target_dir, ["fetch", "origin", source.ref], "FETCH_HEAD")
            return
        target = await self._get_local_git_update_target(target_dir)
        await self._ensure_git_ref(target_dir, target["fetch_args"], target["ref"])

    async def _remove_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        _rmtree(target_dir)
        _remove_if_exists(self._get_git_update_marker_path(target_dir))
        self._prune_empty_git_parents(target_dir, self._get_git_install_root(scope))

    async def _ensure_git_ref(self, target_dir: str, fetch_args: list[str], ref: str) -> None:
        await self._runner.run("git", fetch_args, cwd=target_dir)

        local_head = await self._runner.run_capture(
            "git", ["rev-parse", "HEAD"], cwd=target_dir, timeout=NETWORK_TIMEOUT_S
        )
        commit_ref = f"{ref}^{{commit}}"
        target_head = await self._runner.run_capture(
            "git", ["rev-parse", commit_ref], cwd=target_dir, timeout=NETWORK_TIMEOUT_S
        )
        marker_path = self._get_git_update_marker_path(target_dir)
        if local_head.strip() == target_head.strip():
            if os.path.exists(marker_path):
                await self._runner.run("git", ["clean", "-fdx"], cwd=target_dir)
                _remove_if_exists(marker_path)
            return

        with open(marker_path, "w", encoding="utf-8"):
            pass
        await self._runner.run("git", ["reset", "--hard", commit_ref], cwd=target_dir)
        await self._runner.run("git", ["clean", "-fdx"], cwd=target_dir)
        _remove_if_exists(marker_path)

    async def _get_local_git_update_target(self, installed_path: str) -> dict[str, Any]:
        try:
            upstream = await self._runner.run_capture(
                "git", ["rev-parse", "--abbrev-ref", "@{upstream}"], cwd=installed_path, timeout=NETWORK_TIMEOUT_S
            )
            trimmed_upstream = upstream.strip()
            if not trimmed_upstream.startswith("origin/"):
                raise ValueError(f"Unsupported upstream remote: {trimmed_upstream}")
            branch = trimmed_upstream[len("origin/") :]
            if not branch:
                raise ValueError("Missing upstream branch name")
            return {
                "ref": "@{upstream}",
                "fetch_args": [
                    "fetch",
                    "--prune",
                    "--no-tags",
                    "origin",
                    f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                ],
            }
        except Exception:
            with contextlib.suppress(Exception):
                await self._runner.run("git", ["remote", "set-head", "origin", "-a"], cwd=installed_path)
            try:
                origin_head_ref = await self._runner.run_capture(
                    "git", ["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=installed_path, timeout=NETWORK_TIMEOUT_S
                )
            except Exception:
                origin_head_ref = ""
            branch = re.sub(r"^refs/remotes/origin/", "", origin_head_ref.strip())
            if branch:
                return {
                    "ref": "origin/HEAD",
                    "fetch_args": [
                        "fetch",
                        "--prune",
                        "--no-tags",
                        "origin",
                        f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                    ],
                }
            return {
                "ref": "origin/HEAD",
                "fetch_args": ["fetch", "--prune", "--no-tags", "origin", "+HEAD:refs/remotes/origin/HEAD"],
            }

    async def _refresh_temporary_git_source(self, source: GitSource, source_str: str) -> None:
        if _is_offline_mode_enabled():
            return
        with contextlib.suppress(Exception):
            await self._with_progress(
                "pull", source_str, f"Refreshing {source_str}...", lambda: self._update_git(source, "temporary")
            )

    async def check_for_available_updates(self) -> list[PackageUpdate]:
        """Report configured git packages whose local HEAD differs from `origin`'s.

        Port of `checkForAvailableUpdates()` narrowed to git sources (the
        npm-registry-version-lookup half has no Python equivalent; see the
        module docstring).
        """
        if _is_offline_mode_enabled():
            return []
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        all_packages: list[tuple[PackageSourceEntry, SourceScope]] = []
        for pkg in project_settings.get("packages") or []:
            all_packages.append((pkg, "project"))
        for pkg in global_settings.get("packages") or []:
            all_packages.append((pkg, "user"))

        package_sources = self._dedupe_packages(all_packages)
        checks: list[Callable[[], Awaitable[PackageUpdate | None]]] = []
        for pkg, scope in package_sources:
            if scope == "temporary":
                continue
            source = _package_source_string(pkg)
            parsed = self.parse_source(source)
            if not isinstance(parsed, GitSource) or parsed.pinned:
                continue
            installed_path = self._get_git_install_path(parsed, scope)
            if not os.path.exists(installed_path):
                continue

            async def check(
                path: str = installed_path, src: str = source, p: GitSource = parsed, s: SourceScope = scope
            ) -> PackageUpdate | None:
                if not await self._git_has_available_update(path):
                    return None
                return PackageUpdate(source=src, display_name=f"{p.host}/{p.path}", type="git", scope=s)

            checks.append(check)

        results = await _run_with_concurrency(checks, UPDATE_CHECK_CONCURRENCY)
        return [result for result in results if result is not None]

    async def _git_has_available_update(self, installed_path: str) -> bool:
        if _is_offline_mode_enabled():
            return False
        try:
            local_head = await self._runner.run_capture(
                "git", ["rev-parse", "HEAD"], cwd=installed_path, timeout=NETWORK_TIMEOUT_S
            )
            remote_head = await self._get_remote_git_head(installed_path)
            return local_head.strip() != remote_head.strip()
        except Exception:
            return False

    async def _get_remote_git_head(self, installed_path: str) -> str:
        upstream_ref = await self._get_git_upstream_ref(installed_path)
        if upstream_ref:
            remote_head = await self._run_git_remote_command(installed_path, ["ls-remote", "origin", upstream_ref])
            match = re.search(r"^([0-9a-f]{40})\s+", remote_head, re.MULTILINE)
            if match:
                return match.group(1)

        remote_head = await self._run_git_remote_command(installed_path, ["ls-remote", "origin", "HEAD"])
        match = re.search(r"^([0-9a-f]{40})\s+HEAD$", remote_head, re.MULTILINE)
        if not match:
            raise ValueError("Failed to determine remote HEAD")
        return match.group(1)

    async def _get_git_upstream_ref(self, installed_path: str) -> str | None:
        try:
            upstream = await self._runner.run_capture(
                "git", ["rev-parse", "--abbrev-ref", "@{upstream}"], cwd=installed_path, timeout=NETWORK_TIMEOUT_S
            )
            trimmed = upstream.strip()
            if not trimmed.startswith("origin/"):
                return None
            branch = trimmed[len("origin/") :]
            return f"refs/heads/{branch}" if branch else None
        except Exception:
            return None

    async def _run_git_remote_command(self, installed_path: str, args: list[str]) -> str:
        return await self._runner.run_capture(
            "git", args, cwd=installed_path, timeout=NETWORK_TIMEOUT_S, env={"GIT_TERMINAL_PROMPT": "0"}
        )


def _remove_if_exists(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "RESOURCE_TYPES",
    "CommandRunner",
    "ConfiguredPackage",
    "GitSource",
    "LocalSource",
    "MissingSourceAction",
    "PackageFilter",
    "PackageManager",
    "PackageUpdate",
    "ParsedSource",
    "PathMetadata",
    "ProgressCallback",
    "ProgressEvent",
    "ResolvedPaths",
    "ResolvedResource",
    "ResourceType",
    "SourceScope",
    "SubprocessCommandRunner",
    "apply_autoload_disabled_patterns",
    "apply_patterns",
    "collect_ancestor_agents_skill_dirs",
    "collect_auto_extension_entries",
    "collect_auto_prompt_entries",
    "collect_auto_skill_entries",
    "collect_auto_theme_entries",
    "collect_files",
    "collect_resource_files",
    "collect_skill_entries",
    "find_git_repo_root",
    "get_extension_temp_folder",
    "get_override_patterns",
    "has_glob_pattern",
    "is_enabled_by_overrides",
    "is_override_pattern",
    "is_pattern",
    "matches_any_exact_pattern",
    "matches_any_pattern",
    "split_patterns",
]
