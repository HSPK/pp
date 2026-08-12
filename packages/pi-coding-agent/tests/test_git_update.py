"""Python port of `packages/coding-agent/test/git-update.test.ts`.

Every case drives a real local git repository through `PackageManager`; no
network access is involved (the "remote" is a directory on disk, exactly as in
the TypeScript original, which calls `allowNetwork()` only because its harness
blocks sockets by default).

TypeScript patches the private `runCommand`/`runCommandCapture` methods on the
manager instance to record commands. This port injects a `CommandRunner`
subclass instead (`package_manager.py` takes a `command_runner=` constructor
parameter for exactly this), which records the same `"<command> <args...>"`
strings.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pi_coding_agent.core.package_manager import CommandRunner, PackageManager, SubprocessCommandRunner
from pi_coding_agent.core.settings_manager import SettingsManager

GIT_SOURCE = "git:github.com/test/extension"


def _git(args: list[str], cwd: str | Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: git {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def _init_git_repo(repo_dir: Path) -> None:
    _git(["init", "--initial-branch=main"], repo_dir)
    _git(["config", "--local", "user.email", "test@test.com"], repo_dir)
    _git(["config", "--local", "user.name", "Test"], repo_dir)


def _create_commit(repo_dir: Path, filename: str, content: str, message: str) -> str:
    (repo_dir / filename).write_text(content, encoding="utf-8")
    _git(["add", filename], repo_dir)
    _git(["commit", "-m", message], repo_dir)
    return _git(["rev-parse", "HEAD"], repo_dir)


def _current_commit(repo_dir: Path) -> str:
    return _git(["rev-parse", "HEAD"], repo_dir)


def _file_content(repo_dir: Path, filename: str) -> str:
    return (repo_dir / filename).read_text(encoding="utf-8")


class _RecordingRunner(SubprocessCommandRunner):
    """Records every command, then runs it for real.

    `skip_npm` mirrors the TypeScript override in the "already up to date" case,
    which records `npm ...` invocations but returns without executing them.
    """

    def __init__(self, *, skip_npm: bool = False) -> None:
        self.executed: list[str] = []
        self._skip_npm = skip_npm

    async def run(self, command: str, args: list[str], *, cwd: str | None = None) -> None:
        self.executed.append(f"{command} {' '.join(args)}")
        if self._skip_npm and command == "npm":
            return
        await super().run(command, args, cwd=cwd)


class _Fixture:
    def __init__(self, tmp_path: Path, runner: CommandRunner | None = None) -> None:
        self.temp_dir = tmp_path
        self.remote_dir = tmp_path / "remote"
        self.agent_dir = tmp_path / "agent"
        self.installed_dir = self.agent_dir / "git" / "github.com" / "test" / "extension"
        self.agent_dir.mkdir(parents=True)
        self.settings = SettingsManager.in_memory()
        self.manager = PackageManager(str(self.temp_dir), str(self.agent_dir), self.settings, command_runner=runner)

    def setup_remote_and_install(self, source_override: str | None = None) -> None:
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        _init_git_repo(self.remote_dir)
        _create_commit(self.remote_dir, "extension.ts", "// v1", "Initial commit")

        (self.agent_dir / "git" / "github.com" / "test").mkdir(parents=True, exist_ok=True)
        _git(["clone", str(self.remote_dir), str(self.installed_dir)], self.temp_dir)
        _git(["config", "--local", "user.email", "test@test.com"], self.installed_dir)
        _git(["config", "--local", "user.name", "Test"], self.installed_dir)

        self.settings.set_packages([source_override or GIT_SOURCE])


# ---------------------------------------------------------------------------
# normal updates (no force-push)
# ---------------------------------------------------------------------------


async def test_should_skip_reset_clean_and_install_when_already_up_to_date(tmp_path: Path) -> None:
    runner = _RecordingRunner(skip_npm=True)
    fixture = _Fixture(tmp_path, runner)

    fixture.remote_dir.mkdir(parents=True)
    _init_git_repo(fixture.remote_dir)
    (fixture.remote_dir / "package.json").write_text(json.dumps({"name": "test-extension", "version": "1.0.0"}))
    _create_commit(fixture.remote_dir, "extension.ts", "// v1", "Initial commit")

    (fixture.agent_dir / "git" / "github.com" / "test").mkdir(parents=True)
    _git(["clone", str(fixture.remote_dir), str(fixture.installed_dir)], fixture.temp_dir)
    fixture.settings.set_packages([GIT_SOURCE])

    await fixture.manager.update()

    assert "git fetch --prune --no-tags origin +refs/heads/main:refs/remotes/origin/main" in runner.executed
    assert "git fetch --prune origin" not in runner.executed
    assert "git reset --hard @{upstream}" not in runner.executed
    assert "git reset --hard origin/HEAD" not in runner.executed
    assert "git clean -fdx" not in runner.executed
    assert "npm install" not in runner.executed


async def test_should_update_to_latest_commit_when_remote_has_new_commits(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.setup_remote_and_install()
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v1"

    new_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Second commit")

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == new_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2"


async def test_should_handle_multiple_commits_ahead(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.setup_remote_and_install()

    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Second commit")
    _create_commit(fixture.remote_dir, "extension.ts", "// v3", "Third commit")
    latest_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v4", "Fourth commit")

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == latest_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v4"


async def test_should_update_even_when_local_checkout_has_no_upstream(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    fixture = _Fixture(tmp_path, runner)
    fixture.setup_remote_and_install()
    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Second commit")
    latest_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v3", "Third commit")

    detached_commit = _current_commit(fixture.installed_dir)
    _git(["checkout", detached_commit], fixture.installed_dir)

    await fixture.manager.update()

    assert "git fetch --prune --no-tags origin +refs/heads/main:refs/remotes/origin/main" in runner.executed
    assert _current_commit(fixture.installed_dir) == latest_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v3"


# ---------------------------------------------------------------------------
# force-push scenarios
# ---------------------------------------------------------------------------


async def test_should_recover_when_remote_history_is_rewritten(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.setup_remote_and_install()
    initial_commit = _current_commit(fixture.remote_dir)

    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Commit to keep")

    await fixture.manager.update()
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2"

    _git(["reset", "--hard", initial_commit], fixture.remote_dir)
    rewritten_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v2-rewritten", "Rewritten commit")

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == rewritten_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2-rewritten"


async def test_should_recover_when_local_commit_no_longer_exists_in_remote(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.setup_remote_and_install()

    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Commit A")
    _create_commit(fixture.remote_dir, "extension.ts", "// v3", "Commit B")

    await fixture.manager.update()
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v3"

    _git(["reset", "--hard", "HEAD~2"], fixture.remote_dir)
    new_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v2-new", "New commit replacing A and B")

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == new_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2-new"


async def test_should_handle_complete_history_rewrite(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.setup_remote_and_install()

    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "v2")
    _create_commit(fixture.remote_dir, "extension.ts", "// v3", "v3")

    await fixture.manager.update()
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v3"

    _git(["reset", "--hard", "HEAD~2"], fixture.remote_dir)
    _create_commit(fixture.remote_dir, "extension.ts", "// rewrite-a", "Rewrite A")
    final_commit = _create_commit(fixture.remote_dir, "extension.ts", "// rewrite-b", "Rewrite B")

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == final_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// rewrite-b"


# ---------------------------------------------------------------------------
# pinned sources
# ---------------------------------------------------------------------------


async def test_should_not_move_pinned_git_sources_past_their_configured_ref(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.remote_dir.mkdir(parents=True)
    _init_git_repo(fixture.remote_dir)
    initial_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v1", "Initial commit")

    (fixture.agent_dir / "git" / "github.com" / "test").mkdir(parents=True)
    _git(["clone", str(fixture.remote_dir), str(fixture.installed_dir)], fixture.temp_dir)
    _git(["checkout", initial_commit], fixture.installed_dir)
    _git(["config", "--local", "user.email", "test@test.com"], fixture.installed_dir)
    _git(["config", "--local", "user.name", "Test"], fixture.installed_dir)

    fixture.settings.set_packages([f"{GIT_SOURCE}@{initial_commit}"])

    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Second commit")

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == initial_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v1"


async def test_should_checkout_the_configured_pinned_git_ref_during_full_and_targeted_updates(
    tmp_path: Path,
) -> None:
    fixture = _Fixture(tmp_path)
    fixture.remote_dir.mkdir(parents=True)
    _init_git_repo(fixture.remote_dir)
    v1_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v1", "Initial commit")
    _git(["tag", "v1"], fixture.remote_dir)
    v2_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Second commit")
    _git(["tag", "v2"], fixture.remote_dir)

    (fixture.agent_dir / "git" / "github.com" / "test").mkdir(parents=True)
    _git(["clone", str(fixture.remote_dir), str(fixture.installed_dir)], fixture.temp_dir)
    _git(["checkout", "v1"], fixture.installed_dir)
    assert _current_commit(fixture.installed_dir) == v1_commit

    pinned_source = f"{GIT_SOURCE}@v2"
    fixture.settings.set_packages([pinned_source])

    await fixture.manager.update()

    assert _current_commit(fixture.installed_dir) == v2_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2"

    _git(["checkout", "v1"], fixture.installed_dir)

    await fixture.manager.update(pinned_source)

    assert _current_commit(fixture.installed_dir) == v2_commit
    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2"


async def test_should_not_reset_an_annotated_tag_checkout_that_already_matches_the_configured_ref(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    fixture = _Fixture(tmp_path, runner)
    fixture.remote_dir.mkdir(parents=True)
    _init_git_repo(fixture.remote_dir)
    tagged_commit = _create_commit(fixture.remote_dir, "extension.ts", "// v1", "Initial commit")
    _git(["tag", "-a", "v1", "-m", "v1"], fixture.remote_dir)

    (fixture.agent_dir / "git" / "github.com" / "test").mkdir(parents=True)
    _git(["clone", str(fixture.remote_dir), str(fixture.installed_dir)], fixture.temp_dir)
    _git(["checkout", "v1"], fixture.installed_dir)
    assert _current_commit(fixture.installed_dir) == tagged_commit

    fixture.settings.set_packages([f"{GIT_SOURCE}@v1"])

    await fixture.manager.update()

    assert "git fetch origin v1" in runner.executed
    # The annotated tag already points at HEAD, so `_ensure_git_ref` must take
    # its equal-heads early return: no reset and no clean.
    assert not any(command.startswith("git reset --hard") for command in runner.executed)
    assert "git clean -fdx" not in runner.executed
    assert _current_commit(fixture.installed_dir) == tagged_commit


# ---------------------------------------------------------------------------
# temporary git sources
# ---------------------------------------------------------------------------


class _ScriptedTemporaryRunner(CommandRunner):
    """Records commands and answers `rev-parse` without touching a real repo.

    Port of the TypeScript overrides of both `runCommand` and
    `runCommandCapture` in the temporary-source cases.
    """

    def __init__(self, extension_file: Path | None = None) -> None:
        self.executed: list[str] = []
        self._extension_file = extension_file

    async def run(self, command: str, args: list[str], *, cwd: str | None = None) -> None:
        self.executed.append(f"{command} {' '.join(args)}")
        if self._extension_file is not None and command == "git" and args and args[0] == "reset":
            self._extension_file.write_text("// fresh", encoding="utf-8")

    async def run_capture(
        self,
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        if len(args) >= 2 and args[0] == "rev-parse" and args[1] == "HEAD":
            return "local-head"
        if len(args) >= 2 and args[0] == "rev-parse" and args[1] == "@{upstream}":
            return "remote-head"
        if len(args) >= 2 and args[0] == "rev-parse" and args[1] == "--abbrev-ref":
            return "origin/main"
        return ""


def _prepare_temporary_cache(manager: PackageManager, content: str) -> tuple[Path, Path]:
    cached_dir = Path(manager._get_git_install_path(manager.parse_source(GIT_SOURCE), "temporary"))
    extension_file = cached_dir / "pi-extensions" / "session-breakdown.py"
    if cached_dir.exists():
        subprocess.run(["rm", "-rf", str(cached_dir)], check=True)
    (cached_dir / "pi-extensions").mkdir(parents=True)
    (cached_dir / "package.json").write_text(json.dumps({"pi": {"extensions": ["./pi-extensions"]}}, indent=2))
    extension_file.write_text(content, encoding="utf-8")
    return cached_dir, extension_file


async def test_should_refresh_cached_temporary_git_sources_when_resolving(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    cached_dir, extension_file = _prepare_temporary_cache(fixture.manager, "// stale")

    runner = _ScriptedTemporaryRunner(extension_file)
    fixture.manager._runner = runner

    await fixture.manager.resolve_extension_sources([GIT_SOURCE], temporary=True)

    assert "git fetch --prune --no-tags origin +refs/heads/main:refs/remotes/origin/main" in runner.executed
    assert _file_content(cached_dir, "pi-extensions/session-breakdown.py") == "// fresh"


async def test_should_not_refresh_pinned_temporary_git_sources(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    cached_dir, _extension_file = _prepare_temporary_cache(fixture.manager, "// pinned")

    runner = _ScriptedTemporaryRunner()
    fixture.manager._runner = runner

    await fixture.manager.resolve_extension_sources([f"{GIT_SOURCE}@main"], temporary=True)

    assert runner.executed == []
    assert _file_content(cached_dir, "pi-extensions/session-breakdown.py") == "// pinned"


# ---------------------------------------------------------------------------
# scope-aware update
# ---------------------------------------------------------------------------


async def test_should_not_install_locally_when_source_is_only_registered_globally(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.setup_remote_and_install()

    _create_commit(fixture.remote_dir, "extension.ts", "// v2", "Second commit")

    project_git_dir = fixture.temp_dir / ".pi" / "git" / "github.com" / "test" / "extension"
    assert not project_git_dir.exists()

    await fixture.manager.update(GIT_SOURCE)

    assert _file_content(fixture.installed_dir, "extension.ts") == "// v2"
    assert not os.path.exists(project_git_dir)
