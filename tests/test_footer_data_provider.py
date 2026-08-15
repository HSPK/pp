"""Python port of `packages/coding-agent/test/footer-data-provider.test.ts`.

The TypeScript provider watches git ref files with Node's `fs.watch`; this
port polls instead (see `core/footer_data_provider.py`'s module docstring), so
the watching tests drive `start_watching(poll_interval_ms=...)` rather than
mocking `fs.watch`. Everything else -- HEAD parsing, the reftable `.invalid`
fallback through `git symbolic-ref`, debouncing, and the change-notification
contract -- ports directly.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from pi_coding_agent.core import footer_data_provider
from pi_coding_agent.core.footer_data_provider import FooterDataProvider


@dataclass
class _WorktreeFixture:
    worktree_dir: Path
    reftable_dir: Path


class _BranchResolver:
    """Stand-in for the TypeScript test's mocked `spawnSync`/`execFile`."""

    def __init__(self, branch: str | None = "main") -> None:
        self.branch = branch
        self.calls: list[str] = []

    def __call__(self, repo_dir: str) -> str | None:
        self.calls.append(repo_dir)
        return self.branch or None


def _create_plain_reftable_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git" / "reftable").mkdir(parents=True)
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/.invalid\n", encoding="utf-8")
    return repo_dir


def _create_plain_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return repo_dir


def _create_reftable_worktree(tmp_path: Path) -> _WorktreeFixture:
    repo_dir = tmp_path / "repo"
    common_git_dir = repo_dir / ".git"
    git_dir = common_git_dir / "worktrees" / "src"
    worktree_dir = tmp_path / "worktree"
    reftable_dir = common_git_dir / "reftable"

    git_dir.mkdir(parents=True)
    reftable_dir.mkdir(parents=True)
    worktree_dir.mkdir(parents=True)

    (worktree_dir / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/.invalid\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (reftable_dir / "tables.list").write_text("0\n", encoding="utf-8")

    return _WorktreeFixture(worktree_dir=worktree_dir, reftable_dir=reftable_dir)


async def _wait_for(condition: object, timeout_s: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    while not condition():  # type: ignore[operator]
        if loop.time() - started_at > timeout_s:
            raise AssertionError("Timed out waiting for condition")
        await asyncio.sleep(0.01)


class _PollCounter:
    """Counts `_watch_loop` poll cycles so tests can wait on progress, not the clock.

    Several assertions here are negative ("the loop had chances to act and did
    not"). Spelling that as `await asyncio.sleep(0.1)` makes them depend on
    wall-clock time, which is unreliable now that the suite runs in parallel
    under load. Counting the loop's own poll sleeps pins the same claim
    deterministically: after N observed polls, the loop has genuinely had N
    opportunities.

    The replacement is an `async def` with the same signature as the real
    module-level `_sleep`, and it still performs the real delay, so the watch
    loop cannot be satisfied more easily than in production.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, poll_interval_ms: int) -> None:
        self.polls = 0
        poll_seconds = poll_interval_ms / 1000
        real_sleep = asyncio.sleep

        async def counting_sleep(delay: float) -> None:
            await real_sleep(delay)
            if delay == poll_seconds:
                self.polls += 1

        monkeypatch.setattr(footer_data_provider, "_sleep", counting_sleep)

    async def wait_for_polls(self, count: int) -> None:
        target = self.polls + count
        await _wait_for(lambda: self.polls >= target)


def test_uses_head_directly_in_a_regular_repo_from_a_nested_directory(tmp_path: Path) -> None:
    repo_dir = _create_plain_repo(tmp_path)
    nested_dir = repo_dir / "src" / "nested"
    nested_dir.mkdir(parents=True)
    resolver = _BranchResolver()

    provider = FooterDataProvider(str(nested_dir), resolve_branch=resolver)
    try:
        assert provider.get_git_branch() == "main"
        assert resolver.calls == []
    finally:
        provider.dispose()


def test_resolves_the_branch_via_git_when_head_is_invalid_in_a_reftable_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_dir = _create_plain_reftable_repo(tmp_path)
    recorded: list[tuple[list[str], dict[str, object]]] = []
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"main\n", stderr=b"")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        recorded.append((args, kwargs))
        return completed

    monkeypatch.setattr("pi_coding_agent.core.footer_data_provider.subprocess.run", fake_run)

    provider = FooterDataProvider(str(repo_dir))
    try:
        assert provider.get_git_branch() == "main"
        assert len(recorded) == 1
        args, kwargs = recorded[0]
        assert args == ["git", "--no-optional-locks", "symbolic-ref", "--quiet", "--short", "HEAD"]
        assert str(kwargs["cwd"]).endswith("repo")
        # TypeScript pins `stdio: ["ignore", "pipe", "ignore"]` and `encoding: "utf8"`.
        # `subprocess.run` spells the stdin half as `stdin=DEVNULL`; stdout is piped by
        # `capture_output=True` and decoded as utf-8 by the caller.
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["capture_output"] is True
    finally:
        provider.dispose()


def test_resolves_the_branch_via_git_in_a_reftable_backed_worktree(tmp_path: Path) -> None:
    fixture = _create_reftable_worktree(tmp_path)
    resolver = _BranchResolver()

    provider = FooterDataProvider(str(fixture.worktree_dir), resolve_branch=resolver)
    try:
        assert provider.get_git_branch() == "main"
    finally:
        provider.dispose()


def test_treats_an_unresolved_invalid_reftable_head_as_detached(tmp_path: Path) -> None:
    repo_dir = _create_plain_reftable_repo(tmp_path)
    resolver = _BranchResolver(branch="")

    provider = FooterDataProvider(str(repo_dir), resolve_branch=resolver)
    try:
        assert provider.get_git_branch() == "detached"
    finally:
        provider.dispose()


async def test_does_not_notify_listeners_when_reftable_updates_keep_the_same_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _create_reftable_worktree(tmp_path)
    resolver = _BranchResolver()

    polls = _PollCounter(monkeypatch, poll_interval_ms=10)
    provider = FooterDataProvider(str(fixture.worktree_dir), resolve_branch=resolver)
    provider.WATCH_DEBOUNCE_MS = 20
    try:
        assert provider.get_git_branch() == "main"
        resolver.calls.clear()
        changes: list[None] = []
        provider.on_branch_change(lambda: changes.append(None))
        provider.start_watching(poll_interval_ms=10)

        (fixture.reftable_dir / "tables.list").write_text("1\n", encoding="utf-8")
        await _wait_for(lambda: len(resolver.calls) >= 1)
        # Give the loop three further poll cycles to refresh again if it were
        # going to; it must not, because the ref state has not changed since.
        await polls.wait_for_polls(3)

        assert len(resolver.calls) == 1
        assert provider.get_git_branch() == "main"
        assert changes == []
    finally:
        provider.dispose()


async def test_debounces_rapid_reftable_updates_into_a_single_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _create_reftable_worktree(tmp_path)
    resolver = _BranchResolver()

    # The three writes must land inside ONE debounce window for a single
    # refresh to be the correct answer. Spacing them with `asyncio.sleep(0.02)`
    # under a wall-clock debounce only achieves that on an idle machine: under
    # parallel load a 20 ms sleep can overshoot the 100 ms window, the third
    # write lands after `_watch_loop`'s post-debounce re-snapshot, and a second
    # refresh is then correct behaviour rather than a bug. So the debounce wait
    # is driven explicitly instead: it blocks until the test has finished
    # writing, which is the condition the assertion actually depends on.
    writes_done = asyncio.Event()
    debounce_entered = asyncio.Event()
    poll_seconds = 0.01
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        if delay == poll_seconds:
            await real_sleep(delay)
            return
        debounce_entered.set()
        await writes_done.wait()

    monkeypatch.setattr(footer_data_provider, "_sleep", fake_sleep)

    provider = FooterDataProvider(str(fixture.worktree_dir), resolve_branch=resolver)
    provider.WATCH_DEBOUNCE_MS = 100
    try:
        assert provider.get_git_branch() == "main"
        resolver.calls.clear()
        provider.start_watching(poll_interval_ms=int(poll_seconds * 1000))

        for value in ("1", "2", "3"):
            (fixture.reftable_dir / "tables.list").write_text(f"{value}\n", encoding="utf-8")
            await _wait_for(debounce_entered.is_set)
        writes_done.set()

        await _wait_for(lambda: len(resolver.calls) >= 1)
        # Give the loop room to refresh a second time if it were going to.
        for _ in range(20):
            await real_sleep(0)

        assert len(resolver.calls) == 1
    finally:
        provider.dispose()


async def test_updates_the_cached_branch_when_the_reftable_directory_changes(tmp_path: Path) -> None:
    fixture = _create_reftable_worktree(tmp_path)
    resolver = _BranchResolver()

    provider = FooterDataProvider(str(fixture.worktree_dir), resolve_branch=resolver)
    provider.WATCH_DEBOUNCE_MS = 20
    try:
        assert provider.get_git_branch() == "main"
        resolver.branch = "foo"
        resolver.calls.clear()
        changes: list[None] = []
        provider.on_branch_change(lambda: changes.append(None))
        provider.start_watching(poll_interval_ms=10)

        (fixture.reftable_dir / "tables.list").write_text("1\n", encoding="utf-8")
        await _wait_for(lambda: provider.get_git_branch() == "foo")

        assert len(resolver.calls) == 1
        assert provider.get_git_branch() == "foo"
        assert len(changes) == 1
    finally:
        provider.dispose()


# The TypeScript test "retries git watchers 5 seconds after an async fs.watch
# error" pokes `provider.headWatcher`, an `fs.FSWatcher`, and emits a synthetic
# `error` event on it. Node's `fs.watch` has no standard-library equivalent in
# Python, so this port polls the git ref state instead of installing watchers
# (documented in `core/footer_data_provider.py`); there is no watcher object to
# fail and no watcher to re-create. What that TypeScript case actually pins --
# a failing watch does not permanently stop branch tracking -- does have a
# counterpart, and is asserted below against the polling loop.


async def test_recovers_from_a_transient_git_ref_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _create_reftable_worktree(tmp_path)
    resolver = _BranchResolver()

    polls = _PollCounter(monkeypatch, poll_interval_ms=10)
    provider = FooterDataProvider(str(fixture.worktree_dir), resolve_branch=resolver)
    provider.WATCH_DEBOUNCE_MS = 20
    try:
        assert provider.get_git_branch() == "main"
        provider.start_watching(poll_interval_ms=10)

        # Make the ref state unreadable, the way an EMFILE/ENOENT would break
        # `fs.watch` in TypeScript, and let the loop poll through it.
        tables = fixture.reftable_dir / "tables.list"
        tables.unlink()
        fixture.reftable_dir.rmdir()
        await polls.wait_for_polls(3)

        resolver.branch = "foo"
        fixture.reftable_dir.mkdir()
        tables.write_text("1\n", encoding="utf-8")

        await _wait_for(lambda: provider.get_git_branch() == "foo")
    finally:
        provider.dispose()
