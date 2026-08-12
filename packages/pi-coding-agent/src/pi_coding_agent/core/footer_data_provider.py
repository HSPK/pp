"""Git branch and extension status data for the footer.

Ported from ``packages/coding-agent/src/core/footer-data-provider.ts``.

``find_git_paths`` already lives in :mod:`pi_coding_agent.core.resource_loader`
and is re-exported here to match the TypeScript module's public surface.

Watching deviates from TypeScript. Node's ``fs.watch`` gives inotify-backed
directory events (wrapped by ``utils/fs-watch.ts``'s ``watchWithErrorHandler``/
``closeWatcher``, which this port does not use); the standard library has no
equivalent, so this port polls the git ref state (HEAD content plus the
reftable directory's mtimes) on an interval instead. The observable contract
is the same: ``on_branch_change`` callbacks fire after the branch actually
changes, coalesced by the same 500ms debounce. ``utils/fs-watch.ts``'s only
other caller, the interactive theme file watcher, is out of scope for the
same reason (see ``modes/interactive/theme/theme.py``'s module docstring).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from .resource_loader import GitPaths, find_git_paths

WATCH_DEBOUNCE_MS = 500
WATCH_POLL_INTERVAL_MS = 250
_HEAD_REF_PREFIX = "ref: refs/heads/"
_WINDOWS_MOUNT_RE = re.compile(r"^/mnt/[a-z](?:/|$)", re.IGNORECASE)


async def _sleep(delay: float) -> None:
    """Seam for the poll and debounce waits in `_watch_loop`.

    Tests that pin the debounce need every write to land inside one debounce
    window. Asserting that with wall-clock sleeps only holds when the machine
    is idle, so they replace this function instead of racing the real clock.
    """
    await asyncio.sleep(delay)


_UNSET = object()


def _resolve_branch_with_git(repo_dir: str, timeout: float = 5.0) -> str | None:
    """Ask git for the current branch. ``None`` on detached HEAD or no git."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo_dir,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.decode("utf-8", errors="replace").strip()
    return branch or None


def is_wsl_environment(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return os.name != "nt" and bool(env.get("WSL_DISTRO_NAME") or env.get("WSL_INTEROP"))


def is_windows_mounted_repo_path(repo_dir: str) -> bool:
    return _WINDOWS_MOUNT_RE.match(repo_dir) is not None


def should_poll_git_head(repo_dir: str, env: Mapping[str, str] | None = None) -> bool:
    return is_wsl_environment(env) and is_windows_mounted_repo_path(repo_dir)


class FooterDataProvider:
    """Git branch and extension statuses - data not otherwise accessible to
    extensions. Token stats and model info come from the session manager."""

    WATCH_DEBOUNCE_MS = WATCH_DEBOUNCE_MS

    def __init__(self, cwd: str, *, resolve_branch: Callable[[str], str | None] | None = None) -> None:
        self._cwd = cwd
        self._extension_statuses: dict[str, str] = {}
        # `_UNSET` distinguishes "not resolved yet" from a resolved `None`
        # (not a git repo), matching the TS `undefined` vs `null` split.
        self._cached_branch: str | object | None = _UNSET
        self._git_paths: GitPaths | None = find_git_paths(cwd)
        self._branch_change_callbacks: list[Callable[[], None]] = []
        self._available_provider_count = 0
        self._disposed = False
        self._watch_task: asyncio.Task[None] | None = None
        self._resolve_branch = resolve_branch or _resolve_branch_with_git

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def git_paths(self) -> GitPaths | None:
        return self._git_paths

    def get_git_branch(self) -> str | None:
        """Current git branch, ``None`` if not in a repo, ``"detached"`` if detached."""
        if self._cached_branch is _UNSET:
            self._cached_branch = self._resolve_git_branch()
        return self._cached_branch  # type: ignore[return-value]

    def get_extension_statuses(self) -> Mapping[str, str]:
        """Extension status texts set via ``ctx.ui.set_status()``."""
        return self._extension_statuses

    def on_branch_change(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to git branch changes. Returns an unsubscribe function."""
        self._branch_change_callbacks.append(callback)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._branch_change_callbacks.remove(callback)

        return unsubscribe

    def set_extension_status(self, key: str, text: str | None) -> None:
        if text is None:
            self._extension_statuses.pop(key, None)
        else:
            self._extension_statuses[key] = text

    def clear_extension_statuses(self) -> None:
        self._extension_statuses.clear()

    def get_available_provider_count(self) -> int:
        """Number of unique providers with available models (for footer display)."""
        return self._available_provider_count

    def set_available_provider_count(self, count: int) -> None:
        self._available_provider_count = count

    def set_cwd(self, cwd: str) -> None:
        if self._cwd == cwd:
            return
        self._cwd = cwd
        self._cached_branch = _UNSET
        self._git_paths = find_git_paths(cwd)
        self._notify_branch_change()

    def dispose(self) -> None:
        self._disposed = True
        self.stop_watching()
        self._branch_change_callbacks.clear()

    def _notify_branch_change(self) -> None:
        for callback in list(self._branch_change_callbacks):
            callback()

    def _resolve_git_branch(self) -> str | None:
        if not self._git_paths:
            return None
        try:
            content = Path(self._git_paths.head_path).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith(_HEAD_REF_PREFIX):
            branch = content[len(_HEAD_REF_PREFIX) :]
            if branch == ".invalid":
                return self._resolve_branch(self._git_paths.repo_dir) or "detached"
            return branch
        return "detached"

    def refresh_git_branch(self) -> bool:
        """Re-resolve the branch, notifying subscribers when it changed.

        Returns whether a change was published. The first resolution only seeds
        the cache; TS likewise stays silent when `cachedBranch` was `undefined`.
        """
        if self._disposed:
            return False
        next_branch = self._resolve_git_branch()
        was_seeded = self._cached_branch is not _UNSET
        changed = was_seeded and self._cached_branch != next_branch
        self._cached_branch = next_branch
        if changed:
            self._notify_branch_change()
        return changed

    def _git_ref_state(self) -> tuple[object, ...]:
        """Cheap fingerprint of everything a branch switch can touch."""
        if not self._git_paths:
            return ()
        state: list[object] = []
        try:
            state.append(Path(self._git_paths.head_path).read_text(encoding="utf-8"))
        except OSError:
            state.append(None)
        reftable_dir = Path(self._git_paths.common_git_dir) / "reftable"
        with contextlib.suppress(OSError):
            for entry in sorted(reftable_dir.iterdir()):
                stat = entry.stat()
                state.append((entry.name, stat.st_mtime_ns, stat.st_size))
        return tuple(state)

    def start_watching(self, *, poll_interval_ms: int = WATCH_POLL_INTERVAL_MS) -> None:
        """Poll git ref state and publish branch changes until disposed."""
        if self._disposed or self._watch_task is not None:
            return
        # Snapshot synchronously: the coroutine body does not run until the
        # caller next awaits, and a branch switch in that window would
        # otherwise be baked into the baseline and never reported.
        baseline = self._git_ref_state()
        self._watch_task = asyncio.ensure_future(self._watch_loop(poll_interval_ms, baseline))

    def stop_watching(self) -> None:
        task = self._watch_task
        self._watch_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _watch_loop(self, poll_interval_ms: int, baseline: tuple[object, ...]) -> None:
        previous = baseline
        try:
            while not self._disposed:
                await _sleep(poll_interval_ms / 1000)
                current = self._git_ref_state()
                if current == previous:
                    continue
                previous = current
                # Debounce: git writes several files per branch switch.
                await _sleep(self.WATCH_DEBOUNCE_MS / 1000)
                previous = self._git_ref_state()
                if self._disposed:
                    return
                self.refresh_git_branch()
        except asyncio.CancelledError:
            pass


__all__ = [
    "WATCH_DEBOUNCE_MS",
    "WATCH_POLL_INTERVAL_MS",
    "FooterDataProvider",
    "GitPaths",
    "find_git_paths",
    "is_windows_mounted_repo_path",
    "is_wsl_environment",
    "should_poll_git_head",
]
