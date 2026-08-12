"""Project trust store.

Ported from ``packages/coding-agent/src/core/trust-manager.ts``.

The TypeScript version guards ``trust.json`` with ``proper-lockfile``; this port
uses an equivalent directory-based lock (``trust.json.lock``) with the same
10-attempt / 20ms retry policy, so concurrent ``pp`` processes cannot
interleave a read-modify-write and lose a decision.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.paths import canonicalize_path, resolve_path
from .config import CONFIG_DIR_NAME

ProjectTrustDecision = bool | None

TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES = (
    "settings.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
)


@dataclass
class ProjectTrustStoreEntry:
    path: str
    decision: bool


@dataclass
class ProjectTrustUpdate:
    path: str
    decision: ProjectTrustDecision


@dataclass
class ProjectTrustOption:
    label: str
    trusted: bool
    updates: list[ProjectTrustUpdate] = field(default_factory=list)
    saved_path: str | None = None


def _normalize_cwd(cwd: str) -> str:
    return canonicalize_path(resolve_path(cwd))


def _parent_dir(path: str) -> str:
    return os.path.dirname(path)


def _find_nearest_trust_entry(data: dict[str, Any], cwd: str) -> ProjectTrustStoreEntry | None:
    current_dir = _normalize_cwd(cwd)
    while True:
        value = data.get(current_dir)
        if value is True or value is False:
            return ProjectTrustStoreEntry(path=current_dir, decision=value)

        parent_dir = _parent_dir(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def get_project_trust_parent_path(cwd: str) -> str | None:
    trust_path = _normalize_cwd(cwd)
    parent_dir = _parent_dir(trust_path)
    return None if parent_dir == trust_path else parent_dir


def get_project_trust_options(cwd: str, *, include_session_only: bool = False) -> list[ProjectTrustOption]:
    trust_path = _normalize_cwd(cwd)
    trust_options: list[ProjectTrustOption] = [
        ProjectTrustOption(
            label="Trust",
            trusted=True,
            updates=[ProjectTrustUpdate(path=trust_path, decision=True)],
            saved_path=trust_path,
        )
    ]
    parent_path = get_project_trust_parent_path(cwd)
    if parent_path is not None:
        trust_options.append(
            ProjectTrustOption(
                label=f"Trust parent folder ({parent_path})",
                trusted=True,
                updates=[
                    ProjectTrustUpdate(path=parent_path, decision=True),
                    ProjectTrustUpdate(path=trust_path, decision=None),
                ],
                saved_path=parent_path,
            )
        )
    if include_session_only:
        trust_options.append(ProjectTrustOption(label="Trust (this session only)", trusted=True))
    trust_options.append(
        ProjectTrustOption(
            label="Do not trust",
            trusted=False,
            updates=[ProjectTrustUpdate(path=trust_path, decision=False)],
            saved_path=trust_path,
        )
    )
    if include_session_only:
        trust_options.append(ProjectTrustOption(label="Do not trust (this session only)", trusted=False))
    return trust_options


def _read_trust_file(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to read trust store {path}: {error}") from error

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid trust store {path}: expected an object")

    data: dict[str, Any] = {}
    for key, value in parsed.items():
        if value is not True and value is not False and value is not None:
            raise RuntimeError(f"Invalid trust store {path}: value for {json.dumps(key)} must be true, false, or null")
        data[key] = value
    return data


def _write_trust_file(path: str, data: dict[str, Any]) -> None:
    sorted_data: dict[str, Any] = {}
    for key in sorted(data.keys()):
        value = data[key]
        if value is True or value is False or value is None:
            sorted_data[key] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(sorted_data, indent=2) + "\n")


class _TrustFileLock:
    """Directory-based advisory lock matching the TS ``proper-lockfile`` usage."""

    MAX_ATTEMPTS = 10
    DELAY_SECONDS = 0.02

    def __init__(self, path: str) -> None:
        self._lock_path = f"{path}.lock"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> _TrustFileLock:
        last_error: OSError | None = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                os.mkdir(self._lock_path)
                return self
            except OSError as error:
                if error.errno != errno.EEXIST or attempt == self.MAX_ATTEMPTS:
                    raise
                last_error = error
                time.sleep(self.DELAY_SECONDS)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to acquire trust store lock")

    def __exit__(self, *_exc: object) -> None:
        with contextlib.suppress(OSError):
            os.rmdir(self._lock_path)


def has_trust_requiring_project_resources(cwd: str) -> bool:
    """Whether ``cwd`` has project-local resources that must be gated by trust.

    True when trust-requiring entries exist under ``cwd/.pi``, or when
    ``.agents/skills`` exists in ``cwd`` or one of its ancestors. The
    user/global ``~/.agents/skills`` directory is always a trusted user
    resource and is ignored here, even when ``cwd`` is ``$HOME``.
    """
    home_dir = canonicalize_path(resolve_path(os.environ.get("HOME") or str(Path.home())))
    user_agents_skills_dir = os.path.join(home_dir, ".agents", "skills")
    current_dir = canonicalize_path(resolve_path(cwd))

    config_dir = os.path.join(current_dir, CONFIG_DIR_NAME)
    if any(os.path.exists(os.path.join(config_dir, entry)) for entry in TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES):
        return True

    while True:
        agents_skills_dir = os.path.join(current_dir, ".agents", "skills")
        if agents_skills_dir != user_agents_skills_dir and os.path.exists(agents_skills_dir):
            return True

        parent_dir = _parent_dir(current_dir)
        if parent_dir == current_dir:
            return False
        current_dir = parent_dir


class ProjectTrustStore:
    def __init__(self, agent_dir: str) -> None:
        self.trust_path = os.path.join(resolve_path(agent_dir), "trust.json")

    def get(self, cwd: str) -> ProjectTrustDecision:
        entry = self.get_entry(cwd)
        return entry.decision if entry is not None else None

    def get_entry(self, cwd: str) -> ProjectTrustStoreEntry | None:
        with _TrustFileLock(self.trust_path):
            data = _read_trust_file(self.trust_path)
            return _find_nearest_trust_entry(data, cwd)

    def set(self, cwd: str, decision: ProjectTrustDecision) -> None:
        self.set_many([ProjectTrustUpdate(path=cwd, decision=decision)])

    def set_many(self, decisions: list[ProjectTrustUpdate]) -> None:
        with _TrustFileLock(self.trust_path):
            data = _read_trust_file(self.trust_path)
            for update in decisions:
                key = _normalize_cwd(update.path)
                if update.decision is None:
                    data.pop(key, None)
                else:
                    data[key] = update.decision
            _write_trust_file(self.trust_path, data)
