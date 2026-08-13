"""Persisted settings storage, migration, and typed accessors.

Python port of `packages/coding-agent/src/core/settings-manager.ts`. Settings
are stored as two loosely-typed JSON documents (global at
``<agent_dir>/settings.json`` and project at ``<cwd>/.pi/settings.json``),
deep-merged with project values overriding global ones. The manager also
tracks which top-level (and nested) fields were explicitly modified during
the session so that concurrent external edits to *other* fields survive a
save instead of being clobbered by stale in-memory state (see
``settings-manager-bug.test.ts``, ported as ``test_settings_manager.py``'s
``TestExternalEditPreservation``).

Settings are represented as a plain ``dict[str, Any]`` (matching the loosely
typed `Settings` TypeScript interface, whose ~50 fields are all optional)
rather than as a dataclass with one attribute per field; every getter/setter
below has the same name and default-value behavior as its TypeScript
counterpart.

**File locking deviation.** The TypeScript implementation uses
`proper-lockfile` (a directory-based advisory lock with staleness detection)
via a synchronous busy-wait retry loop. Python has no dependency-free
equivalent with staleness detection, so `FileSettingsStorage` uses a simpler
atomic-create lock file (`<path>.lock`, created with `O_CREAT | O_EXCL`) with
the same retry-with-delay shape. This is intentionally simpler than
`proper-lockfile` (no stale-lock takeover) since within-session/tests only
ever exercise a single writer.

**Write-queue deviation.** TypeScript serializes concurrent async saves
through a `Promise` chain (`writeQueue`) so overlapping `save()` calls apply
in order; `flush()` awaits it. This port's I/O is synchronous (`with_lock`
runs to completion before the setter returns), so there is no queue to
serialize or await — `flush()` is a no-op kept for API parity, and
`SettingsManager.reload()`'s no-arg pre-reload wait (`await this.writeQueue`)
has nothing to await.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pi_coding_agent.core.config import CONFIG_DIR_NAME, get_agent_dir
from pi_coding_agent.utils.paths import normalize_path, resolve_path

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000

SettingsScope = Literal["global", "project"]
Settings = dict[str, Any]


def _parse_http_idle_timeout_ms(value: Any) -> int | None:
    """Port of `parseHttpIdleTimeoutMs` from `core/http-dispatcher.ts`."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if not trimmed:
            return None
        try:
            return _parse_http_idle_timeout_ms(float(trimmed))
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:  # NaN/inf guard
        return None
    return int(value)


def _parse_timeout_setting(value: Any, setting_name: str) -> int | None:
    timeout_ms = _parse_http_idle_timeout_ms(value)
    if timeout_ms is not None:
        return timeout_ms
    if value is not None:
        raise ValueError(f"Invalid {setting_name} setting: {value}")
    return None


def _is_mergeable_object(value: Any) -> bool:
    return isinstance(value, dict)


def deep_merge_objects(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts; ``None`` in overrides is treated as "not set" (like `undefined`)."""
    result = dict(base)
    for key, override_value in overrides.items():
        if override_value is None:
            continue
        base_value = base.get(key)
        if _is_mergeable_object(base_value) and _is_mergeable_object(override_value):
            result[key] = deep_merge_objects(base_value, override_value)
        else:
            result[key] = override_value
    return result


def deep_merge_settings(base: Settings, overrides: Settings) -> Settings:
    """Deep merge settings: project/overrides take precedence, nested objects merge recursively."""
    return deep_merge_objects(base, overrides)


@dataclass
class SettingsManagerCreateOptions:
    project_trusted: bool = True


@dataclass
class SettingsError:
    scope: SettingsScope
    error: Exception


class SettingsStorage:
    """Storage backend contract: read-modify-write a scope's JSON text under a lock."""

    def with_lock(self, scope: SettingsScope, fn: Callable[[str | None], str | None]) -> None:
        raise NotImplementedError


class FileSettingsStorage(SettingsStorage):
    """Reads/writes ``settings.json`` under ``<agent_dir>`` and ``<cwd>/<CONFIG_DIR_NAME>``."""

    def __init__(self, cwd: str, agent_dir: str) -> None:
        resolved_cwd = resolve_path(cwd)
        resolved_agent_dir = resolve_path(agent_dir)
        self._global_settings_path = str(Path(resolved_agent_dir) / "settings.json")
        self._project_settings_path = str(Path(resolved_cwd) / CONFIG_DIR_NAME / "settings.json")

    def _acquire_lock_sync_with_retry(self, path: str) -> Callable[[], None]:
        lock_path = f"{path}.lock"
        max_attempts = 10
        delay_s = 0.02
        last_error: OSError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)

                def release() -> None:
                    with contextlib.suppress(OSError):
                        os.remove(lock_path)

                return release
            except FileExistsError as error:
                last_error = error
                if attempt == max_attempts:
                    raise
                time.sleep(delay_s)

        raise last_error or RuntimeError("Failed to acquire settings lock")

    def with_lock(self, scope: SettingsScope, fn: Callable[[str | None], str | None]) -> None:
        path = self._global_settings_path if scope == "global" else self._project_settings_path
        directory = os.path.dirname(path)

        release: Callable[[], None] | None = None
        try:
            file_exists = os.path.exists(path)
            if file_exists:
                release = self._acquire_lock_sync_with_retry(path)
            current = None
            if file_exists:
                with open(path, encoding="utf-8") as fh:
                    current = fh.read()
            next_content = fn(current)
            if next_content is not None:
                if not os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)
                if release is None:
                    release = self._acquire_lock_sync_with_retry(path)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(next_content)
        finally:
            if release is not None:
                release()


class InMemorySettingsStorage(SettingsStorage):
    """No file I/O; keeps global/project scopes as strings in memory."""

    def __init__(self) -> None:
        self._global: str | None = None
        self._project: str | None = None

    def with_lock(self, scope: SettingsScope, fn: Callable[[str | None], str | None]) -> None:
        current = self._global if scope == "global" else self._project
        next_content = fn(current)
        if next_content is not None:
            if scope == "global":
                self._global = next_content
            else:
                self._project = next_content


def _migrate_settings(settings: dict[str, Any]) -> Settings:
    """Migrate old settings shapes to the current one. Mutates and returns ``settings``."""
    # queueMode -> steeringMode
    if "queueMode" in settings and "steeringMode" not in settings:
        settings["steeringMode"] = settings.pop("queueMode")
    settings.pop("queueMode", None)

    # legacy websockets boolean -> transport enum
    if "transport" not in settings and isinstance(settings.get("websockets"), bool):
        settings["transport"] = "websocket" if settings["websockets"] else "sse"
        settings.pop("websockets", None)

    # old skills object format -> new array format
    skills = settings.get("skills")
    if "skills" in settings and isinstance(skills, dict):
        enable_skill_commands = skills.get("enableSkillCommands")
        if enable_skill_commands is not None and settings.get("enableSkillCommands") is None:
            settings["enableSkillCommands"] = enable_skill_commands
        custom_directories = skills.get("customDirectories")
        if isinstance(custom_directories, list) and len(custom_directories) > 0:
            settings["skills"] = custom_directories
        else:
            settings.pop("skills", None)

    # retry.maxDelayMs -> retry.provider.maxRetryDelayMs
    retry = settings.get("retry")
    if "retry" in settings and isinstance(retry, dict):
        provider_settings = retry.get("provider") if isinstance(retry.get("provider"), dict) else None
        max_delay_ms = retry.get("maxDelayMs")
        if (
            isinstance(max_delay_ms, (int, float))
            and not isinstance(max_delay_ms, bool)
            and (provider_settings is None or provider_settings.get("maxRetryDelayMs") is None)
        ):
            retry["provider"] = {**(provider_settings or {}), "maxRetryDelayMs": max_delay_ms}
        retry.pop("maxDelayMs", None)

    return settings


class SettingsManager:
    """Deep-merged global/project settings with save-time field tracking."""

    def __init__(
        self,
        storage: SettingsStorage,
        initial_global: Settings,
        initial_project: Settings,
        global_load_error: Exception | None = None,
        project_load_error: Exception | None = None,
        initial_errors: list[SettingsError] | None = None,
        project_trusted: bool = True,
    ) -> None:
        self._storage = storage
        self._global_settings = initial_global
        self._project_settings = initial_project
        self._project_trusted = project_trusted
        self._global_settings_load_error = global_load_error
        self._project_settings_load_error = project_load_error
        self._errors: list[SettingsError] = list(initial_errors or [])
        # Fields (and nested keys within them) modified during this session, tracked
        # separately per scope so save() only overrides what this session actually
        # changed and doesn't clobber concurrent external edits to other fields.
        self._modified_fields: set[str] = set()
        self._modified_nested_fields: dict[str, set[str]] = {}
        self._modified_project_fields: set[str] = set()
        self._modified_project_nested_fields: dict[str, set[str]] = {}
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    # -- construction -----------------------------------------------------

    @staticmethod
    def create(
        cwd: str,
        agent_dir: str | None = None,
        options: SettingsManagerCreateOptions | None = None,
    ) -> SettingsManager:
        """Create a SettingsManager that loads from files."""
        storage = FileSettingsStorage(cwd, agent_dir if agent_dir is not None else get_agent_dir())
        return SettingsManager.from_storage(storage, options)

    @staticmethod
    def from_storage(storage: SettingsStorage, options: SettingsManagerCreateOptions | None = None) -> SettingsManager:
        """Create a SettingsManager from an arbitrary storage backend."""
        options = options or SettingsManagerCreateOptions()
        project_trusted = options.project_trusted
        global_settings, global_error = SettingsManager._try_load_from_storage(storage, "global")
        project_settings, project_error = SettingsManager._try_load_from_storage(storage, "project", project_trusted)
        initial_errors: list[SettingsError] = []
        if global_error is not None:
            initial_errors.append(SettingsError("global", global_error))
        if project_error is not None:
            initial_errors.append(SettingsError("project", project_error))

        return SettingsManager(
            storage,
            global_settings,
            project_settings,
            global_error,
            project_error,
            initial_errors,
            project_trusted,
        )

    @staticmethod
    def in_memory(
        settings: Settings | None = None, options: SettingsManagerCreateOptions | None = None
    ) -> SettingsManager:
        """Create an in-memory SettingsManager (no file I/O)."""
        storage = InMemorySettingsStorage()
        initial_settings = _migrate_settings(copy.deepcopy(settings or {}))
        storage.with_lock("global", lambda _current: json.dumps(initial_settings, indent=2))
        return SettingsManager.from_storage(storage, options)

    @staticmethod
    def _load_from_storage(storage: SettingsStorage, scope: SettingsScope, project_trusted: bool = True) -> Settings:
        if scope == "project" and not project_trusted:
            return {}

        content: str | None = None

        def capture(current: str | None) -> None:
            nonlocal content
            content = current
            return None

        storage.with_lock(scope, capture)

        if not content:
            return {}
        settings = json.loads(content)
        return _migrate_settings(settings)

    @staticmethod
    def _try_load_from_storage(
        storage: SettingsStorage, scope: SettingsScope, project_trusted: bool = True
    ) -> tuple[Settings, Exception | None]:
        try:
            return SettingsManager._load_from_storage(storage, scope, project_trusted), None
        except Exception as error:
            return {}, error

    # -- scope/trust state --------------------------------------------------

    def get_global_settings(self) -> Settings:
        return copy.deepcopy(self._global_settings)

    def get_project_settings(self) -> Settings:
        return copy.deepcopy(self._project_settings)

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    def set_project_trusted(self, trusted: bool) -> None:
        if self._project_trusted == trusted:
            return

        self._project_trusted = trusted
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

        if not trusted:
            self._project_settings = {}
            self._project_settings_load_error = None
            self._settings = deep_merge_settings(self._global_settings, self._project_settings)
            return

        project_settings, project_error = self._try_load_from_storage(self._storage, "project", trusted)
        self._project_settings = project_settings
        self._project_settings_load_error = project_error
        if project_error is not None:
            self._record_error("project", project_error)
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    async def reload(self) -> None:
        global_settings, global_error = self._try_load_from_storage(self._storage, "global")
        if global_error is None:
            self._global_settings = global_settings
            self._global_settings_load_error = None
        else:
            self._global_settings_load_error = global_error
            self._record_error("global", global_error)

        self._modified_fields.clear()
        self._modified_nested_fields.clear()
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

        project_settings, project_error = self._try_load_from_storage(self._storage, "project", self._project_trusted)
        if project_error is None:
            self._project_settings = project_settings
            self._project_settings_load_error = None
        else:
            self._project_settings_load_error = project_error
            self._record_error("project", project_error)

        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    def apply_overrides(self, overrides: Settings) -> None:
        """Apply additional overrides on top of current settings (not persisted)."""
        self._settings = deep_merge_settings(self._settings, overrides)

    # -- change tracking / persistence ---------------------------------------

    def _mark_modified(self, field_name: str, nested_key: str | None = None) -> None:
        self._modified_fields.add(field_name)
        if nested_key is not None:
            self._modified_nested_fields.setdefault(field_name, set()).add(nested_key)

    def _mark_project_modified(self, field_name: str, nested_key: str | None = None) -> None:
        self._modified_project_fields.add(field_name)
        if nested_key is not None:
            self._modified_project_nested_fields.setdefault(field_name, set()).add(nested_key)

    def _assert_project_trusted_for_write(self) -> None:
        if not self._project_trusted:
            raise RuntimeError("Project is not trusted; refusing to write project settings")

    def _record_error(self, scope: SettingsScope, error: Exception) -> None:
        self._errors.append(SettingsError(scope, error))

    def _persist_scoped_settings(
        self,
        scope: SettingsScope,
        snapshot_settings: Settings,
        modified_fields: set[str],
        modified_nested_fields: dict[str, set[str]],
    ) -> None:
        def update(current: str | None) -> str:
            current_file_settings = _migrate_settings(json.loads(current)) if current else {}
            merged_settings = dict(current_file_settings)
            for field_name in modified_fields:
                value = snapshot_settings.get(field_name)
                if field_name in modified_nested_fields and isinstance(value, dict):
                    nested_modified = modified_nested_fields[field_name]
                    base_nested = current_file_settings.get(field_name) or {}
                    if not isinstance(base_nested, dict):
                        base_nested = {}
                    merged_nested = dict(base_nested)
                    for nested_key in nested_modified:
                        merged_nested[nested_key] = value.get(nested_key)
                    merged_settings[field_name] = merged_nested
                else:
                    merged_settings[field_name] = value
            return json.dumps(merged_settings, indent=2)

        self._storage.with_lock(scope, update)

    def _save(self) -> None:
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

        if self._global_settings_load_error is not None:
            return

        snapshot_global_settings = copy.deepcopy(self._global_settings)
        modified_fields = set(self._modified_fields)
        modified_nested_fields = {k: set(v) for k, v in self._modified_nested_fields.items()}

        try:
            self._persist_scoped_settings("global", snapshot_global_settings, modified_fields, modified_nested_fields)
        except Exception as error:
            self._record_error("global", error)
        else:
            self._modified_fields.clear()
            self._modified_nested_fields.clear()

    def _save_project_settings(self, settings: Settings) -> None:
        self._assert_project_trusted_for_write()
        self._project_settings = copy.deepcopy(settings)
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

        if self._project_settings_load_error is not None:
            return

        snapshot_project_settings = copy.deepcopy(self._project_settings)
        modified_fields = set(self._modified_project_fields)
        modified_nested_fields = {k: set(v) for k, v in self._modified_project_nested_fields.items()}
        try:
            self._persist_scoped_settings("project", snapshot_project_settings, modified_fields, modified_nested_fields)
        except Exception as error:
            self._record_error("project", error)
        else:
            self._modified_project_fields.clear()
            self._modified_project_nested_fields.clear()

    def _update_project_settings(self, field_name: str, update: Callable[[Settings], None]) -> None:
        self._assert_project_trusted_for_write()
        project_settings = copy.deepcopy(self._project_settings)
        update(project_settings)
        self._mark_project_modified(field_name)
        self._save_project_settings(project_settings)

    async def flush(self) -> None:
        """No-op: writes are synchronous in this port (no async write queue)."""
        return None

    def drain_errors(self) -> list[SettingsError]:
        drained = list(self._errors)
        self._errors = []
        return drained

    # -- typed accessors ------------------------------------------------------

    def get_last_changelog_version(self) -> str | None:
        return self._settings.get("lastChangelogVersion")

    def set_last_changelog_version(self, version: str) -> None:
        self._global_settings["lastChangelogVersion"] = version
        self._mark_modified("lastChangelogVersion")
        self._save()

    def get_session_dir(self) -> str | None:
        session_dir = self._settings.get("sessionDir")
        return normalize_path(session_dir) if session_dir else session_dir

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def set_default_provider(self, provider: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._mark_modified("defaultProvider")
        self._save()

    def set_default_model(self, model_id: str) -> None:
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultModel")
        self._save()

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultProvider")
        self._mark_modified("defaultModel")
        self._save()

    def get_steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._settings.get("steeringMode") or "one-at-a-time"

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._global_settings["steeringMode"] = mode
        self._mark_modified("steeringMode")
        self._save()

    def get_follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._settings.get("followUpMode") or "one-at-a-time"

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._global_settings["followUpMode"] = mode
        self._mark_modified("followUpMode")
        self._save()

    def get_theme_setting(self) -> str | None:
        value = self._settings.get("theme")
        return value if isinstance(value, str) else None

    def get_theme(self) -> str | None:
        theme = self.get_theme_setting()
        return None if theme and "/" in theme else theme

    def set_theme(self, theme: str) -> None:
        self._global_settings["theme"] = theme
        self._mark_modified("theme")
        self._save()

    def get_default_tools(self) -> list[str] | None:
        """Initial tool allowlist, same format as the `--tools` CLI flag.

        Port of `getDefaultTools` (`settings-manager.ts:1192`). Returns a copy:
        the caller filters it in place when `--exclude-tools` is set, and
        mutating the stored settings would make the exclusion permanent.
        """
        tools = self._settings.get("defaultTools")
        return list(tools) if tools else None

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    def set_default_thinking_level(self, level: str) -> None:
        self._global_settings["defaultThinkingLevel"] = level
        self._mark_modified("defaultThinkingLevel")
        self._save()

    def get_transport(self) -> str:
        return self._settings.get("transport") or "auto"

    def set_transport(self, transport: str) -> None:
        self._global_settings["transport"] = transport
        self._mark_modified("transport")
        self._save()

    def get_compaction_enabled(self) -> bool:
        return (self._settings.get("compaction") or {}).get("enabled", True)

    def set_compaction_enabled(self, enabled: bool) -> None:
        self._global_settings.setdefault("compaction", {})
        self._global_settings["compaction"]["enabled"] = enabled
        self._mark_modified("compaction", "enabled")
        self._save()

    def get_compaction_reserve_tokens(self) -> int:
        return (self._settings.get("compaction") or {}).get("reserveTokens", 16384)

    def get_compaction_keep_recent_tokens(self) -> int:
        return (self._settings.get("compaction") or {}).get("keepRecentTokens", 20000)

    def get_compaction_settings(self) -> dict[str, Any]:
        return {
            "enabled": self.get_compaction_enabled(),
            "reserveTokens": self.get_compaction_reserve_tokens(),
            "keepRecentTokens": self.get_compaction_keep_recent_tokens(),
        }

    def get_branch_summary_settings(self) -> dict[str, Any]:
        branch_summary = self._settings.get("branchSummary") or {}
        return {
            "reserveTokens": branch_summary.get("reserveTokens", 16384),
            "skipPrompt": branch_summary.get("skipPrompt", False),
        }

    def get_branch_summary_skip_prompt(self) -> bool:
        return (self._settings.get("branchSummary") or {}).get("skipPrompt", False)

    def get_retry_enabled(self) -> bool:
        return (self._settings.get("retry") or {}).get("enabled", True)

    def set_retry_enabled(self, enabled: bool) -> None:
        self._global_settings.setdefault("retry", {})
        self._global_settings["retry"]["enabled"] = enabled
        self._mark_modified("retry", "enabled")
        self._save()

    def get_retry_settings(self) -> dict[str, Any]:
        retry = self._settings.get("retry") or {}
        return {
            "enabled": self.get_retry_enabled(),
            "maxRetries": retry.get("maxRetries", 3),
            "baseDelayMs": retry.get("baseDelayMs", 2000),
        }

    def get_http_idle_timeout_ms(self) -> int:
        timeout = _parse_timeout_setting(self._settings.get("httpIdleTimeoutMs"), "httpIdleTimeoutMs")
        return timeout if timeout is not None else DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def set_http_idle_timeout_ms(self, timeout_ms: float) -> None:
        if timeout_ms != timeout_ms or timeout_ms in (float("inf"), float("-inf")) or timeout_ms < 0:
            raise ValueError(f"Invalid httpIdleTimeoutMs setting: {timeout_ms}")
        self._global_settings["httpIdleTimeoutMs"] = int(timeout_ms)
        self._mark_modified("httpIdleTimeoutMs")
        self._save()

    def get_provider_retry_settings(self) -> dict[str, Any]:
        provider = (self._settings.get("retry") or {}).get("provider") or {}
        return {
            "timeoutMs": provider.get("timeoutMs"),
            "maxRetries": provider.get("maxRetries"),
            "maxRetryDelayMs": provider.get("maxRetryDelayMs", 60000),
        }

    def get_websocket_connect_timeout_ms(self) -> int | None:
        return _parse_timeout_setting(self._settings.get("websocketConnectTimeoutMs"), "websocketConnectTimeoutMs")

    def get_hide_thinking_block(self) -> bool:
        return self._settings.get("hideThinkingBlock", False)

    def get_show_cache_miss_notices(self) -> bool:
        return self._settings.get("showCacheMissNotices", False)

    def get_external_editor_command(self, *, env: dict[str, str] | None = None) -> str:
        env = env if env is not None else os.environ
        configured_editor = self._settings.get("externalEditor")
        if isinstance(configured_editor, str) and configured_editor.strip() != "":
            return configured_editor
        environment_editor = env.get("VISUAL") or env.get("EDITOR")
        if environment_editor:
            return environment_editor
        return "notepad" if os.name == "nt" else "nano"

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._global_settings["hideThinkingBlock"] = hide
        self._mark_modified("hideThinkingBlock")
        self._save()

    def set_show_cache_miss_notices(self, show: bool) -> None:
        self._global_settings["showCacheMissNotices"] = show
        self._mark_modified("showCacheMissNotices")
        self._save()

    def get_shell_path(self) -> str | None:
        shell_path = self._settings.get("shellPath")
        return normalize_path(shell_path) if shell_path else shell_path

    def set_shell_path(self, path: str | None) -> None:
        self._global_settings["shellPath"] = path
        self._mark_modified("shellPath")
        self._save()

    def get_version_check_repo(self) -> str | None:
        """GitHub ``owner/name`` to check for new releases.

        Not present upstream: TypeScript checks pi.dev's release API, while
        this port checks GitHub Releases (see `utils/version_check.py`).
        """
        value = self._settings.get("versionCheckRepo")
        return value if isinstance(value, str) and value.strip() else None

    def set_version_check_repo(self, repo: str | None) -> None:
        self._global_settings["versionCheckRepo"] = repo
        self._mark_modified("versionCheckRepo")
        self._save()

    def get_quiet_startup(self) -> bool:
        return self._settings.get("quietStartup", False)

    def set_quiet_startup(self, quiet: bool) -> None:
        self._global_settings["quietStartup"] = quiet
        self._mark_modified("quietStartup")
        self._save()

    def get_default_project_trust(self) -> Literal["ask", "always", "never"]:
        value = self._global_settings.get("defaultProjectTrust")
        return value if value in ("always", "never") else "ask"

    def set_default_project_trust(self, default_project_trust: Literal["ask", "always", "never"]) -> None:
        self._global_settings["defaultProjectTrust"] = default_project_trust
        self._mark_modified("defaultProjectTrust")
        self._save()

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.get("shellCommandPrefix")

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        self._global_settings["shellCommandPrefix"] = prefix
        self._mark_modified("shellCommandPrefix")
        self._save()

    def get_npm_command(self) -> list[str] | None:
        command = self._settings.get("npmCommand")
        return list(command) if command else None

    def set_npm_command(self, command: list[str] | None) -> None:
        self._global_settings["npmCommand"] = list(command) if command else None
        self._mark_modified("npmCommand")
        self._save()

    def get_collapse_changelog(self) -> bool:
        return self._settings.get("collapseChangelog", False)

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._global_settings["collapseChangelog"] = collapse
        self._mark_modified("collapseChangelog")
        self._save()

    def get_enable_install_telemetry(self) -> bool:
        return self._settings.get("enableInstallTelemetry", True)

    def set_enable_install_telemetry(self, enabled: bool) -> None:
        self._global_settings["enableInstallTelemetry"] = enabled
        self._mark_modified("enableInstallTelemetry")
        self._save()

    def get_enable_analytics(self) -> bool:
        return self._settings.get("enableAnalytics", False)

    def get_tracking_id(self) -> str | None:
        return self._settings.get("trackingId")

    def set_enable_analytics(self, enabled: bool) -> None:
        """Set the analytics opt-in preference; generates a tracking id on first opt-in."""
        self._global_settings["enableAnalytics"] = enabled
        self._mark_modified("enableAnalytics")
        if enabled and not self._global_settings.get("trackingId"):
            self._global_settings["trackingId"] = str(uuid.uuid4())
            self._mark_modified("trackingId")
        self._save()

    def get_packages(self) -> list[Any]:
        return list(self._settings.get("packages") or [])

    def set_packages(self, packages: list[Any]) -> None:
        self._global_settings["packages"] = packages
        self._mark_modified("packages")
        self._save()

    def set_project_packages(self, packages: list[Any]) -> None:
        def update(settings: Settings) -> None:
            settings["packages"] = packages

        self._update_project_settings("packages", update)

    def get_extension_paths(self) -> list[str]:
        return list(self._settings.get("extensions") or [])

    def set_extension_paths(self, paths: list[str]) -> None:
        self._global_settings["extensions"] = paths
        self._mark_modified("extensions")
        self._save()

    def set_project_extension_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["extensions"] = paths

        self._update_project_settings("extensions", update)

    def get_skill_paths(self) -> list[str]:
        return list(self._settings.get("skills") or [])

    def set_skill_paths(self, paths: list[str]) -> None:
        self._global_settings["skills"] = paths
        self._mark_modified("skills")
        self._save()

    def set_project_skill_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["skills"] = paths

        self._update_project_settings("skills", update)

    def get_prompt_template_paths(self) -> list[str]:
        return list(self._settings.get("prompts") or [])

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._global_settings["prompts"] = paths
        self._mark_modified("prompts")
        self._save()

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["prompts"] = paths

        self._update_project_settings("prompts", update)

    def get_theme_paths(self) -> list[str]:
        return list(self._settings.get("themes") or [])

    def set_theme_paths(self, paths: list[str]) -> None:
        self._global_settings["themes"] = paths
        self._mark_modified("themes")
        self._save()

    def set_project_theme_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["themes"] = paths

        self._update_project_settings("themes", update)

    def get_enable_skill_commands(self) -> bool:
        return self._settings.get("enableSkillCommands", True)

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._global_settings["enableSkillCommands"] = enabled
        self._mark_modified("enableSkillCommands")
        self._save()

    def get_thinking_budgets(self) -> dict[str, Any] | None:
        return self._settings.get("thinkingBudgets")

    def get_show_images(self) -> bool:
        return (self._settings.get("terminal") or {}).get("showImages", True)

    def set_show_images(self, show: bool) -> None:
        self._global_settings.setdefault("terminal", {})
        self._global_settings["terminal"]["showImages"] = show
        self._mark_modified("terminal", "showImages")
        self._save()

    def get_image_width_cells(self) -> int:
        width = (self._settings.get("terminal") or {}).get("imageWidthCells")
        if not isinstance(width, (int, float)) or isinstance(width, bool) or width != width:
            return 60
        return max(1, int(width))

    def set_image_width_cells(self, width: float) -> None:
        self._global_settings.setdefault("terminal", {})
        self._global_settings["terminal"]["imageWidthCells"] = max(1, int(width))
        self._mark_modified("terminal", "imageWidthCells")
        self._save()

    def get_clear_on_shrink(self, *, env: dict[str, str] | None = None) -> bool:
        terminal = self._settings.get("terminal") or {}
        if terminal.get("clearOnShrink") is not None:
            return terminal["clearOnShrink"]
        env = env if env is not None else os.environ
        return env.get("PI_CLEAR_ON_SHRINK") == "1"

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self._global_settings.setdefault("terminal", {})
        self._global_settings["terminal"]["clearOnShrink"] = enabled
        self._mark_modified("terminal", "clearOnShrink")
        self._save()

    def get_show_terminal_progress(self) -> bool:
        return (self._settings.get("terminal") or {}).get("showTerminalProgress", False)

    def set_show_terminal_progress(self, enabled: bool) -> None:
        self._global_settings.setdefault("terminal", {})
        self._global_settings["terminal"]["showTerminalProgress"] = enabled
        self._mark_modified("terminal", "showTerminalProgress")
        self._save()

    def get_tui_mode(self) -> Literal["regular", "fullscreen"]:
        return "fullscreen" if self._settings.get("tuiMode") == "fullscreen" else "regular"

    def set_tui_mode(self, mode: Literal["regular", "fullscreen"]) -> None:
        self._global_settings["tuiMode"] = mode
        self._mark_modified("tuiMode")
        self._save()

    def get_fullscreen_exit_output(self) -> Literal["transcript", "resume-hint"]:
        return "resume-hint" if self._settings.get("fullscreenExitOutput") == "resume-hint" else "transcript"

    def set_fullscreen_exit_output(self, output: Literal["transcript", "resume-hint"]) -> None:
        self._global_settings["fullscreenExitOutput"] = output
        self._mark_modified("fullscreenExitOutput")
        self._save()

    def get_fullscreen_scrollbar(self) -> Literal["auto", "always", "hidden"]:
        mode = self._settings.get("fullscreenScrollbar")
        return mode if mode in ("always", "hidden") else "auto"

    def set_fullscreen_scrollbar(self, mode: Literal["auto", "always", "hidden"]) -> None:
        self._global_settings["fullscreenScrollbar"] = mode
        self._mark_modified("fullscreenScrollbar")
        self._save()

    def get_image_auto_resize(self) -> bool:
        return (self._settings.get("images") or {}).get("autoResize", True)

    def set_image_auto_resize(self, enabled: bool) -> None:
        self._global_settings.setdefault("images", {})
        self._global_settings["images"]["autoResize"] = enabled
        self._mark_modified("images", "autoResize")
        self._save()

    def get_block_images(self) -> bool:
        return (self._settings.get("images") or {}).get("blockImages", False)

    def set_block_images(self, blocked: bool) -> None:
        self._global_settings.setdefault("images", {})
        self._global_settings["images"]["blockImages"] = blocked
        self._mark_modified("images", "blockImages")
        self._save()

    def get_enabled_models(self) -> list[str] | None:
        return self._settings.get("enabledModels")

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._global_settings["enabledModels"] = patterns
        self._mark_modified("enabledModels")
        self._save()

    def get_double_escape_action(self) -> Literal["fork", "tree", "none"]:
        return self._settings.get("doubleEscapeAction") or "tree"

    def set_double_escape_action(self, action: Literal["fork", "tree", "none"]) -> None:
        self._global_settings["doubleEscapeAction"] = action
        self._mark_modified("doubleEscapeAction")
        self._save()

    def get_tree_filter_mode(self) -> Literal["default", "no-tools", "user-only", "labeled-only", "all"]:
        mode = self._settings.get("treeFilterMode")
        valid = ("default", "no-tools", "user-only", "labeled-only", "all")
        return mode if mode in valid else "default"

    def set_tree_filter_mode(self, mode: Literal["default", "no-tools", "user-only", "labeled-only", "all"]) -> None:
        self._global_settings["treeFilterMode"] = mode
        self._mark_modified("treeFilterMode")
        self._save()

    def get_show_hardware_cursor(self, *, env: dict[str, str] | None = None) -> bool:
        value = self._settings.get("showHardwareCursor")
        if value is not None:
            return value
        env = env if env is not None else os.environ
        return env.get("PI_HARDWARE_CURSOR") == "1"

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._global_settings["showHardwareCursor"] = enabled
        self._mark_modified("showHardwareCursor")
        self._save()

    def get_editor_padding_x(self) -> int:
        return self._settings.get("editorPaddingX", 0)

    def set_editor_padding_x(self, padding: float) -> None:
        self._global_settings["editorPaddingX"] = max(0, min(3, int(padding)))
        self._mark_modified("editorPaddingX")
        self._save()

    def get_output_pad(self) -> Literal[0, 1]:
        return 0 if self._settings.get("outputPad") == 0 else 1

    def set_output_pad(self, padding: Literal[0, 1]) -> None:
        self._global_settings["outputPad"] = padding
        self._mark_modified("outputPad")
        self._save()

    def get_autocomplete_max_visible(self) -> int:
        return self._settings.get("autocompleteMaxVisible", 5)

    def set_autocomplete_max_visible(self, max_visible: float) -> None:
        self._global_settings["autocompleteMaxVisible"] = max(3, min(20, int(max_visible)))
        self._mark_modified("autocompleteMaxVisible")
        self._save()

    def get_code_block_indent(self) -> str:
        return (self._settings.get("markdown") or {}).get("codeBlockIndent", "  ")

    def get_mermaid_rendering_mode(self) -> Literal["off", "final", "streaming"]:
        mode = (self._settings.get("markdown") or {}).get("mermaid")
        return mode if mode in ("off", "final") else "streaming"

    def set_mermaid_rendering_mode(self, mode: Literal["off", "final", "streaming"]) -> None:
        self._global_settings.setdefault("markdown", {})
        self._global_settings["markdown"]["mermaid"] = mode
        self._mark_modified("markdown", "mermaid")
        self._save()

    def get_warnings(self) -> dict[str, Any]:
        return dict(self._settings.get("warnings") or {})

    def set_warnings(self, warnings: dict[str, Any]) -> None:
        self._global_settings["warnings"] = dict(warnings)
        self._mark_modified("warnings")
        self._save()


__all__ = [
    "DEFAULT_HTTP_IDLE_TIMEOUT_MS",
    "FileSettingsStorage",
    "InMemorySettingsStorage",
    "Settings",
    "SettingsError",
    "SettingsManager",
    "SettingsManagerCreateOptions",
    "SettingsScope",
    "SettingsStorage",
    "deep_merge_objects",
    "deep_merge_settings",
]
