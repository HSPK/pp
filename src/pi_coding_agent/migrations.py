"""One-time upgrades that run once on startup.

Ported from ``packages/coding-agent/src/migrations.ts``.

Older versions of pi stored things in places later versions no longer read:
OAuth tokens in ``oauth.json``, API keys inside ``settings.json``, sessions
directly in the agent directory, prompt templates under ``commands/``, managed
``fd``/``rg`` binaries under ``tools/``. Each migration moves the old layout to
the current one so an upgrading user does not silently lose credentials,
history or config.

Every step is best-effort: a migration that cannot run must never stop the CLI
from starting, so failures are swallowed and the old files are left alone.
`run_migrations` also returns warnings about directories (``hooks/``,
``tools/`` with custom entries) that cannot be migrated automatically because
their contents need human review.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pi_coding_agent.core.app_keybindings import migrate_keybindings_config
from pi_coding_agent.core.config import CONFIG_DIR_NAME, get_agent_dir, get_bin_dir

MIGRATION_GUIDE_URL = (
    "https://github.com/earendil-works/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md#extensions-migration"
)
EXTENSIONS_DOC_URL = "https://github.com/earendil-works/pi-mono/blob/main/packages/coding-agent/docs/extensions.md"

_MANAGED_BINARIES = ("fd", "rg", "fd.exe", "rg.exe")
_LEADING_SEPARATOR_RE = re.compile(r"^[/\\]")
_PATH_SEPARATOR_RE = re.compile(r"[/\\:]")


@dataclass
class MigrationResult:
    migrated_auth_providers: list[str] = field(default_factory=list)
    deprecation_warnings: list[str] = field(default_factory=list)


def migrate_auth_to_auth_json(agent_dir: str | None = None) -> list[str]:
    """Fold legacy ``oauth.json`` and ``settings.json`` API keys into ``auth.json``.

    Returns the provider names that were moved. Does nothing when
    ``auth.json`` already exists, so it is safe to run on every startup.
    """
    base = Path(agent_dir or get_agent_dir())
    auth_path = base / "auth.json"
    oauth_path = base / "oauth.json"
    settings_path = base / "settings.json"

    if auth_path.exists():
        return []

    migrated: dict[str, object] = {}
    providers: list[str] = []

    if oauth_path.exists():
        try:
            oauth = json.loads(oauth_path.read_text(encoding="utf-8"))
            for provider, credential in oauth.items():
                if isinstance(credential, dict):
                    migrated[provider] = {"type": "oauth", **credential}
                    providers.append(provider)
            oauth_path.rename(oauth_path.with_name(f"{oauth_path.name}.migrated"))
        except Exception:
            pass

    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            api_keys = settings.get("apiKeys")
            if isinstance(api_keys, dict):
                for provider, key in api_keys.items():
                    if provider not in migrated and isinstance(key, str):
                        migrated[provider] = {"type": "api_key", "key": key}
                        providers.append(provider)
                del settings["apiKeys"]
                settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    if migrated:
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
        os.chmod(auth_path, 0o600)

    return providers


def _encode_session_dir_name(cwd: str) -> str:
    """The session directory name for `cwd` (same encoding as `session_manager`)."""
    stripped = _LEADING_SEPARATOR_RE.sub("", cwd)
    return f"--{_PATH_SEPARATOR_RE.sub('-', stripped)}--"


def migrate_sessions_from_agent_root(agent_dir: str | None = None) -> None:
    """Move stray ``*.jsonl`` sessions into their per-project session directory.

    v0.30.0 wrote sessions to ``~/.pi/agent/`` instead of
    ``~/.pi/agent/sessions/<encoded-cwd>/``, where later versions cannot find
    them. Each file's header records the cwd it belongs to, so it can be filed
    correctly after the fact.
    """
    base = Path(agent_dir or get_agent_dir())
    try:
        files = [entry for entry in base.iterdir() if entry.is_file() and entry.suffix == ".jsonl"]
    except OSError:
        return

    for file in files:
        try:
            first_line = file.read_text(encoding="utf-8").split("\n")[0]
            if not first_line.strip():
                continue
            header = json.loads(first_line)
            if header.get("type") != "session" or not header.get("cwd"):
                continue

            correct_dir = base / "sessions" / _encode_session_dir_name(header["cwd"])
            correct_dir.mkdir(parents=True, exist_ok=True)
            target = correct_dir / file.name
            if target.exists():
                continue
            file.rename(target)
        except Exception:
            continue


def migrate_commands_to_prompts(base_dir: str, label: str) -> bool:
    """Rename ``commands/`` to ``prompts/``; works for symlinks too."""
    commands_dir = Path(base_dir) / "commands"
    prompts_dir = Path(base_dir) / "prompts"

    if (commands_dir.exists() or commands_dir.is_symlink()) and not prompts_dir.exists():
        try:
            commands_dir.rename(prompts_dir)
            print(f"Migrated {label} commands/ -> prompts/")
            return True
        except OSError as error:
            print(f"Warning: Could not migrate {label} commands/ to prompts/: {error}")
    return False


def migrate_keybindings_config_file(agent_dir: str | None = None) -> None:
    config_path = Path(agent_dir or get_agent_dir()) / "keybindings.json"
    if not config_path.exists():
        return
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            return
        config, migrated = migrate_keybindings_config(parsed)
        if not migrated:
            return
        config_path.write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf-8")
    except Exception:
        pass


def migrate_tools_to_bin(agent_dir: str | None = None) -> None:
    """Move managed ``fd``/``rg`` binaries from ``tools/`` to ``bin/``."""
    base = Path(agent_dir or get_agent_dir())
    tools_dir = base / "tools"
    bin_dir = Path(get_bin_dir(str(base)))

    if not tools_dir.exists():
        return

    moved_any = False
    for binary in _MANAGED_BINARIES:
        old_path = tools_dir / binary
        new_path = bin_dir / binary
        if not old_path.exists():
            continue
        if new_path.exists():
            # Already migrated; the stale copy is just taking up space.
            with contextlib.suppress(OSError):
                old_path.unlink()
            continue
        try:
            bin_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            moved_any = True
        except OSError:
            pass

    if moved_any:
        print("Migrated managed binaries tools/ -> bin/")


def check_deprecated_extension_dirs(base_dir: str, label: str) -> list[str]:
    """Warn about ``hooks/`` and custom ``tools/`` that need manual migration.

    ``tools/`` also holds auto-extracted ``fd``/``rg`` binaries, so it only
    warrants a warning when it contains something else.
    """
    base = Path(base_dir)
    warnings: list[str] = []

    if (base / "hooks").exists():
        warnings.append(f"{label} hooks/ directory found. Hooks have been renamed to extensions.")

    tools_dir = base / "tools"
    if tools_dir.exists():
        try:
            custom = [
                entry.name
                for entry in tools_dir.iterdir()
                if entry.name.lower() not in _MANAGED_BINARIES and not entry.name.startswith(".")
            ]
            if custom:
                warnings.append(
                    f"{label} tools/ directory contains custom tools. Custom tools have been merged into extensions."
                )
        except OSError:
            pass

    return warnings


def migrate_extension_system(cwd: str, agent_dir: str | None = None) -> list[str]:
    base = agent_dir or get_agent_dir()
    project_dir = os.path.join(cwd, CONFIG_DIR_NAME)

    migrate_commands_to_prompts(base, "Global")
    migrate_commands_to_prompts(project_dir, "Project")

    return [
        *check_deprecated_extension_dirs(base, "Global"),
        *check_deprecated_extension_dirs(project_dir, "Project"),
    ]


def format_deprecation_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    lines = [f"Warning: {warning}" for warning in warnings]
    lines.append("\nMove your extensions to the extensions/ directory.")
    lines.append(f"Migration guide: {MIGRATION_GUIDE_URL}")
    lines.append(f"Documentation: {EXTENSIONS_DOC_URL}")
    return "\n".join(lines)


def show_deprecation_warnings(warnings: list[str]) -> None:
    """Print deprecation warnings.

    The TS version also waits for a keypress. This port only prints: the
    startup path may be non-interactive (print/json modes redirect stdout and
    may have no TTY at all), and blocking there would hang a piped invocation.
    """
    if not warnings:
        return
    print(format_deprecation_warnings(warnings))


def run_migrations(cwd: str, agent_dir: str | None = None) -> MigrationResult:
    """Run every startup migration. Called once, before any session is built."""
    base = agent_dir or get_agent_dir()
    migrated_auth_providers = migrate_auth_to_auth_json(base)
    migrate_sessions_from_agent_root(base)
    migrate_tools_to_bin(base)
    migrate_keybindings_config_file(base)
    deprecation_warnings = migrate_extension_system(cwd, base)
    return MigrationResult(
        migrated_auth_providers=migrated_auth_providers,
        deprecation_warnings=deprecation_warnings,
    )


__all__ = [
    "EXTENSIONS_DOC_URL",
    "MIGRATION_GUIDE_URL",
    "MigrationResult",
    "check_deprecated_extension_dirs",
    "format_deprecation_warnings",
    "migrate_auth_to_auth_json",
    "migrate_commands_to_prompts",
    "migrate_extension_system",
    "migrate_keybindings_config_file",
    "migrate_sessions_from_agent_root",
    "migrate_tools_to_bin",
    "run_migrations",
    "show_deprecation_warnings",
]
