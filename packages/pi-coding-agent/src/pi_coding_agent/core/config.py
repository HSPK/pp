"""App identity and config-path resolution.

Python port of the portable subset of `packages/coding-agent/src/config.ts`:
app name/version constants and the `~/.pi/agent/*` path helpers
(`get_agent_dir`, `get_models_path`, `get_auth_path`, `get_settings_path`,
`get_tools_dir`, `get_bin_dir`, `get_prompts_dir`, `get_sessions_dir`,
`get_debug_log_path`), plus `expand_tilde_path` and `get_share_viewer_url`.

**Not ported (no Python equivalent):** the TypeScript file's install-method
detection (`detectInstallMethod`), self-update command construction
(`getSelfUpdateCommand`, `getSelfUpdateUnavailableInstruction`,
`getUpdateInstruction`), and Bun/npm/pnpm/yarn packaging path helpers
(`getPackageDir`, `getThemesDir`, `getExportTemplateDir`, etc.). Those
inspect `node_modules`/global npm-pnpm-yarn-bun install layouts and Bun
compiled-binary markers (`import.meta.url` containing `$bunfs`) that have no
meaning for a `uv`/`pip`-installed Python package; there is no equivalent
"self-update via the package manager that installed me" story to port.

Every path helper below accepts optional ``env``/``home_dir`` overrides so
tests can point resolution at a ``tmp_path`` sandbox instead of the real
``$HOME`` and process environment; callers that omit them get the real
values, matching the TypeScript behavior of reading `process.env`/`homedir()`
directly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pi_coding_agent.utils.paths import PathInputOptions, normalize_path

CONFIG_DIR_NAME = ".pi"
APP_NAME = "pi"
APP_TITLE = "pi"
PACKAGE_NAME = "pi-coding-agent"
VERSION = "0.0.1"

# e.g. PI_CODING_AGENT_DIR / PI_CODING_AGENT_SESSION_DIR
ENV_AGENT_DIR = f"{APP_NAME.upper()}_CODING_AGENT_DIR"
ENV_SESSION_DIR = f"{APP_NAME.upper()}_CODING_AGENT_SESSION_DIR"

_DEFAULT_SHARE_VIEWER_URL = "https://pi.dev/session/"


def _env_get(env: Mapping[str, str] | None, key: str) -> str | None:
    return (env if env is not None else os.environ).get(key)


def expand_tilde_path(path: str, *, home_dir: str | None = None) -> str:
    """Expand a leading ``~`` and normalize unicode spaces / `file://` URLs."""
    return normalize_path(path, PathInputOptions(home_dir=home_dir))


def get_share_viewer_url(gist_id: str, *, env: Mapping[str, str] | None = None) -> str:
    """Get the share viewer URL for a gist ID."""
    base_url = _env_get(env, "PI_SHARE_VIEWER_URL") or _DEFAULT_SHARE_VIEWER_URL
    return f"{base_url}#{gist_id}"


# --------------------------------------------------------------------------
# User config paths (~/.pi/agent/*)
# --------------------------------------------------------------------------


def get_agent_dir(*, env: Mapping[str, str] | None = None, home_dir: str | None = None) -> str:
    """Get the agent config directory (e.g. ``~/.pi/agent/``)."""
    env_dir = _env_get(env, ENV_AGENT_DIR)
    if env_dir:
        return expand_tilde_path(env_dir, home_dir=home_dir)
    home = home_dir if home_dir is not None else str(Path.home())
    return str(Path(home) / CONFIG_DIR_NAME / "agent")


def get_custom_themes_dir(agent_dir: str | None = None) -> str:
    """Get path to the user's custom themes directory."""
    return str(Path(agent_dir or get_agent_dir()) / "themes")


def get_models_path(agent_dir: str | None = None) -> str:
    """Get path to ``models.json``."""
    return str(Path(agent_dir or get_agent_dir()) / "models.json")


def get_auth_path(agent_dir: str | None = None) -> str:
    """Get path to ``auth.json``."""
    return str(Path(agent_dir or get_agent_dir()) / "auth.json")


def get_settings_path(agent_dir: str | None = None) -> str:
    """Get path to ``settings.json``."""
    return str(Path(agent_dir or get_agent_dir()) / "settings.json")


def get_tools_dir(agent_dir: str | None = None) -> str:
    """Get path to the tools directory."""
    return str(Path(agent_dir or get_agent_dir()) / "tools")


def get_bin_dir(agent_dir: str | None = None) -> str:
    """Get path to the managed binaries directory (fd, rg)."""
    return str(Path(agent_dir or get_agent_dir()) / "bin")


def get_prompts_dir(agent_dir: str | None = None) -> str:
    """Get path to the prompt templates directory."""
    return str(Path(agent_dir or get_agent_dir()) / "prompts")


def get_sessions_dir(agent_dir: str | None = None) -> str:
    """Get path to the sessions directory."""
    return str(Path(agent_dir or get_agent_dir()) / "sessions")


def get_debug_log_path(agent_dir: str | None = None) -> str:
    """Get path to the debug log file."""
    return str(Path(agent_dir or get_agent_dir()) / f"{APP_NAME}-debug.log")


def get_changelog_path() -> str:
    """Get path to the bundled CHANGELOG.md."""
    return str(Path(get_package_dir()) / "CHANGELOG.md")


__all__ = [
    "APP_NAME",
    "APP_TITLE",
    "CONFIG_DIR_NAME",
    "ENV_AGENT_DIR",
    "ENV_PACKAGE_DIR",
    "ENV_SESSION_DIR",
    "PACKAGE_NAME",
    "VERSION",
    "expand_tilde_path",
    "get_agent_dir",
    "get_auth_path",
    "get_bin_dir",
    "get_changelog_path",
    "get_custom_themes_dir",
    "get_debug_log_path",
    "get_docs_path",
    "get_examples_path",
    "get_models_path",
    "get_package_dir",
    "get_prompts_dir",
    "get_readme_path",
    "get_sessions_dir",
    "get_settings_path",
    "get_share_viewer_url",
    "get_tools_dir",
]


# --------------------------------------------------------------------------
# Self-documentation paths
#
# Port of `config.ts`'s `getPackageDir`/`getReadmePath`/`getDocsPath`/
# `getExamplesPath`. These exist for one reason: `system_prompt.py` embeds
# them so the model can `read` pi's own documentation when asked about pi
# itself. There is no retrieval index -- the prompt names real files and the
# model opens them with the ordinary read tool, so these paths must resolve
# on disk in both a source checkout and an installed wheel.
#
# TypeScript walks up from `__dirname` until it finds a `package.json`. The
# Python equivalent walks up from this module until it finds the directory
# holding both `docs/` and the package source, which covers the source
# checkout; in a wheel those same directories are installed *inside* the
# package, so the package directory itself is the answer. `PI_PACKAGE_DIR`
# overrides both, matching TypeScript.
# --------------------------------------------------------------------------

ENV_PACKAGE_DIR = "PI_PACKAGE_DIR"


def get_package_dir(*, env: Mapping[str, str] | None = None) -> str:
    """Directory holding this package's `README.md`, `docs/` and `examples/`."""
    env_dir = _env_get(env, ENV_PACKAGE_DIR)
    if env_dir:
        return normalize_path(env_dir)

    package_root = Path(__file__).resolve().parent.parent

    # Installed wheel: `docs/` ships inside the package.
    if (package_root / "docs").is_dir():
        return str(package_root)

    # Source checkout: `src/pi_coding_agent/core/config.py` -> `pi-coding-agent/`.
    for candidate in package_root.parents:
        if (candidate / "docs").is_dir() and (candidate / "pyproject.toml").is_file():
            return str(candidate)

    return str(package_root)


def get_readme_path(*, env: Mapping[str, str] | None = None) -> str:
    """Path to the shipped `README.md`."""
    return str(Path(get_package_dir(env=env)) / "README.md")


def get_docs_path(*, env: Mapping[str, str] | None = None) -> str:
    """Path to the shipped `docs/` directory."""
    return str(Path(get_package_dir(env=env)) / "docs")


def get_examples_path(*, env: Mapping[str, str] | None = None) -> str:
    """Path to the shipped `examples/` directory."""
    return str(Path(get_package_dir(env=env)) / "examples")
