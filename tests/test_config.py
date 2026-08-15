"""Tests for `core/config.py`.

Ported subset: `config.ts` is almost entirely install-method detection and
self-update command construction (see `config.py`'s module docstring for why
that isn't ported), so `config.test.ts` has no applicable cases here -- this
file instead directly exercises the ported path helpers and precedence
rules (env override vs. `home_dir` default), always pointed at `tmp_path`.
"""

from __future__ import annotations

import os

import pytest

from pi_coding_agent.core.config import (
    APP_NAME,
    CONFIG_DIR_NAME,
    ENV_AGENT_DIR,
    ENV_SESSION_DIR,
    expand_tilde_path,
    get_agent_dir,
    get_auth_path,
    get_bin_dir,
    get_custom_themes_dir,
    get_debug_log_path,
    get_models_path,
    get_prompts_dir,
    get_sessions_dir,
    get_settings_path,
    get_share_viewer_url,
    get_tools_dir,
)


def test_get_agent_dir_defaults_to_home_pi_agent(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = get_agent_dir(env={}, home_dir=str(home))
    assert result == str(home / CONFIG_DIR_NAME / "agent")


def test_get_agent_dir_env_override_wins_over_home_dir(tmp_path):
    home = tmp_path / "home"
    custom = tmp_path / "custom-agent-dir"
    env = {ENV_AGENT_DIR: str(custom)}
    result = get_agent_dir(env=env, home_dir=str(home))
    assert result == str(custom)


def test_get_agent_dir_env_expands_tilde(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {ENV_AGENT_DIR: "~/custom-pi"}
    result = get_agent_dir(env=env, home_dir=str(home))
    assert result == str(home / "custom-pi")


def test_env_var_names_use_app_name():
    assert f"{APP_NAME.upper()}_CODING_AGENT_DIR" == ENV_AGENT_DIR
    assert f"{APP_NAME.upper()}_CODING_AGENT_SESSION_DIR" == ENV_SESSION_DIR


def test_expand_tilde_path_expands_against_given_home(tmp_path):
    result = expand_tilde_path("~/foo/bar", home_dir=str(tmp_path))
    assert result == str(tmp_path / "foo" / "bar")


def test_expand_tilde_path_leaves_absolute_path_unchanged(tmp_path):
    absolute = str(tmp_path / "already" / "absolute")
    assert expand_tilde_path(absolute, home_dir=str(tmp_path)) == absolute


def test_get_share_viewer_url_default_base(monkeypatch):
    monkeypatch.delenv("PI_SHARE_VIEWER_URL", raising=False)
    assert get_share_viewer_url("abc123", env={}) == "https://pi.dev/session/#abc123"


def test_get_share_viewer_url_custom_base_from_env():
    env = {"PI_SHARE_VIEWER_URL": "https://example.test/share/"}
    assert get_share_viewer_url("xyz", env=env) == "https://example.test/share/#xyz"


def test_derived_paths_default_to_agent_dir(tmp_path):
    agent_dir = str(tmp_path / "agent")
    assert get_models_path(agent_dir) == os.path.join(agent_dir, "models.json")
    assert get_auth_path(agent_dir) == os.path.join(agent_dir, "auth.json")
    assert get_settings_path(agent_dir) == os.path.join(agent_dir, "settings.json")
    assert get_tools_dir(agent_dir) == os.path.join(agent_dir, "tools")
    assert get_bin_dir(agent_dir) == os.path.join(agent_dir, "bin")
    assert get_prompts_dir(agent_dir) == os.path.join(agent_dir, "prompts")
    assert get_sessions_dir(agent_dir) == os.path.join(agent_dir, "sessions")
    assert get_custom_themes_dir(agent_dir) == os.path.join(agent_dir, "themes")
    assert get_debug_log_path(agent_dir) == os.path.join(agent_dir, f"{APP_NAME}-debug.log")


def test_derived_paths_fall_back_to_real_get_agent_dir_when_omitted(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_AGENT_DIR, raising=False)
    monkeypatch.setattr("pi_coding_agent.core.config.Path.home", lambda: tmp_path)
    expected_agent_dir = str(tmp_path / CONFIG_DIR_NAME / "agent")
    assert get_models_path() == os.path.join(expected_agent_dir, "models.json")


# --------------------------------------------------------------------------
# Deliberately-omitted surface
# --------------------------------------------------------------------------


@pytest.mark.skip(
    reason="All 15 `detectInstallMethod` cases in config.test.ts (`detects pnpm from Windows "
    ".pnpm install paths`, `does not self-update unknown wrapper installs`, `self-updates npm "
    "installs from custom prefixes`, `self-updates exact npm versions without uninstalling the "
    "current package`, `self-updates renamed packages from the current install prefix`, "
    "`self-update respects configured npmCommand`, `self-update treats empty npmCommand as "
    "unset`, `quotes npm self-update display paths`, `does not infer Windows npm custom prefixes "
    "from package paths`, `self-updates bun global installs from bun pm bin`, `self-updates "
    "renamed pnpm global installs by removing the old package first`, `self-updates pnpm v11 "
    "global installs resolved through the store`, `self-updates renamed yarn global installs by "
    "removing the old package first`, `self-updates renamed bun global installs by removing the "
    "old package first`, `does not self-update when npm install path is not writable`) inspect "
    "npm/pnpm/yarn/bun global install layouts and build `npm install -g` self-update commands. "
    "`config.py`'s module docstring records dropping that whole surface: a `uv`/`pip`-installed "
    "Python package has no equivalent install-method detection or self-update story. The path "
    "helpers `config.ts` does share with `config.py` are covered above."
)
def test_detect_install_method_and_self_update_commands() -> None:
    pass
