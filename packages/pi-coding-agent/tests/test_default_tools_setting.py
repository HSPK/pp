"""The `defaultTools` setting seeds the initial built-in tool selection.

Port of upstream 4d9aa837c plus its follow-up 541045ae0. The follow-up matters:
the first revision also narrowed `allowedToolNames`, which disabled extension
tools the user had never listed. Only the *initial built-in selection* is
seeded; the allowlist is left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from pi_coding_agent.core.settings_manager import SettingsManager


def _settings_manager(tmp_path: Path, settings: dict) -> SettingsManager:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text(json.dumps(settings))
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    return SettingsManager.create(str(cwd), str(agent_dir))


def test_returns_none_when_unset(tmp_path: Path):
    assert _settings_manager(tmp_path, {}).get_default_tools() is None


def test_returns_the_configured_list(tmp_path: Path):
    manager = _settings_manager(tmp_path, {"defaultTools": ["read", "bash"]})

    assert manager.get_default_tools() == ["read", "bash"]


def test_an_empty_list_reads_as_unset(tmp_path: Path):
    """Matches `tools ? [...tools] : undefined` upstream.

    An empty allowlist would otherwise mean "no tools at all", which is what
    `--no-tools` is for.
    """
    assert _settings_manager(tmp_path, {"defaultTools": []}).get_default_tools() is None


def test_the_caller_cannot_mutate_the_stored_setting(tmp_path: Path):
    """`create_agent_session` filters this list in place for `--exclude-tools`."""
    manager = _settings_manager(tmp_path, {"defaultTools": ["read", "bash"]})

    manager.get_default_tools().remove("bash")

    assert manager.get_default_tools() == ["read", "bash"]
