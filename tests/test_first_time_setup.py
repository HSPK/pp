"""Python port of `packages/coding-agent/test/first-time-setup.test.ts`.

Two things: the four conditions that gate the one-time setup wizard, and the
analytics opt-in/tracking-id behaviour of `SettingsManager`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pi_coding_agent.cli import startup_ui
from pi_coding_agent.core.settings_manager import SettingsManager


@pytest.fixture(autouse=True)
def _experimental_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.delenv(startup_ui.ENV_AGENT_DIR, raising=False)


def test_returns_true_when_experimental_default_agent_dir_and_no_settings(tmp_path: Path) -> None:
    assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is True


def test_returns_false_when_experimental_features_are_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_EXPERIMENTAL", raising=False)

    assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is False


def test_returns_false_when_a_custom_agent_dir_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(startup_ui.ENV_AGENT_DIR, str(tmp_path))

    assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is False


def test_returns_false_when_settings_json_already_exists(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    assert startup_ui.should_run_first_time_setup(str(settings_path)) is False


def test_analytics_defaults_to_disabled_with_no_tracking_identifier() -> None:
    manager = SettingsManager.in_memory()

    assert manager.get_enable_analytics() is False
    assert manager.get_tracking_id() is None


def test_generates_a_tracking_identifier_on_opt_in() -> None:
    manager = SettingsManager.in_memory()

    manager.set_enable_analytics(True)

    assert manager.get_enable_analytics() is True
    tracking_id = manager.get_tracking_id()
    assert tracking_id is not None
    assert re.fullmatch(r"[0-9a-f-]{36}", tracking_id)


def test_does_not_generate_a_tracking_identifier_on_opt_out() -> None:
    manager = SettingsManager.in_memory()

    manager.set_enable_analytics(False)

    assert manager.get_enable_analytics() is False
    assert manager.get_tracking_id() is None


def test_keeps_the_tracking_identifier_when_toggling_analytics() -> None:
    manager = SettingsManager.in_memory()

    manager.set_enable_analytics(True)
    tracking_id = manager.get_tracking_id()
    manager.set_enable_analytics(False)
    manager.set_enable_analytics(True)

    assert manager.get_tracking_id() == tracking_id
