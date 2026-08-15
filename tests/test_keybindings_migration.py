"""Python port of `packages/coding-agent/test/keybindings-migration.test.ts`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.config import ENV_AGENT_DIR
from pi_coding_agent.migrations import run_migrations


def _create_agent_dir(tmp_path: Path, config: dict[str, object]) -> Path:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "keybindings.json").write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf-8")
    return agent_dir


def _read_config(agent_dir: Path) -> dict[str, object]:
    return json.loads((agent_dir / "keybindings.json").read_text(encoding="utf-8"))


def test_rewrites_old_key_names_to_namespaced_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    agent_dir = _create_agent_dir(tmp_path, {"cursorUp": ["up", "ctrl+p"], "expandTools": "ctrl+x"})
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))

    run_migrations(str(tmp_path), str(agent_dir))

    assert _read_config(agent_dir) == {
        "tui.editor.cursorUp": ["up", "ctrl+p"],
        "app.tools.expand": "ctrl+x",
    }


def test_keeps_the_namespaced_value_when_old_and_new_names_both_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    agent_dir = _create_agent_dir(tmp_path, {"expandTools": "ctrl+x", "app.tools.expand": "ctrl+y"})
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))

    run_migrations(str(tmp_path), str(agent_dir))

    assert _read_config(agent_dir) == {"app.tools.expand": "ctrl+y"}


def test_loads_old_key_names_in_memory_before_the_file_is_rewritten(tmp_path: Path):
    agent_dir = _create_agent_dir(tmp_path, {"selectConfirm": "enter", "interrupt": "ctrl+x"})

    keybindings = KeybindingsManager.create(str(agent_dir))

    assert keybindings.get_user_bindings() == {
        "tui.select.confirm": "enter",
        "app.interrupt": "ctrl+x",
    }
    effective = keybindings.get_effective_config()
    assert effective["tui.select.confirm"] == "enter"
    assert effective["app.interrupt"] == "ctrl+x"
