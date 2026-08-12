"""Python port of `packages/coding-agent/test/trust-manager.test.ts`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pi_coding_agent.core.trust_manager import ProjectTrustStore, has_trust_requiring_project_resources


def test_stores_decisions_and_inherits_from_parent_directories(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    store = ProjectTrustStore(str(agent_dir))
    parent_dir = tmp_path / "trusted-parent"
    child_dir = parent_dir / "project"
    child_dir.mkdir(parents=True)

    assert store.get(str(child_dir)) is None
    store.set(str(parent_dir), True)
    assert store.get(str(child_dir)) is True
    store.set(str(child_dir), False)
    assert store.get(str(child_dir)) is False
    store.set(str(child_dir), None)
    assert store.get(str(child_dir)) is True


def test_detects_trust_requiring_project_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))

    (tmp_path / ".pi" / "agent").mkdir(parents=True)
    (tmp_path / ".agents" / "skills").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(tmp_path)) is False
    assert has_trust_requiring_project_resources(str(cwd)) is False

    (tmp_path / ".pi" / "settings.json").write_text("{}", encoding="utf-8")
    assert has_trust_requiring_project_resources(str(tmp_path)) is True
    (tmp_path / ".pi" / "settings.json").unlink()

    (cwd / ".pi").mkdir(parents=True)
    (cwd / ".pi" / "settings.json").write_text("{}", encoding="utf-8")
    assert has_trust_requiring_project_resources(str(cwd)) is True

    shutil.rmtree(cwd / ".pi")
    (cwd / ".agents" / "skills").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(cwd)) is True
