"""Python port of `packages/coding-agent/test/first-time-setup-fork.test.ts`.

The TypeScript test mocks the `PACKAGE_NAME` export of `src/config.ts`; here the
constant is imported into `cli/startup_ui.py`, so the port patches it there.
"""

from __future__ import annotations

import pytest

from pi_coding_agent.cli import startup_ui


def test_returns_false_for_a_forked_package(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(startup_ui, "PACKAGE_NAME", "@example/pi-coding-agent")
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.delenv(startup_ui.ENV_AGENT_DIR, raising=False)
    settings_path = str(tmp_path / "settings.json")

    assert startup_ui.should_run_first_time_setup(settings_path) is False
