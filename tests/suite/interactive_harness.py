"""Helper for suite tests that need a real `InteractiveMode`.

Several TypeScript regression tests call `InteractiveMode.prototype.<method>`
with a hand-built `this`. Where the method under test touches enough of the
mode to make that impractical in Python, these tests build a real
`InteractiveMode` over the faux provider and a `FakeTerminal` instead, the same
way `tests/test_interactive_mode.py` does.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pi_ai.providers.faux import faux_provider
from pi_tui.testing import FakeTerminal

from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode, InteractiveModeOptions
from pi_coding_agent.modes.interactive.theme.theme import init_theme

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def strip_ansi_lines(lines: list[str]) -> str:
    return _ANSI_RE.sub("", "\n".join(lines))


async def make_interactive_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InteractiveMode:
    init_theme("dark")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    home_dir = tmp_path / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("PI_OFFLINE", "1")

    faux = faux_provider()
    model_runtime = await ModelRuntime.create(agent_dir=str(agent_dir), providers=[faux.provider])
    await model_runtime.login(faux.provider.id, "faux-key")
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            model=faux.models[0],
            model_runtime=model_runtime,
        )
    )

    async def create_runtime(**_kwargs: Any) -> Any:
        raise AssertionError("session replacement is not exercised by these tests")

    runtime = AgentSessionRuntime(result.session, str(agent_dir), create_runtime, result.model_fallback_message)
    mode = InteractiveMode(runtime, InteractiveModeOptions(verbose=False), terminal=FakeTerminal(columns=120, rows=40))
    mode.faux = faux  # type: ignore[attr-defined]
    return mode
