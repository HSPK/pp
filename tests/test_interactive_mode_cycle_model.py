"""Regression cover for `cycleModel` in `packages/coding-agent/src/modes/interactive/interactive-mode.ts`.

The TypeScript suite has no test file dedicated to `cycleModel`, so this is not
a port of one. It exists because reading `interactive-mode.ts` against
`interactive_mode.py` side by side showed the Python `_cycle_model` had drifted
from the TypeScript original in four separate ways (status text, the
"only one model" branch, the editor border refresh, and error handling); these
cases pin the TypeScript behaviour so it cannot drift back.

The session driven here is a real `AgentSession` built by
`test_agent_session.build_session` against a scripted, offline provider -- only
the surrounding TUI surface (`footer`, `show_status`, `show_error`) is stood
in, because those are what `cycleModel` is asserted *about*.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_agent_session import REASONING_MODEL, TEST_MODEL, build_session

from pi_coding_agent.core.agent_session import AgentSession, ScopedModel
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


class _Footer:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _CycleModelThis:
    """Stand-in `self` for `_cycle_model`, holding a real `AgentSession`."""

    _cycle_model = InteractiveMode._cycle_model

    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self.footer = _Footer()
        self.statuses: list[str] = []
        self.errors: list[str] = []
        self.border_updates = 0
        self.warned_models: list[str] = []

    def show_status(self, message: str) -> None:
        self.statuses.append(message)

    def show_error(self, message: str) -> None:
        self.errors.append(message)

    def _update_editor_border_color(self) -> None:
        self.border_updates += 1

    async def _maybe_warn_about_anthropic_subscription_auth(self, model: object = None) -> None:
        self.warned_models.append(getattr(model, "id", ""))


async def test_reports_the_switched_to_model_name_and_refreshes_the_border(tmp_path: Path) -> None:
    session, _sm, _stm, _stream = await build_session(tmp_path, extra_provider_models=[REASONING_MODEL])
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)

        # TS: `Switched to ${result.model.name || result.model.id}${thinkingStr}`.
        # A reasoning model defaults to thinking level "medium", so the suffix
        # is present.
        assert fake.statuses == ["Switched to Test Reasoning Model (thinking: medium)"]
        assert fake.errors == []
        assert fake.footer.invalidated == 1
        assert fake.border_updates == 1
        assert session.model.id == "test-reasoning"
    finally:
        session.dispose()


async def test_appends_the_thinking_level_for_a_reasoning_model(tmp_path: Path) -> None:
    scoped = [
        ScopedModel(model=TEST_MODEL),
        ScopedModel(model=REASONING_MODEL, thinking_level="low"),
    ]
    session, _sm, _stm, _stream = await build_session(
        tmp_path, extra_provider_models=[REASONING_MODEL], scoped_models=scoped
    )
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)

        assert fake.statuses == ["Switched to Test Reasoning Model (thinking: low)"]
    finally:
        session.dispose()


async def test_falls_back_to_the_model_id_when_the_model_has_no_name(tmp_path: Path) -> None:
    from dataclasses import replace

    unnamed = replace(REASONING_MODEL, name="")
    session, _sm, _stm, _stream = await build_session(tmp_path, extra_provider_models=[unnamed])
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)

        assert fake.statuses == ["Switched to test-reasoning (thinking: medium)"]
    finally:
        session.dispose()


async def test_reports_only_one_model_available_when_there_is_nothing_to_cycle_to(
    tmp_path: Path,
) -> None:
    session, _sm, _stm, _stream = await build_session(tmp_path)
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)

        assert fake.statuses == ["Only one model available"]
        # TS returns before `footer.invalidate()` / `updateEditorBorderColor()`
        # on this branch.
        assert fake.footer.invalidated == 0
        assert fake.border_updates == 0
        assert fake.warned_models == []
    finally:
        session.dispose()


async def test_reports_only_one_model_in_scope_when_models_are_scoped(tmp_path: Path) -> None:
    """Both scoped models exist but only one provider model is registered, so cycling finds no target."""
    scoped = [ScopedModel(model=TEST_MODEL), ScopedModel(model=REASONING_MODEL)]
    session, _sm, _stm, _stream = await build_session(tmp_path, scoped_models=scoped)
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)

        assert fake.statuses == ["Only one model in scope"]
    finally:
        session.dispose()


async def test_shows_a_cycle_failure_as_an_error_instead_of_propagating(tmp_path: Path) -> None:
    session, _sm, _stm, _stream = await build_session(tmp_path, extra_provider_models=[REASONING_MODEL])
    fake = _CycleModelThis(session)

    # The replacement matches the real `AgentSession.cycle_model` shape: an
    # async method taking the direction, so an unawaited call could not pass.
    async def failing_cycle_model(direction: str = "forward") -> None:
        raise RuntimeError("cycle exploded")

    session.cycle_model = failing_cycle_model  # type: ignore[method-assign]
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)

        assert fake.errors == ["cycle exploded"]
        assert fake.statuses == []
        assert fake.footer.invalidated == 0
    finally:
        session.dispose()


async def test_warns_about_anthropic_subscription_auth_for_the_new_model(tmp_path: Path) -> None:
    session, _sm, _stm, _stream = await build_session(tmp_path, extra_provider_models=[REASONING_MODEL])
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)
        for _ in range(20):
            if fake.warned_models:
                break
            await asyncio.sleep(0)

        assert fake.warned_models == ["test-reasoning"]
    finally:
        session.dispose()


@pytest.mark.parametrize("direction", ["forward", "backward"])
async def test_cycles_in_both_directions(tmp_path: Path, direction: str) -> None:
    session, _sm, _stm, _stream = await build_session(tmp_path, extra_provider_models=[REASONING_MODEL])
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model(direction), timeout=5)

        assert session.model.id == "test-reasoning"
        assert fake.statuses == ["Switched to Test Reasoning Model (thinking: medium)"]
    finally:
        session.dispose()


async def test_omits_the_thinking_suffix_for_a_non_reasoning_model(tmp_path: Path) -> None:
    session, _sm, _stm, _stream = await build_session(tmp_path, extra_provider_models=[REASONING_MODEL])
    fake = _CycleModelThis(session)
    try:
        await asyncio.wait_for(fake._cycle_model("forward"), timeout=5)
        await asyncio.wait_for(fake._cycle_model("backward"), timeout=5)

        # TS only appends the suffix when `result.model.reasoning` is set;
        # `TEST_MODEL` is not a reasoning model.
        assert session.model.id == "test-model"
        assert fake.statuses[-1] == "Switched to Test Model"
    finally:
        session.dispose()
