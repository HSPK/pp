"""Python port of `packages/coding-agent/test/suite/regressions/7153-scoped-models-refresh.test.ts`.

The TypeScript case drives `InteractiveMode.showModelsSelector`, which starts an
*asynchronous* `modelRuntime.refresh({ signal })` behind the already-rendered
selector, swaps in the refreshed snapshot when it settles, and aborts the
signal when the selector closes. This port's `ModelRuntime` has no remote
catalog layer (see its module docstring and the README's "not ported" list):
`refresh()` is synchronous, takes no signal, and cannot be in flight while the
selector is open. The parts of the regression that survive that omission are
the selector contract itself — cached models render immediately under a
"Refreshing model catalogs…" status, a completed refresh swaps the model list
and the status message, and cancelling closes the selector — and those are
asserted here. The abort-on-close assertion is skipped at its exact spot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from harness import Harness, create_harness
from pi_ai.providers.faux import FauxModelDefinition
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.keybindings import get_keybindings, set_keybindings


@pytest.fixture(autouse=True)
def _theme_and_keybindings():
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


@pytest.fixture
def harnesses() -> list[Harness]:
    created: list[Harness] = []
    yield created
    while created:
        created.pop().cleanup()


def _render(selector: ScopedModelsSelectorComponent, width: int = 100) -> str:
    return strip_ansi("\n".join(selector.render(width)))


async def test_renders_cached_models_immediately_and_updates_after_refresh(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    harness = await create_harness(
        tmp_path,
        models=[FauxModelDefinition(id="cached", name="Cached"), FauxModelDefinition(id="refreshed", name="Refreshed")],
    )
    harnesses.append(harness)

    selector = ScopedModelsSelectorComponent(
        ModelsConfig(all_models=[harness.models[0]], refresh_status="Refreshing model catalogs…"),
        ModelsCallbacks(),
    )

    initial = _render(selector)
    assert "cached" in initial
    assert "Refreshing model catalogs…" in initial
    assert "refreshed" not in initial

    selector.update_models(harness.models)
    selector.set_refresh_status("Model catalogs refreshed.", "success")

    rendered = _render(selector)
    assert "refreshed" in rendered
    assert "Model catalogs refreshed." in rendered


async def test_closing_the_selector_cancels(tmp_path: Path, harnesses: list[Harness]) -> None:
    harness = await create_harness(tmp_path, models=[FauxModelDefinition(id="cached", name="Cached")])
    harnesses.append(harness)

    cancelled: list[bool] = []
    selector = ScopedModelsSelectorComponent(
        ModelsConfig(all_models=list(harness.models), refresh_status="Refreshing model catalogs…"),
        ModelsCallbacks(on_cancel=lambda: cancelled.append(True)),
    )

    # Skipped: `expect(refresh.refreshSignal?.aborted).toBe(true)`. There is no
    # in-flight refresh to abort here -- `ModelRuntime.refresh()` is synchronous
    # and signal-less in this port.
    selector.handle_input("\x1b")

    assert cancelled == [True]
