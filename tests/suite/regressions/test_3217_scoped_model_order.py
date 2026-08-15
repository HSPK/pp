"""Python port of `packages/coding-agent/test/suite/regressions/3217-scoped-model-order.test.ts`."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness import create_harness
from pi_ai.providers.faux import FauxModelDefinition
from pi_tui.keybindings import get_keybindings, set_keybindings

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.model_selector import ModelSelectorComponent, ScopedModelItem
from pi_coding_agent.modes.interactive.components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi

_MODELS = [
    FauxModelDefinition(id="faux-1", name="One", reasoning=True),
    FauxModelDefinition(id="faux-2", name="Two", reasoning=True),
    FauxModelDefinition(id="faux-3", name="Three", reasoning=True),
]


class _FakeTui:
    def request_render(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolate_globals():
    init_theme("dark")
    # Keybindings are a global singleton; reset them so other tests cannot leak in.
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


async def test_propagates_reordered_scoped_models_back_to_the_session_state(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, models=_MODELS)
    try:
        ordered_ids = [f"{model.provider}/{model.id}" for model in harness.models]
        changes: list[list[str] | None] = []
        selector = ScopedModelsSelectorComponent(
            ModelsConfig(all_models=list(harness.models), enabled_model_ids=ordered_ids),
            ModelsCallbacks(on_change=changes.append, on_persist=lambda _ids: None, on_cancel=lambda: None),
        )

        selector.handle_input("\x1b[1;3B")

        assert changes == [[ordered_ids[1], ordered_ids[0], ordered_ids[2]]]
    finally:
        harness.cleanup()


async def test_preserves_scoped_model_order_in_the_model_scoped_tab(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, models=_MODELS)
    try:
        model_one = harness.get_model("faux-1")
        model_two = harness.get_model("faux-2")
        model_three = harness.get_model("faux-3")
        assert model_one is not None and model_two is not None and model_three is not None

        selector = ModelSelectorComponent(
            _FakeTui(),
            model_one,
            harness.settings_manager,
            harness.session.model_runtime,
            [ScopedModelItem(model=model_two), ScopedModelItem(model=model_one), ScopedModelItem(model=model_three)],
            lambda _model: None,
            lambda: None,
        )

        rendered = strip_ansi("\n".join(selector.render(120)))
        assert f"[{model_one.provider}]" in rendered
        assert "Model catalogs refreshed." in rendered

        rendered_lines = [line for line in rendered.split("\n") if f"[{model_one.provider}]" in line]
        ordered_ids = [line.strip().lstrip("→").strip().split(" [")[0].strip() for line in rendered_lines[:3]]

        assert ordered_ids == [model_two.id, model_one.id, model_three.id]
    finally:
        harness.cleanup()
