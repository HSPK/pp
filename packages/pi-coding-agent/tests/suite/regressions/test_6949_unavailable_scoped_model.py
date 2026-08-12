"""Python port of `packages/coding-agent/test/suite/regressions/6949-unavailable-scoped-model.test.ts`.

TypeScript calls `InteractiveMode.prototype.showModelsSelector` with a
hand-built object literal for `this`. This port drives a **real**
`InteractiveMode` (via `interactive_harness.make_interactive_mode`, which is
offline: faux provider, `FakeTerminal`) and monkeypatches only the one input
the test needs to control -- the availability snapshot. Everything the method
writes to (`session.set_scoped_models`, `settings_manager.set_enabled_models`,
`_show_selector`) is the production object, so the assertions below are on real
session state rather than on calls recorded by a stand-in that could have a
different shape or async-ness than production.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from interactive_harness import make_interactive_mode
from pi_ai.registry import Model
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.model_resolver import ScopedModel
from pi_coding_agent.modes.interactive.components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
)
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.keybindings import get_keybindings, set_keybindings


@pytest.fixture(autouse=True)
def _theme_and_keybindings() -> Iterator[None]:
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


def _render(selector: ScopedModelsSelectorComponent, width: int = 100) -> str:
    return strip_ansi("\n".join(selector.render(width)))


async def _open_scoped_models_selector(
    mode: InteractiveMode,
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: list[Model],
    enabled_model_ids: list[str] | None,
    scoped_models: list[ScopedModel] | None = None,
) -> tuple[ScopedModelsSelectorComponent, list[int]]:
    """Open the selector through the real method, controlling availability only."""
    snapshot_calls = [0]

    def get_available_snapshot() -> list[Model]:
        snapshot_calls[0] += 1
        return available

    monkeypatch.setattr(mode.session.model_runtime, "get_available_snapshot", get_available_snapshot)
    mode.settings_manager.set_enabled_models(enabled_model_ids)
    mode.session.set_scoped_models(list(scoped_models or []))

    mode.show_scoped_models_selector()

    selector = mode._active_selector
    assert isinstance(selector, ScopedModelsSelectorComponent), "Expected scoped-model selector to open"
    return selector, snapshot_calls


async def test_shows_and_removes_an_enabled_model_without_a_catalog_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    model = mode.session.model_runtime.get_available_snapshot()[0]
    available_id = f"{model.provider}/{model.id}"
    unavailable_id = f"{model.provider}/unavailable"
    changes: list[list[str] | None] = []
    persisted: list[list[str] | None] = []

    # TypeScript constructs the component directly here too -- no stand-in.
    selector = ScopedModelsSelectorComponent(
        ModelsConfig(all_models=[model], enabled_model_ids=[unavailable_id, available_id]),
        ModelsCallbacks(
            on_change=changes.append,
            on_persist=persisted.append,
            on_cancel=lambda: None,
        ),
    )

    assert f"{unavailable_id} [unavailable] ✗" in _render(selector)
    selector.handle_input("\r")
    assert changes == [[available_id]]
    selector.handle_input("\x13")
    assert persisted == [[available_id]]


async def test_passes_unmatched_settings_patterns_to_the_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    provider = mode.session.model_runtime.get_available_snapshot()[0].provider
    unavailable_ids = [f"{provider}/{name}" for name in ("unavailable-one", "unavailable-two")]

    selector, snapshot_calls = await _open_scoped_models_selector(
        mode, monkeypatch, available=[], enabled_model_ids=unavailable_ids
    )

    rendered = _render(selector)
    for unavailable_id in unavailable_ids:
        assert f"{unavailable_id} [unavailable] ✗" in rendered
    assert snapshot_calls[0] > 0


async def test_opens_when_only_a_session_scoped_model_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    model = mode.session.model_runtime.get_available_snapshot()[0]
    full_id = f"{model.provider}/{model.id}"

    selector, _ = await _open_scoped_models_selector(
        mode,
        monkeypatch,
        available=[],
        enabled_model_ids=None,
        scoped_models=[ScopedModel(model=model)],
    )

    assert f"{full_id} [unavailable] ✗" in _render(selector)


async def test_does_not_clear_a_partial_scope_when_an_enabled_model_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    faux = mode.faux  # type: ignore[attr-defined]
    all_models = list(faux.models[:1])
    # The faux provider the interactive harness builds has a single model, so
    # the second and third scope entries are derived from it.
    one = all_models[0]
    two = type(one)(**{**one.__dict__, "id": "two", "name": "Two"})
    three = type(one)(**{**one.__dict__, "id": "three", "name": "Three"})
    models = [one, two, three]
    enabled_ids = [f"{model.provider}/{model.id}" for model in (one, two)]
    unavailable_id = f"{one.provider}/unavailable"

    selector, _ = await _open_scoped_models_selector(
        mode,
        monkeypatch,
        available=models,
        enabled_model_ids=[*enabled_ids, unavailable_id],
        scoped_models=[ScopedModel(model=one), ScopedModel(model=two)],
    )

    selector.handle_input("\x1b[1;3B")

    # Asserted on the real session rather than on a recorded call: the scope
    # rotates, it is not cleared by the unavailable pattern.
    assert [(scoped.model.id, scoped.thinking_level) for scoped in mode.session.scoped_models] == [
        ("two", None),
        (one.id, None),
    ]
