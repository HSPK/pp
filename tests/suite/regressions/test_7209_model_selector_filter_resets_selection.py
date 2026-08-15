"""Python port of `packages/coding-agent/test/suite/regressions/7209-model-selector-filter-resets-selection.test.ts`."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness import Harness, create_harness
from pi_ai.providers.faux import FauxModelDefinition
from pi_tui.keybindings import get_keybindings, set_keybindings

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.model_selector import ModelSelectorComponent, ScopedModelItem
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi


class _FakeTui:
    def request_render(self) -> None:
        return None


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


def selected_model_id(rendered: str) -> str | None:
    """Model id of the highlighted (→) row in the rendered selector."""
    line = next((line for line in rendered.split("\n") if line.startswith("→ ")), None)
    if line is None:
        return None
    rest = line[2:].lstrip()
    return rest.split(" [")[0].strip() or None


def _render(selector: ModelSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(120)))


async def test_moves_selection_to_the_first_row_in_the_all_tab_when_typing_a_query(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    harness = await create_harness(
        tmp_path,
        models=[
            FauxModelDefinition(id="alpha-1", name="Alpha One", reasoning=True),
            FauxModelDefinition(id="alpha-2", name="Alpha Two", reasoning=True),
            FauxModelDefinition(id="alpha-3", name="Alpha Three", reasoning=True),
            FauxModelDefinition(id="beta-1", name="Beta One", reasoning=True),
        ],
    )
    harnesses.append(harness)

    current = harness.get_model("alpha-1")
    assert current is not None
    selector = ModelSelectorComponent(
        _FakeTui(),
        current,
        harness.settings_manager,
        harness.session.model_runtime,
        [],
        lambda _model: None,
        lambda: None,
    )

    # TS awaits `vi.waitFor` for the asynchronous remote catalog refresh; this
    # port's `ModelRuntime.refresh()` is synchronous, so the banner is already
    # rendered by the time the constructor returns.
    assert "Model catalogs refreshed." in _render(selector)

    # Current model (alpha-1) is sorted first, so selection starts on row 0.
    assert selected_model_id(_render(selector)) == "alpha-1"

    # Move selection down two rows to alpha-3.
    selector.handle_input("\x1b[B")
    selector.handle_input("\x1b[B")
    assert selected_model_id(_render(selector)) == "alpha-3"

    # Type a query that matches the three alpha models. The selection must
    # move back to the top row (alpha-1), not stay clamped at index 2.
    for char in "alpha":
        selector.handle_input(char)

    rendered = _render(selector)
    assert selected_model_id(rendered) == "alpha-1"
    # Sanity: the filter actually narrowed the list.
    assert "beta-1" not in rendered


async def test_moves_selection_to_the_first_row_in_the_scoped_tab_when_typing_a_query(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    harness = await create_harness(
        tmp_path,
        models=[
            FauxModelDefinition(id="alpha-1", name="Alpha One", reasoning=True),
            FauxModelDefinition(id="alpha-2", name="Alpha Two", reasoning=True),
            FauxModelDefinition(id="alpha-3", name="Alpha Three", reasoning=True),
        ],
    )
    harnesses.append(harness)

    alpha1 = harness.get_model("alpha-1")
    alpha2 = harness.get_model("alpha-2")
    alpha3 = harness.get_model("alpha-3")
    assert alpha1 is not None and alpha2 is not None and alpha3 is not None

    # Scoped list is intentionally not in current-model-first order; the
    # current model (alpha-1) sits at index 2.
    selector = ModelSelectorComponent(
        _FakeTui(),
        alpha1,
        harness.settings_manager,
        harness.session.model_runtime,
        [ScopedModelItem(model=alpha2), ScopedModelItem(model=alpha3), ScopedModelItem(model=alpha1)],
        lambda _model: None,
        lambda: None,
    )

    assert "Model catalogs refreshed." in _render(selector)

    # Selection starts on the current model (alpha-1), which is row 2 here.
    assert selected_model_id(_render(selector)) == "alpha-1"

    # Type a query matching all three scoped models. Selection must move to
    # the top row (alpha-2), not stay clamped at index 2 (alpha-1).
    for char in "alpha":
        selector.handle_input(char)

    assert selected_model_id(_render(selector)) == "alpha-2"
