"""Python port of `packages/coding-agent/test/model-selector.test.ts`.

The TypeScript case ("lists every catalog that failed to refresh") stubs
`modelRuntime.refresh()` to resolve with a `{ aborted, errors: Map }` result
and asserts the aggregated
"Could not refresh 2 model catalogs (openai, anthropic); showing cached
models." banner. That result type belongs to the remote catalog layer
(`remote-catalog-provider.ts` + `ModelRuntime.refresh({ signal })`) which this
port deliberately omits -- `ModelRuntime.refresh()` here rebuilds the local
composition only, returns nothing, and can only fail as a whole. The
per-catalog aggregation assertion therefore has no counterpart.

What is portable is the surviving failure path the selector still has: a
refresh that raises surfaces "Could not refresh model catalogs: <error>" over
the cached model list, which is the same user-visible contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_ai.providers.faux import faux_provider
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.components.model_selector import ModelSelectorComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi


class _FakeTui:
    def request_render(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _theme():
    init_theme("dark")


async def _make_runtime(tmp_path: Path) -> tuple[ModelRuntime, object]:
    faux = faux_provider()
    runtime = await ModelRuntime.create(agent_dir=str(tmp_path), providers=[faux.provider])
    await runtime.login(faux.provider.id, "faux-key")
    return runtime, faux


async def test_reports_a_failed_catalog_refresh_over_the_cached_models(tmp_path: Path):
    runtime, faux = await _make_runtime(tmp_path)

    def failing_refresh() -> None:
        raise RuntimeError("unavailable")

    runtime.refresh = failing_refresh  # type: ignore[method-assign]

    selector = ModelSelectorComponent(
        _FakeTui(),
        faux.models[0],
        SettingsManager.in_memory(),
        runtime,
        [],
        lambda _model: None,
        lambda: None,
    )

    rendered = strip_ansi("\n".join(selector.render(120)))
    assert "Could not refresh model catalogs: unavailable" in rendered
    assert selector.refresh_status_message == ""


async def test_reports_success_and_keeps_the_cached_models_listed(tmp_path: Path):
    runtime, faux = await _make_runtime(tmp_path)

    selector = ModelSelectorComponent(
        _FakeTui(),
        faux.models[0],
        SettingsManager.in_memory(),
        runtime,
        [],
        lambda _model: None,
        lambda: None,
    )

    rendered = strip_ansi("\n".join(selector.render(120)))
    assert selector.error_message is None
    assert "Model catalogs refreshed." in rendered
    assert faux.models[0].id in rendered
