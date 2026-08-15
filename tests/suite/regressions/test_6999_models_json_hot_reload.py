"""Python port of `packages/coding-agent/test/suite/regressions/6999-models-json-hot-reload.test.ts`.

TypeScript builds the runtime through `test/model-runtime-test-utils.ts`; here
`ModelRuntime.create(models_path=...)` is the equivalent. The TS test also
awaits a second `requestRender` because its refresh is asynchronous -- this
port's `ModelRuntime.refresh()` is synchronous local-only work (no network
catalog fetch, see `model_runtime.py`), so the selector is already up to date
when the constructor returns; the render-count observation is replaced by a
direct render assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_tui.keybindings import get_keybindings, set_keybindings

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.components.model_selector import ModelSelectorComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi


class _FakeTui:
    def __init__(self) -> None:
        self.render_count = 0

    def request_render(self) -> None:
        self.render_count += 1


@pytest.fixture(autouse=True)
def _theme_and_keybindings():
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


def models_json(provider: str, model: str) -> dict[str, object]:
    return {
        "providers": {
            provider: {
                "baseUrl": "https://example.test/v1",
                "api": "openai-completions",
                "apiKey": "test-key",
                "models": [{"id": model}],
            }
        }
    }


async def test_reloads_models_json_when_opening_model(tmp_path: Path) -> None:
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(models_json("old-provider", "old-model")), encoding="utf-8")
    runtime = await ModelRuntime.create(
        agent_dir=str(tmp_path / "agent"),
        models_path=str(models_path),
        providers=[],
    )
    assert runtime.get_model("old-provider", "old-model") is not None

    models_path.write_text(json.dumps(models_json("new-provider", "new-model")), encoding="utf-8")

    tui = _FakeTui()
    selector = ModelSelectorComponent(
        tui,
        None,
        SettingsManager.in_memory(),
        runtime,
        [],
        lambda model: None,
        lambda: None,
    )

    rendered = strip_ansi("\n".join(selector.render(120)))
    assert "new-model [new-provider]" in rendered
    assert "old-model [old-provider]" not in rendered
    assert tui.render_count >= 2
