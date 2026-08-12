"""Python port of `packages/coding-agent/test/max-thinking.test.ts`."""

from __future__ import annotations

import json
from pathlib import Path

from pi_coding_agent.cli.args import is_valid_thinking_level
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.theme.theme import load_theme_from_path

_THEME_DIR = Path(__file__).resolve().parents[1] / "src" / "pi_coding_agent" / "modes" / "interactive" / "theme"


async def test_max_thinking_level_is_accepted_by_cli_and_settings():
    assert is_valid_thinking_level("max") is True

    settings = SettingsManager.in_memory()
    settings.set_default_thinking_level("max")
    await settings.flush()

    assert settings.get_default_thinking_level() == "max"


def test_falls_back_to_thinking_xhigh_for_legacy_themes(tmp_path: Path):
    dark_theme = json.loads((_THEME_DIR / "dark.json").read_text(encoding="utf-8"))
    dark_theme["name"] = "legacy-theme"
    del dark_theme["colors"]["thinkingMax"]
    theme_path = tmp_path / "legacy-theme.json"
    theme_path.write_text(json.dumps(dark_theme), encoding="utf-8")

    legacy_theme = load_theme_from_path(str(theme_path))

    assert legacy_theme.get_thinking_border_color("max")("border") == legacy_theme.get_thinking_border_color("xhigh")(
        "border"
    )
