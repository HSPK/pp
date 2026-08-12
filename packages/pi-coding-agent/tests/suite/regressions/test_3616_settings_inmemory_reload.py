"""Python port of `packages/coding-agent/test/suite/regressions/3616-settings-inmemory-reload.test.ts`.

An in-memory `SettingsManager` has no file to re-read, so `reload()` (and any
reload driven by the resource loader) must keep the settings it was seeded
with instead of resetting to `{}`.
"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.settings_manager import SettingsManager


async def test_preserves_initial_settings_after_direct_reload() -> None:
    settings_manager = SettingsManager.in_memory(
        {
            "defaultThinkingLevel": "high",
            "images": {"autoResize": False},
            "compaction": {"enabled": False},
        }
    )

    await settings_manager.reload()

    assert settings_manager.get_default_thinking_level() == "high"
    assert settings_manager.get_image_auto_resize() is False
    assert settings_manager.get_compaction_enabled() is False
    assert settings_manager.get_global_settings() == {
        "defaultThinkingLevel": "high",
        "images": {"autoResize": False},
        "compaction": {"enabled": False},
    }


async def test_preserves_initial_settings_when_the_resource_loader_reloads(tmp_path: Path) -> None:
    settings_manager = SettingsManager.in_memory(
        {
            "defaultThinkingLevel": "high",
            "images": {"autoResize": False},
            "compaction": {"enabled": False},
        }
    )
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    # This port's `ResourceLoader` takes no `settingsManager`; it is reloaded
    # alongside the settings the same way the CLI does.
    resource_loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            no_skills=True,
            no_prompt_templates=True,
            no_context_files=True,
        )
    )

    resource_loader.reload()

    assert settings_manager.get_default_thinking_level() == "high"
    assert settings_manager.get_image_auto_resize() is False
    assert settings_manager.get_compaction_enabled() is False


async def test_preserves_initial_settings_after_an_unrelated_setter_flush_and_reload() -> None:
    settings_manager = SettingsManager.in_memory(
        {
            "images": {"autoResize": False},
            "compaction": {"enabled": False},
        }
    )

    settings_manager.set_theme("dark")
    await settings_manager.flush()
    await settings_manager.reload()

    assert settings_manager.get_theme() == "dark"
    assert settings_manager.get_image_auto_resize() is False
    assert settings_manager.get_compaction_enabled() is False
    assert settings_manager.get_global_settings() == {
        "images": {"autoResize": False},
        "compaction": {"enabled": False},
        "theme": "dark",
    }
