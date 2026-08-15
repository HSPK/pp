"""Tests for `core/settings_manager.py`.

Ported from `test/settings-manager.test.ts` and `test/settings-manager-bug.test.ts`
in the TypeScript coding-agent package. All filesystem work uses `tmp_path`;
no test reads or writes the real `$HOME`/agent dir.
"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.core import settings_manager as settings_manager_module
from pi_coding_agent.core.settings_manager import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    SettingsManager,
    SettingsManagerCreateOptions,
)


@pytest.fixture
def dirs(tmp_path):
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir()
    (project_dir / ".pi").mkdir(parents=True)
    return str(project_dir), str(agent_dir)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# -- preserves externally added settings -------------------------------------


async def test_preserves_enabled_models_when_changing_thinking_level(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"theme": "dark", "defaultModel": "claude-sonnet"})

    manager = SettingsManager.create(project_dir, agent_dir)

    current = json.loads(settings_path.read_text())
    current["enabledModels"] = ["claude-opus-4-5", "gpt-5.2-codex"]
    settings_path.write_text(json.dumps(current, indent=2))

    manager.set_default_thinking_level("high")
    await manager.flush()

    saved = json.loads(settings_path.read_text())
    assert saved["enabledModels"] == ["claude-opus-4-5", "gpt-5.2-codex"]
    assert saved["defaultThinkingLevel"] == "high"
    assert saved["theme"] == "dark"
    assert saved["defaultModel"] == "claude-sonnet"


async def test_preserves_custom_settings_when_changing_theme(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"defaultModel": "claude-sonnet"})

    manager = SettingsManager.create(project_dir, agent_dir)

    current = json.loads(settings_path.read_text())
    current["shellPath"] = "/bin/zsh"
    current["extensions"] = ["/path/to/extension.ts"]
    settings_path.write_text(json.dumps(current, indent=2))

    manager.set_theme("light")
    await manager.flush()

    saved = json.loads(settings_path.read_text())
    assert saved["shellPath"] == "/bin/zsh"
    assert saved["extensions"] == ["/path/to/extension.ts"]
    assert saved["theme"] == "light"


async def test_in_memory_changes_override_file_changes_for_same_key(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"theme": "dark"})

    manager = SettingsManager.create(project_dir, agent_dir)

    current = json.loads(settings_path.read_text())
    current["defaultThinkingLevel"] = "low"
    settings_path.write_text(json.dumps(current, indent=2))

    manager.set_default_thinking_level("high")
    await manager.flush()

    saved = json.loads(settings_path.read_text())
    assert saved["defaultThinkingLevel"] == "high"


# -- packages migration -------------------------------------------------------


def test_keeps_local_only_extensions_in_extensions_array(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(
        tmp_path / "agent" / "settings.json",
        {"extensions": ["/local/ext.ts", "./relative/ext.ts"]},
    )

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_packages() == []
    assert manager.get_extension_paths() == ["/local/ext.ts", "./relative/ext.ts"]


def test_handles_packages_with_filtering_objects(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(
        tmp_path / "agent" / "settings.json",
        {
            "packages": [
                "npm:simple-pkg",
                {"source": "npm:shitty-extensions", "extensions": ["extensions/oracle.ts"], "skills": []},
            ]
        },
    )

    manager = SettingsManager.create(project_dir, agent_dir)

    packages = manager.get_packages()
    assert len(packages) == 2
    assert packages[0] == "npm:simple-pkg"
    assert packages[1] == {"source": "npm:shitty-extensions", "extensions": ["extensions/oracle.ts"], "skills": []}


# -- reload --------------------------------------------------------------------


async def test_reload_reloads_global_settings_from_disk(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"theme": "dark", "extensions": ["/before.ts"]})

    manager = SettingsManager.create(project_dir, agent_dir)

    write_json(settings_path, {"theme": "light", "extensions": ["/after.ts"], "defaultModel": "claude-sonnet"})

    await manager.reload()

    assert manager.get_theme() == "light"
    assert manager.get_extension_paths() == ["/after.ts"]
    assert manager.get_default_model() == "claude-sonnet"


async def test_reload_keeps_previous_settings_when_file_is_invalid(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"theme": "dark"})

    manager = SettingsManager.create(project_dir, agent_dir)

    settings_path.write_text("{ invalid json")
    await manager.reload()

    assert manager.get_theme() == "dark"


# -- theme setting ---------------------------------------------------------------


async def test_theme_slash_separated_automatic_setting_stored_separately(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "light/dark"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_theme() is None
    assert manager.get_theme_setting() == "light/dark"

    manager.set_theme("solarized-light/tokyo-night")
    await manager.flush()

    saved = json.loads((tmp_path / "agent" / "settings.json").read_text())
    assert saved["theme"] == "solarized-light/tokyo-night"


# -- error tracking ----------------------------------------------------------------


def test_collects_and_clears_load_errors_via_drain_errors(dirs, tmp_path):
    project_dir, agent_dir = dirs
    (tmp_path / "agent" / "settings.json").write_text("{ invalid global json")
    (tmp_path / "project" / ".pi" / "settings.json").write_text("{ invalid project json")

    manager = SettingsManager.create(project_dir, agent_dir)
    errors = manager.drain_errors()

    assert len(errors) == 2
    assert sorted(e.scope for e in errors) == ["global", "project"]
    assert manager.drain_errors() == []


# -- project trust ------------------------------------------------------------------


def test_skips_project_settings_when_project_is_not_trusted(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "global"})
    write_json(tmp_path / "project" / ".pi" / "settings.json", {"theme": "project"})

    manager = SettingsManager.create(project_dir, agent_dir, SettingsManagerCreateOptions(project_trusted=False))

    assert manager.is_project_trusted() is False
    assert manager.get_theme() == "global"
    assert manager.get_project_settings() == {}


def test_reloads_project_settings_after_trust_changes_to_true(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "global"})
    write_json(tmp_path / "project" / ".pi" / "settings.json", {"theme": "project"})
    manager = SettingsManager.create(project_dir, agent_dir, SettingsManagerCreateOptions(project_trusted=False))

    manager.set_project_trusted(True)

    assert manager.is_project_trusted() is True
    assert manager.get_theme() == "project"


async def test_fails_project_settings_writes_when_project_is_not_trusted(dirs, tmp_path):
    project_dir, agent_dir = dirs
    project_settings_path = tmp_path / "project" / ".pi" / "settings.json"
    write_json(project_settings_path, {"packages": ["npm:existing"]})
    manager = SettingsManager.create(project_dir, agent_dir, SettingsManagerCreateOptions(project_trusted=False))

    with pytest.raises(RuntimeError, match="Project is not trusted; refusing to write project settings"):
        manager.set_project_packages(["npm:new"])
    await manager.flush()

    assert manager.get_project_settings() == {}
    assert json.loads(project_settings_path.read_text()) == {"packages": ["npm:existing"]}


def test_reads_default_project_trust_from_global_settings_only(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"defaultProjectTrust": "always"})
    write_json(tmp_path / "project" / ".pi" / "settings.json", {"defaultProjectTrust": "never"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_default_project_trust() == "always"


def test_defaults_invalid_project_trust_settings_to_ask(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"defaultProjectTrust": "sometimes"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_default_project_trust() == "ask"


# -- project settings directory creation --------------------------------------------


def test_does_not_create_pi_folder_when_only_reading_project_settings(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "dark"})
    import shutil

    shutil.rmtree(tmp_path / "project" / ".pi")

    manager = SettingsManager.create(project_dir, agent_dir)

    assert not (tmp_path / "project" / ".pi").exists()
    assert manager.get_theme() == "dark"


async def test_creates_pi_folder_when_writing_project_settings(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "dark"})
    import shutil

    shutil.rmtree(tmp_path / "project" / ".pi")

    manager = SettingsManager.create(project_dir, agent_dir)
    assert not (tmp_path / "project" / ".pi").exists()

    manager.set_project_packages([{"source": "npm:test-pkg"}])
    await manager.flush()

    assert (tmp_path / "project" / ".pi").exists()
    assert (tmp_path / "project" / ".pi" / "settings.json").exists()


# -- httpIdleTimeoutMs ---------------------------------------------------------------


def test_http_idle_timeout_defaults_to_5_minutes(dirs):
    project_dir, agent_dir = dirs
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_http_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS


def test_http_idle_timeout_uses_merged_global_and_project_settings(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"httpIdleTimeoutMs": 300000})
    write_json(tmp_path / "project" / ".pi" / "settings.json", {"httpIdleTimeoutMs": 0})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_http_idle_timeout_ms() == 0


def test_http_idle_timeout_rejects_invalid_values(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"httpIdleTimeoutMs": -1})
    manager = SettingsManager.create(project_dir, agent_dir)

    with pytest.raises(ValueError, match="Invalid httpIdleTimeoutMs setting"):
        manager.get_http_idle_timeout_ms()


# -- externalEditor -----------------------------------------------------------------


def test_external_editor_resolves_by_precedence(monkeypatch):
    # TS mutates `process.env` directly, so `getExternalEditorCommand()` is
    # exercised on its real environment lookup. Do the same here rather than
    # passing `env=`, which would leave the production `env is None -> os.environ`
    # branch untested.
    monkeypatch.setenv("VISUAL", "vim")
    monkeypatch.setenv("EDITOR", "nano")
    assert SettingsManager.in_memory({"externalEditor": "code --wait"}).get_external_editor_command() == "code --wait"
    assert SettingsManager.in_memory().get_external_editor_command() == "vim"

    monkeypatch.delenv("VISUAL")
    monkeypatch.setenv("EDITOR", "emacs")
    assert SettingsManager.in_memory().get_external_editor_command() == "emacs"


def test_external_editor_falls_back_to_platform_defaults(monkeypatch):
    # TS flips `process.platform` between win32/darwin/linux; the port branches on
    # `os.name`, which is "nt" on Windows and "posix" on both darwin and linux.
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr(settings_manager_module.os, "name", "nt")
    assert SettingsManager.in_memory().get_external_editor_command() == "notepad"

    monkeypatch.setattr(settings_manager_module.os, "name", "posix")
    assert SettingsManager.in_memory().get_external_editor_command() == "nano"


# -- TUI mode -------------------------------------------------------------------------


async def test_tui_mode_defaults_to_regular_and_persists_fullscreen(dirs, tmp_path):
    project_dir, agent_dir = dirs
    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_tui_mode() == "regular"

    manager.set_tui_mode("fullscreen")
    await manager.flush()

    assert manager.get_tui_mode() == "fullscreen"
    saved = json.loads((tmp_path / "agent" / "settings.json").read_text())
    assert saved["tuiMode"] == "fullscreen"


def test_tui_mode_falls_back_to_regular_for_unsupported_values(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"tuiMode": "other"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_tui_mode() == "regular"


def test_tui_mode_does_not_recognize_old_ui_mode_setting(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"uiMode": "fullscreen"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_tui_mode() == "regular"


async def test_validates_and_persists_fullscreen_settings(dirs, tmp_path):
    project_dir, agent_dir = dirs
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_fullscreen_exit_output() == "transcript"
    assert manager.get_fullscreen_scrollbar() == "auto"

    manager.set_fullscreen_exit_output("resume-hint")
    manager.set_fullscreen_scrollbar("hidden")
    await manager.flush()
    saved = json.loads((tmp_path / "agent" / "settings.json").read_text())
    assert saved["fullscreenExitOutput"] == "resume-hint"
    assert saved["fullscreenScrollbar"] == "hidden"

    write_json(
        tmp_path / "agent" / "settings.json",
        {"fullscreenExitOutput": "nothing", "fullscreenScrollbar": "sometimes"},
    )
    reloaded = SettingsManager.create(project_dir, agent_dir)
    assert reloaded.get_fullscreen_exit_output() == "transcript"
    assert reloaded.get_fullscreen_scrollbar() == "auto"


# -- outputPad -----------------------------------------------------------------------


async def test_output_pad_defaults_to_1_and_persists_binary_values(dirs, tmp_path):
    project_dir, agent_dir = dirs
    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_output_pad() == 1

    manager.set_output_pad(0)
    await manager.flush()

    assert manager.get_output_pad() == 0
    saved = json.loads((tmp_path / "agent" / "settings.json").read_text())
    assert saved["outputPad"] == 0


def test_output_pad_treats_unsupported_values_as_default(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"outputPad": 2})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_output_pad() == 1


# -- markdown.mermaid ------------------------------------------------------------------


async def test_mermaid_defaults_to_streaming_and_persists(dirs, tmp_path):
    project_dir, agent_dir = dirs
    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_mermaid_rendering_mode() == "streaming"

    manager.set_mermaid_rendering_mode("final")
    await manager.flush()

    assert manager.get_mermaid_rendering_mode() == "final"
    saved = json.loads((tmp_path / "agent" / "settings.json").read_text())
    assert saved["markdown"]["mermaid"] == "final"


def test_mermaid_falls_back_to_streaming_for_unsupported_values(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"markdown": {"mermaid": "sometimes"}})

    assert SettingsManager.create(project_dir, agent_dir).get_mermaid_rendering_mode() == "streaming"


# -- shellCommandPrefix -----------------------------------------------------------------


def test_loads_shell_command_prefix_from_settings(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"shellCommandPrefix": "shopt -s expand_aliases"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_shell_command_prefix() == "shopt -s expand_aliases"


def test_shell_command_prefix_is_none_when_not_set(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "dark"})

    manager = SettingsManager.create(project_dir, agent_dir)

    assert manager.get_shell_command_prefix() is None


async def test_preserves_shell_command_prefix_when_saving_unrelated_settings(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"shellCommandPrefix": "shopt -s expand_aliases"})

    manager = SettingsManager.create(project_dir, agent_dir)
    manager.set_theme("light")
    await manager.flush()

    saved = json.loads(settings_path.read_text())
    assert saved["shellCommandPrefix"] == "shopt -s expand_aliases"
    assert saved["theme"] == "light"


# -- getSessionDir ------------------------------------------------------------------


def test_session_dir_is_none_when_not_set(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "dark"})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_session_dir() is None


def test_session_dir_returns_global_session_dir(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"sessionDir": "/tmp/sessions"})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_session_dir() == "/tmp/sessions"


def test_session_dir_project_overrides_global(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"sessionDir": "/global/sessions"})
    write_json(tmp_path / "project" / ".pi" / "settings.json", {"sessionDir": "./sessions"})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_session_dir() == "./sessions"


def test_session_dir_expands_tilde(dirs, tmp_path, monkeypatch):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"sessionDir": "~/sessions"})
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr("pi_coding_agent.utils.paths.Path.home", lambda: fake_home)
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_session_dir() == str(fake_home / "sessions")


# -- getShellPath --------------------------------------------------------------------


def test_shell_path_is_none_when_not_set(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"theme": "dark"})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_shell_path() is None


def test_shell_path_returns_absolute_path_unchanged(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"shellPath": "/bin/zsh"})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_shell_path() == "/bin/zsh"


def test_shell_path_expands_tilde(dirs, tmp_path, monkeypatch):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"shellPath": "~/.local/bin/agent-shell-sandbox"})
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr("pi_coding_agent.utils.paths.Path.home", lambda: fake_home)
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_shell_path() == str(fake_home / ".local" / "bin" / "agent-shell-sandbox")


def test_shell_path_expands_bare_tilde(dirs, tmp_path, monkeypatch):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"shellPath": "~"})
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr("pi_coding_agent.utils.paths.Path.home", lambda: fake_home)
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_shell_path() == str(fake_home)


# -- external edit preservation (ported from settings-manager-bug.test.ts) -----------


async def test_preserves_file_changes_to_packages_array_when_changing_unrelated_setting(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"theme": "dark", "packages": ["npm:pi-mcp-adapter"]})

    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_packages() == ["npm:pi-mcp-adapter"]

    current = json.loads(settings_path.read_text())
    current["packages"] = []
    settings_path.write_text(json.dumps(current, indent=2))
    assert json.loads(settings_path.read_text())["packages"] == []

    manager.set_theme("light")
    await manager.flush()

    saved = json.loads(settings_path.read_text())
    assert saved["packages"] == []
    assert saved["theme"] == "light"


async def test_preserves_file_changes_to_extensions_array_when_changing_unrelated_setting(dirs, tmp_path):
    project_dir, agent_dir = dirs
    settings_path = tmp_path / "agent" / "settings.json"
    write_json(settings_path, {"theme": "dark", "extensions": ["/old/extension.ts"]})

    manager = SettingsManager.create(project_dir, agent_dir)

    current = json.loads(settings_path.read_text())
    current["extensions"] = ["/new/extension.ts"]
    settings_path.write_text(json.dumps(current, indent=2))

    manager.set_default_thinking_level("high")
    await manager.flush()

    saved = json.loads(settings_path.read_text())
    assert saved["extensions"] == ["/new/extension.ts"]


async def test_preserves_external_project_settings_changes_when_updating_unrelated_project_field(dirs, tmp_path):
    project_dir, agent_dir = dirs
    project_settings_path = tmp_path / "project" / ".pi" / "settings.json"
    write_json(project_settings_path, {"extensions": ["./old-extension.ts"], "prompts": ["./old-prompt.md"]})

    manager = SettingsManager.create(project_dir, agent_dir)

    current = json.loads(project_settings_path.read_text())
    current["prompts"] = ["./new-prompt.md"]
    project_settings_path.write_text(json.dumps(current, indent=2))

    manager.set_project_extension_paths(["./updated-extension.ts"])
    await manager.flush()

    saved = json.loads(project_settings_path.read_text())
    assert saved["prompts"] == ["./new-prompt.md"]
    assert saved["extensions"] == ["./updated-extension.ts"]


async def test_in_memory_project_changes_override_external_changes_for_same_field(dirs, tmp_path):
    project_dir, agent_dir = dirs
    project_settings_path = tmp_path / "project" / ".pi" / "settings.json"
    write_json(project_settings_path, {"extensions": ["./initial-extension.ts"]})

    manager = SettingsManager.create(project_dir, agent_dir)

    current = json.loads(project_settings_path.read_text())
    current["extensions"] = ["./external-extension.ts"]
    project_settings_path.write_text(json.dumps(current, indent=2))

    manager.set_project_extension_paths(["./in-memory-extension.ts"])
    await manager.flush()

    saved = json.loads(project_settings_path.read_text())
    assert saved["extensions"] == ["./in-memory-extension.ts"]


# -- migration (settings.json shape upgrades; SettingsManager._migrate_settings) -------


def test_migrates_queue_mode_to_steering_mode(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"queueMode": "all"})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_steering_mode() == "all"


def test_does_not_overwrite_existing_steering_mode_with_queue_mode(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(
        tmp_path / "agent" / "settings.json",
        {"queueMode": "all", "steeringMode": "one-at-a-time"},
    )
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_steering_mode() == "one-at-a-time"


def test_migrates_legacy_websockets_boolean_to_transport(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"websockets": True})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_transport() == "websocket"


def test_migrates_legacy_websockets_false_to_sse(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"websockets": False})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_transport() == "sse"


def test_migrates_old_skills_object_format_to_array(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(
        tmp_path / "agent" / "settings.json",
        {"skills": {"enableSkillCommands": False, "customDirectories": ["/a", "/b"]}},
    )
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_skill_paths() == ["/a", "/b"]
    assert manager.get_enable_skill_commands() is False


def test_migrates_old_skills_object_format_with_empty_custom_directories(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(
        tmp_path / "agent" / "settings.json",
        {"skills": {"enableSkillCommands": True, "customDirectories": []}},
    )
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_skill_paths() == []
    assert manager.get_enable_skill_commands() is True


def test_migrates_retry_max_delay_ms_to_provider_max_retry_delay_ms(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(tmp_path / "agent" / "settings.json", {"retry": {"maxDelayMs": 5000}})
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_provider_retry_settings().get("maxRetryDelayMs") == 5000


def test_retry_max_delay_ms_does_not_override_existing_provider_setting(dirs, tmp_path):
    project_dir, agent_dir = dirs
    write_json(
        tmp_path / "agent" / "settings.json",
        {"retry": {"maxDelayMs": 5000, "provider": {"maxRetryDelayMs": 9000}}},
    )
    manager = SettingsManager.create(project_dir, agent_dir)
    assert manager.get_provider_retry_settings().get("maxRetryDelayMs") == 9000


def test_never_touches_real_home_directory(tmp_path, monkeypatch):
    """Sanity guard: in-memory SettingsManager path resolution must never read the real HOME."""

    def _boom():
        raise AssertionError("real HOME touched")

    monkeypatch.setattr("pi_coding_agent.utils.paths.Path.home", _boom)
    manager = SettingsManager.in_memory({"theme": "dark"})
    assert manager.get_theme() == "dark"
