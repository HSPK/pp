"""Coverage tests for `core/settings_manager.py`.

Covers: `_parse_http_idle_timeout_ms` branches, `_parse_timeout_setting`
error path, `InMemorySettingsStorage`, `set_project_trusted`, `reload`,
`apply_overrides`, and all setter/getter pairs that were uncovered.
"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.core.settings_manager import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    SettingsManager,
    SettingsManagerCreateOptions,
    _parse_http_idle_timeout_ms,
    _parse_timeout_setting,
    deep_merge_objects,
)

# ---------------------------------------------------------------------------
# _parse_http_idle_timeout_ms
# ---------------------------------------------------------------------------


def test_parse_disabled_string_returns_zero():
    assert _parse_http_idle_timeout_ms("disabled") == 0


def test_parse_disabled_string_case_insensitive():
    assert _parse_http_idle_timeout_ms("DISABLED") == 0
    assert _parse_http_idle_timeout_ms("Disabled") == 0


def test_parse_empty_string_returns_none():
    assert _parse_http_idle_timeout_ms("") is None
    assert _parse_http_idle_timeout_ms("   ") is None


def test_parse_numeric_string_returns_int():
    assert _parse_http_idle_timeout_ms("5000") == 5000
    assert _parse_http_idle_timeout_ms("  3000  ") == 3000


def test_parse_non_numeric_string_returns_none():
    assert _parse_http_idle_timeout_ms("not-a-number") is None


def test_parse_bool_returns_none():
    assert _parse_http_idle_timeout_ms(True) is None
    assert _parse_http_idle_timeout_ms(False) is None


def test_parse_none_returns_none():
    assert _parse_http_idle_timeout_ms(None) is None


def test_parse_nan_returns_none():
    assert _parse_http_idle_timeout_ms(float("nan")) is None


def test_parse_inf_returns_none():
    assert _parse_http_idle_timeout_ms(float("inf")) is None
    assert _parse_http_idle_timeout_ms(float("-inf")) is None


def test_parse_negative_returns_none():
    assert _parse_http_idle_timeout_ms(-1) is None


def test_parse_positive_int_returns_int():
    assert _parse_http_idle_timeout_ms(300_000) == 300_000


def test_parse_zero_returns_zero():
    assert _parse_http_idle_timeout_ms(0) == 0


# ---------------------------------------------------------------------------
# _parse_timeout_setting
# ---------------------------------------------------------------------------


def test_parse_timeout_setting_valid_returns_value():
    assert _parse_timeout_setting(5000, "httpIdleTimeoutMs") == 5000


def test_parse_timeout_setting_none_returns_none():
    assert _parse_timeout_setting(None, "httpIdleTimeoutMs") is None


def test_parse_timeout_setting_invalid_raises():
    with pytest.raises(ValueError, match="Invalid httpIdleTimeoutMs"):
        _parse_timeout_setting("bad-value", "httpIdleTimeoutMs")


def test_parse_timeout_setting_disabled_returns_zero():
    assert _parse_timeout_setting("disabled", "httpIdleTimeoutMs") == 0


# ---------------------------------------------------------------------------
# deep_merge_objects
# ---------------------------------------------------------------------------


def test_deep_merge_none_override_skipped():
    base = {"a": 1, "b": {"c": 2}}
    overrides = {"a": None, "b": {"c": None, "d": 3}}
    result = deep_merge_objects(base, overrides)
    assert result["a"] == 1  # None overrides are skipped
    assert result["b"]["c"] == 2
    assert result["b"]["d"] == 3


def test_deep_merge_nested_dicts():
    base = {"retry": {"enabled": True, "maxRetries": 3}}
    overrides = {"retry": {"maxRetries": 5}}
    result = deep_merge_objects(base, overrides)
    assert result["retry"]["enabled"] is True
    assert result["retry"]["maxRetries"] == 5


# ---------------------------------------------------------------------------
# SettingsManager.in_memory factory
# ---------------------------------------------------------------------------


def make_manager(settings: dict | None = None, **options) -> SettingsManager:
    opts = SettingsManagerCreateOptions(**options) if options else None
    return SettingsManager.in_memory(settings, opts)


def test_in_memory_manager_loads_initial_settings():
    mgr = make_manager({"theme": "dark"})
    assert mgr.get_theme() == "dark"


def test_in_memory_manager_empty_by_default():
    mgr = make_manager()
    assert mgr.get_theme() is None


# ---------------------------------------------------------------------------
# set_project_trusted
# ---------------------------------------------------------------------------


def test_set_project_trusted_no_op_when_same():
    mgr = make_manager({"theme": "dark"})
    before = mgr.is_project_trusted()
    mgr.set_project_trusted(before)
    assert mgr.is_project_trusted() == before


def test_set_project_trusted_false_clears_project_settings():
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _: json.dumps({"theme": "dark"}))
    storage.with_lock("project", lambda _: json.dumps({"shellPath": "/bin/zsh"}))

    mgr = SettingsManager.from_storage(storage)
    assert mgr.get_shell_path() is not None

    mgr.set_project_trusted(False)
    assert mgr.is_project_trusted() is False
    assert mgr.get_shell_path() is None  # project settings cleared


def test_set_project_trusted_true_reloads_project():
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _: json.dumps({}))
    storage.with_lock("project", lambda _: json.dumps({"quietStartup": True}))

    opts = SettingsManagerCreateOptions(project_trusted=False)
    mgr = SettingsManager.from_storage(storage, opts)
    assert not mgr.get_quiet_startup()

    mgr.set_project_trusted(True)
    assert mgr.is_project_trusted() is True
    assert mgr.get_quiet_startup() is True


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------


async def test_reload_updates_global_settings():
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _: json.dumps({"theme": "dark"}))

    mgr = SettingsManager.from_storage(storage)
    assert mgr.get_theme() == "dark"

    # Change the storage externally.
    storage.with_lock("global", lambda _: json.dumps({"theme": "light"}))
    await mgr.reload()
    assert mgr.get_theme() == "light"


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


def test_apply_overrides_adds_values():
    mgr = make_manager({"theme": "dark"})
    mgr.apply_overrides({"quietStartup": True})
    assert mgr.get_quiet_startup() is True
    assert mgr.get_theme() == "dark"  # not affected


def test_apply_overrides_overrides_existing():
    mgr = make_manager({"theme": "dark"})
    mgr.apply_overrides({"theme": "light"})
    assert mgr.get_theme() == "light"


# ---------------------------------------------------------------------------
# Setters and getters for settings fields
# ---------------------------------------------------------------------------


def test_set_and_get_last_changelog_version():
    mgr = make_manager()
    assert mgr.get_last_changelog_version() is None
    mgr.set_last_changelog_version("1.2.3")
    assert mgr.get_last_changelog_version() == "1.2.3"


def test_set_and_get_default_model():
    mgr = make_manager()
    mgr.set_default_model("claude-opus-4-8")
    assert mgr.get_default_model() == "claude-opus-4-8"


def test_set_default_model_and_provider():
    mgr = make_manager()
    mgr.set_default_model_and_provider("anthropic", "claude-sonnet-4-5")
    assert mgr.get_default_provider() == "anthropic"
    assert mgr.get_default_model() == "claude-sonnet-4-5"


def test_set_and_get_follow_up_mode():
    mgr = make_manager()
    assert mgr.get_follow_up_mode() == "one-at-a-time"
    mgr.set_follow_up_mode("all")
    assert mgr.get_follow_up_mode() == "all"


def test_get_theme_with_slash_returns_none():
    # A theme value containing "/" is a path-based theme, not a named theme.
    mgr = make_manager({"theme": "/path/to/theme"})
    assert mgr.get_theme() is None
    assert mgr.get_theme_setting() == "/path/to/theme"


def test_get_http_idle_timeout_ms_default():
    mgr = make_manager()
    assert mgr.get_http_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS


def test_get_http_idle_timeout_ms_from_string():
    mgr = make_manager({"httpIdleTimeoutMs": "10000"})
    assert mgr.get_http_idle_timeout_ms() == 10000


def test_get_http_idle_timeout_ms_disabled():
    mgr = make_manager({"httpIdleTimeoutMs": "disabled"})
    assert mgr.get_http_idle_timeout_ms() == 0


def test_set_http_idle_timeout_ms_valid():
    mgr = make_manager()
    mgr.set_http_idle_timeout_ms(10000)
    assert mgr.get_http_idle_timeout_ms() == 10000


def test_set_http_idle_timeout_ms_invalid_raises():
    mgr = make_manager()
    with pytest.raises(ValueError):
        mgr.set_http_idle_timeout_ms(-1)
    with pytest.raises(ValueError):
        mgr.set_http_idle_timeout_ms(float("nan"))
    with pytest.raises(ValueError):
        mgr.set_http_idle_timeout_ms(float("inf"))


def test_get_provider_retry_settings_defaults():
    mgr = make_manager()
    s = mgr.get_provider_retry_settings()
    assert s["maxRetryDelayMs"] == 60000
    assert s["timeoutMs"] is None
    assert s["maxRetries"] is None


def test_get_websocket_connect_timeout_ms_none_by_default():
    mgr = make_manager()
    assert mgr.get_websocket_connect_timeout_ms() is None


def test_get_websocket_connect_timeout_ms_from_settings():
    mgr = make_manager({"websocketConnectTimeoutMs": 5000})
    assert mgr.get_websocket_connect_timeout_ms() == 5000


def test_get_external_editor_env_visual(monkeypatch):
    monkeypatch.setenv("VISUAL", "vim")
    monkeypatch.delenv("EDITOR", raising=False)
    mgr = make_manager()
    assert mgr.get_external_editor_command() == "vim"


def test_get_external_editor_env_editor(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "emacs")
    mgr = make_manager()
    assert mgr.get_external_editor_command() == "emacs"


def test_get_external_editor_fallback_to_nano(monkeypatch):
    import os

    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    mgr = make_manager()
    result = mgr.get_external_editor_command()
    expected = "notepad" if os.name == "nt" else "nano"
    assert result == expected


def test_get_external_editor_configured_value_takes_precedence(monkeypatch):
    monkeypatch.setenv("VISUAL", "vim")
    mgr = make_manager({"externalEditor": "nvim"})
    assert mgr.get_external_editor_command() == "nvim"


def test_set_and_get_hide_thinking_block():
    mgr = make_manager()
    assert mgr.get_hide_thinking_block() is False
    mgr.set_hide_thinking_block(True)
    assert mgr.get_hide_thinking_block() is True


def test_set_and_get_show_cache_miss_notices():
    mgr = make_manager()
    assert mgr.get_show_cache_miss_notices() is False
    mgr.set_show_cache_miss_notices(True)
    assert mgr.get_show_cache_miss_notices() is True


def test_set_and_get_shell_path(tmp_path):
    mgr = make_manager()
    assert mgr.get_shell_path() is None
    mgr.set_shell_path("/bin/zsh")
    assert mgr.get_shell_path() is not None


def test_set_and_get_quiet_startup():
    mgr = make_manager()
    assert mgr.get_quiet_startup() is False
    mgr.set_quiet_startup(True)
    assert mgr.get_quiet_startup() is True


def test_get_default_project_trust_defaults_to_ask():
    mgr = make_manager()
    assert mgr.get_default_project_trust() == "ask"


def test_set_default_project_trust_always():
    mgr = make_manager()
    mgr.set_default_project_trust("always")
    assert mgr.get_default_project_trust() == "always"


def test_set_default_project_trust_never():
    mgr = make_manager()
    mgr.set_default_project_trust("never")
    assert mgr.get_default_project_trust() == "never"


def test_set_and_get_shell_command_prefix():
    mgr = make_manager()
    assert mgr.get_shell_command_prefix() is None
    mgr.set_shell_command_prefix("sudo")
    assert mgr.get_shell_command_prefix() == "sudo"


def test_set_shell_command_prefix_none():
    mgr = make_manager({"shellCommandPrefix": "sudo"})
    mgr.set_shell_command_prefix(None)
    assert mgr.get_shell_command_prefix() is None


def test_set_and_get_npm_command():
    mgr = make_manager()
    assert mgr.get_npm_command() is None
    mgr.set_npm_command(["pnpm"])
    assert mgr.get_npm_command() == ["pnpm"]


def test_set_npm_command_none():
    mgr = make_manager({"npmCommand": ["yarn"]})
    mgr.set_npm_command(None)
    assert mgr.get_npm_command() is None


def test_set_and_get_collapse_changelog():
    mgr = make_manager()
    assert mgr.get_collapse_changelog() is False
    mgr.set_collapse_changelog(True)
    assert mgr.get_collapse_changelog() is True


def test_set_and_get_enable_install_telemetry():
    mgr = make_manager()
    assert mgr.get_enable_install_telemetry() is True
    mgr.set_enable_install_telemetry(False)
    assert mgr.get_enable_install_telemetry() is False


def test_set_enable_analytics_generates_tracking_id():
    mgr = make_manager()
    assert mgr.get_tracking_id() is None
    assert mgr.get_enable_analytics() is False
    mgr.set_enable_analytics(True)
    assert mgr.get_enable_analytics() is True
    assert mgr.get_tracking_id() is not None


def test_set_enable_analytics_keeps_existing_tracking_id():
    mgr = make_manager({"trackingId": "existing-id"})
    mgr.set_enable_analytics(True)
    assert mgr.get_tracking_id() == "existing-id"


def test_set_enable_analytics_disabling_does_not_clear_tracking_id():
    mgr = make_manager({"enableAnalytics": True, "trackingId": "my-id"})
    mgr.set_enable_analytics(False)
    assert mgr.get_tracking_id() == "my-id"


def test_set_and_get_packages():
    mgr = make_manager()
    assert mgr.get_packages() == []
    mgr.set_packages([{"name": "ext1"}])
    assert mgr.get_packages() == [{"name": "ext1"}]


def test_set_project_packages():
    mgr = make_manager()
    mgr.set_project_packages([{"name": "local-ext"}])
    assert any(p.get("name") == "local-ext" for p in mgr.get_packages())


def test_set_and_get_extension_paths():
    mgr = make_manager()
    assert mgr.get_extension_paths() == []
    mgr.set_extension_paths(["/ext/a", "/ext/b"])
    assert mgr.get_extension_paths() == ["/ext/a", "/ext/b"]


def test_set_project_extension_paths():
    mgr = make_manager()
    mgr.set_project_extension_paths(["/project/ext"])
    assert "/project/ext" in mgr.get_extension_paths()


def test_set_and_get_skill_paths():
    mgr = make_manager()
    assert mgr.get_skill_paths() == []
    mgr.set_skill_paths(["/skills/a"])
    assert mgr.get_skill_paths() == ["/skills/a"]


def test_set_project_skill_paths():
    mgr = make_manager()
    mgr.set_project_skill_paths(["/project/skills"])
    assert "/project/skills" in mgr.get_skill_paths()


def test_set_and_get_prompt_template_paths():
    mgr = make_manager()
    assert mgr.get_prompt_template_paths() == []
    mgr.set_prompt_template_paths(["/prompts/a"])
    assert mgr.get_prompt_template_paths() == ["/prompts/a"]


def test_set_project_prompt_template_paths():
    mgr = make_manager()
    mgr.set_project_prompt_template_paths(["/project/prompts"])
    assert "/project/prompts" in mgr.get_prompt_template_paths()


def test_set_and_get_theme_paths():
    mgr = make_manager()
    assert mgr.get_theme_paths() == []
    mgr.set_theme_paths(["/themes/dark"])
    assert mgr.get_theme_paths() == ["/themes/dark"]


def test_set_project_theme_paths():
    mgr = make_manager()
    mgr.set_project_theme_paths(["/project/themes"])
    assert "/project/themes" in mgr.get_theme_paths()


def test_set_and_get_enable_skill_commands():
    mgr = make_manager()
    assert mgr.get_enable_skill_commands() is True
    mgr.set_enable_skill_commands(False)
    assert mgr.get_enable_skill_commands() is False


def test_get_show_images_default_true():
    mgr = make_manager()
    assert mgr.get_show_images() is True


def test_set_and_get_show_images():
    mgr = make_manager()
    mgr.set_show_images(False)
    assert mgr.get_show_images() is False


def test_get_image_width_cells_default():
    mgr = make_manager()
    assert mgr.get_image_width_cells() == 60


def test_set_and_get_image_width_cells():
    mgr = make_manager()
    mgr.set_image_width_cells(80)
    assert mgr.get_image_width_cells() == 80


def test_get_image_width_cells_invalid_returns_default():
    mgr = make_manager({"terminal": {"imageWidthCells": "not-a-number"}})
    assert mgr.get_image_width_cells() == 60


def test_get_image_width_cells_nan_returns_default():
    mgr = make_manager({"terminal": {"imageWidthCells": float("nan")}})
    assert mgr.get_image_width_cells() == 60


def test_get_clear_on_shrink_default_false(monkeypatch):
    monkeypatch.delenv("PI_CLEAR_ON_SHRINK", raising=False)
    mgr = make_manager()
    assert mgr.get_clear_on_shrink() is False


def test_get_clear_on_shrink_from_env(monkeypatch):
    monkeypatch.setenv("PI_CLEAR_ON_SHRINK", "1")
    mgr = make_manager()
    assert mgr.get_clear_on_shrink() is True


def test_set_and_get_clear_on_shrink():
    mgr = make_manager()
    mgr.set_clear_on_shrink(True)
    assert mgr.get_clear_on_shrink() is True


def test_get_show_terminal_progress_default():
    mgr = make_manager()
    assert mgr.get_show_terminal_progress() is False


def test_set_and_get_show_terminal_progress():
    mgr = make_manager()
    mgr.set_show_terminal_progress(True)
    assert mgr.get_show_terminal_progress() is True


def test_get_tui_mode_default():
    mgr = make_manager()
    assert mgr.get_tui_mode() == "regular"


def test_set_and_get_tui_mode():
    mgr = make_manager()
    mgr.set_tui_mode("fullscreen")
    assert mgr.get_tui_mode() == "fullscreen"


def test_get_fullscreen_exit_output_default():
    mgr = make_manager()
    assert mgr.get_fullscreen_exit_output() == "transcript"


def test_set_and_get_fullscreen_exit_output():
    mgr = make_manager()
    mgr.set_fullscreen_exit_output("resume-hint")
    assert mgr.get_fullscreen_exit_output() == "resume-hint"


def test_get_fullscreen_scrollbar_default():
    mgr = make_manager()
    assert mgr.get_fullscreen_scrollbar() == "auto"


def test_set_and_get_fullscreen_scrollbar():
    mgr = make_manager()
    mgr.set_fullscreen_scrollbar("always")
    assert mgr.get_fullscreen_scrollbar() == "always"
    mgr.set_fullscreen_scrollbar("hidden")
    assert mgr.get_fullscreen_scrollbar() == "hidden"


def test_get_image_auto_resize_default():
    mgr = make_manager()
    assert mgr.get_image_auto_resize() is True


def test_set_and_get_image_auto_resize():
    mgr = make_manager()
    mgr.set_image_auto_resize(False)
    assert mgr.get_image_auto_resize() is False


def test_get_block_images_default():
    mgr = make_manager()
    assert mgr.get_block_images() is False


def test_set_and_get_block_images():
    mgr = make_manager()
    mgr.set_block_images(True)
    assert mgr.get_block_images() is True


def test_get_enabled_models_none_by_default():
    mgr = make_manager()
    assert mgr.get_enabled_models() is None


def test_set_and_get_enabled_models():
    mgr = make_manager()
    mgr.set_enabled_models(["claude-*", "gpt-4o"])
    assert mgr.get_enabled_models() == ["claude-*", "gpt-4o"]


def test_set_enabled_models_none():
    mgr = make_manager({"enabledModels": ["claude-*"]})
    mgr.set_enabled_models(None)
    assert mgr.get_enabled_models() is None


def test_get_double_escape_action_default():
    mgr = make_manager()
    assert mgr.get_double_escape_action() == "tree"


def test_set_and_get_double_escape_action():
    mgr = make_manager()
    mgr.set_double_escape_action("fork")
    assert mgr.get_double_escape_action() == "fork"


def test_get_tree_filter_mode_default():
    mgr = make_manager()
    assert mgr.get_tree_filter_mode() == "default"


def test_set_and_get_tree_filter_mode():
    mgr = make_manager()
    mgr.set_tree_filter_mode("all")
    assert mgr.get_tree_filter_mode() == "all"


def test_get_show_hardware_cursor_default_false(monkeypatch):
    monkeypatch.delenv("PI_HARDWARE_CURSOR", raising=False)
    mgr = make_manager()
    assert mgr.get_show_hardware_cursor() is False


def test_get_show_hardware_cursor_from_env(monkeypatch):
    monkeypatch.setenv("PI_HARDWARE_CURSOR", "1")
    mgr = make_manager()
    assert mgr.get_show_hardware_cursor() is True


def test_set_and_get_show_hardware_cursor():
    mgr = make_manager()
    mgr.set_show_hardware_cursor(True)
    assert mgr.get_show_hardware_cursor() is True


def test_get_editor_padding_x_default():
    mgr = make_manager()
    assert mgr.get_editor_padding_x() == 0


def test_set_and_get_editor_padding_x():
    mgr = make_manager()
    mgr.set_editor_padding_x(2)
    assert mgr.get_editor_padding_x() == 2


def test_set_editor_padding_x_clamped():
    mgr = make_manager()
    mgr.set_editor_padding_x(10)  # exceeds max of 3
    assert mgr.get_editor_padding_x() == 3
    mgr.set_editor_padding_x(-5)  # below min of 0
    assert mgr.get_editor_padding_x() == 0


def test_get_output_pad_default():
    mgr = make_manager()
    assert mgr.get_output_pad() == 1


def test_set_and_get_output_pad():
    mgr = make_manager()
    mgr.set_output_pad(0)
    assert mgr.get_output_pad() == 0


def test_get_autocomplete_max_visible_default():
    mgr = make_manager()
    assert mgr.get_autocomplete_max_visible() == 5


def test_set_and_get_autocomplete_max_visible():
    mgr = make_manager()
    mgr.set_autocomplete_max_visible(10)
    assert mgr.get_autocomplete_max_visible() == 10


def test_set_autocomplete_max_visible_clamped():
    mgr = make_manager()
    mgr.set_autocomplete_max_visible(1)  # below min of 3
    assert mgr.get_autocomplete_max_visible() == 3
    mgr.set_autocomplete_max_visible(100)  # above max of 20
    assert mgr.get_autocomplete_max_visible() == 20


def test_get_code_block_indent_default():
    mgr = make_manager()
    assert mgr.get_code_block_indent() == "  "


def test_get_mermaid_rendering_mode_default():
    mgr = make_manager()
    assert mgr.get_mermaid_rendering_mode() == "streaming"


def test_set_and_get_mermaid_rendering_mode():
    mgr = make_manager()
    mgr.set_mermaid_rendering_mode("off")
    assert mgr.get_mermaid_rendering_mode() == "off"
    mgr.set_mermaid_rendering_mode("final")
    assert mgr.get_mermaid_rendering_mode() == "final"


def test_get_warnings_default_empty():
    mgr = make_manager()
    assert mgr.get_warnings() == {}


def test_set_and_get_warnings():
    mgr = make_manager()
    mgr.set_warnings({"deprecation-abc": True})
    assert mgr.get_warnings() == {"deprecation-abc": True}


def test_get_thinking_budgets_none_by_default():
    mgr = make_manager()
    assert mgr.get_thinking_budgets() is None


def test_get_compaction_settings_defaults():
    mgr = make_manager()
    s = mgr.get_compaction_settings()
    assert s["enabled"] is True
    assert s["reserveTokens"] == 16384
    assert s["keepRecentTokens"] == 20000


def test_set_and_get_compaction_enabled():
    mgr = make_manager()
    mgr.set_compaction_enabled(False)
    assert mgr.get_compaction_enabled() is False


def test_get_branch_summary_settings_defaults():
    mgr = make_manager()
    s = mgr.get_branch_summary_settings()
    assert s["reserveTokens"] == 16384
    assert s["skipPrompt"] is False


def test_get_retry_settings_defaults():
    mgr = make_manager()
    s = mgr.get_retry_settings()
    assert s["enabled"] is True
    assert s["maxRetries"] == 3
    assert s["baseDelayMs"] == 2000


def test_set_and_get_retry_enabled():
    mgr = make_manager()
    mgr.set_retry_enabled(False)
    assert mgr.get_retry_enabled() is False


# ---------------------------------------------------------------------------
# drain_errors / error recording
# ---------------------------------------------------------------------------


def test_drain_errors_returns_and_clears():
    # Load from invalid JSON to trigger an error.
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage()
    storage._global = "not-valid-json"  # type: ignore[attr-defined]

    mgr2 = SettingsManager.from_storage(storage)
    errors = mgr2.drain_errors()
    assert len(errors) > 0
    # Second drain is empty.
    assert mgr2.drain_errors() == []


# ---------------------------------------------------------------------------
# Transport, steeringMode accessors
# ---------------------------------------------------------------------------


def test_get_transport_default():
    mgr = make_manager()
    assert mgr.get_transport() == "auto"


def test_set_and_get_transport():
    mgr = make_manager()
    mgr.set_transport("sse")
    assert mgr.get_transport() == "sse"


def test_get_steering_mode_default():
    mgr = make_manager()
    assert mgr.get_steering_mode() == "one-at-a-time"


def test_set_and_get_steering_mode():
    mgr = make_manager()
    mgr.set_steering_mode("all")
    assert mgr.get_steering_mode() == "all"


def test_get_session_dir_none_by_default():
    mgr = make_manager()
    assert mgr.get_session_dir() is None


# ---------------------------------------------------------------------------
# project_trusted write guard
# ---------------------------------------------------------------------------


def test_set_project_packages_raises_when_untrusted():
    opts = SettingsManagerCreateOptions(project_trusted=False)
    mgr = SettingsManager.in_memory({}, opts)
    with pytest.raises(RuntimeError, match="not trusted"):
        mgr.set_project_packages([])


# ---------------------------------------------------------------------------
# Global settings load error suppresses save
# ---------------------------------------------------------------------------


def test_save_skipped_when_global_load_error():
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage()
    storage._global = "!!invalid json!!"  # type: ignore[attr-defined]

    mgr = SettingsManager.from_storage(storage)
    # set_theme will attempt to _save(); with a global load error it should not raise.
    mgr.set_theme("dark")
    # No exception means the save was suppressed gracefully.
