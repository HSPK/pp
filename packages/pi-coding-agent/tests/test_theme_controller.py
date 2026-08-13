"""Python port of `packages/coding-agent/src/modes/interactive/theme/theme-controller.ts`
and of `packages/coding-agent/test/theme-controller.test.ts`.

The tests below the "initial theme setting" heading are the ports of upstream's
test file. The rest predate it: they pin behaviour the TypeScript source
encodes but never tested, because this port previously had no counterpart at
all and the interactive mode did theme handling inline (dropping auto theme
pairs at startup and never enabling terminal color-scheme sync).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pi_coding_agent.modes.interactive.theme.theme import RgbColor, TerminalTheme, init_theme, load_theme
from pi_coding_agent.modes.interactive.theme.theme import theme as current_theme
from pi_coding_agent.modes.interactive.theme.theme_controller import InteractiveThemeController


class FakeUi:
    """The `TUI` surface `InteractiveThemeController` touches."""

    def __init__(
        self,
        *,
        background: RgbColor | None = None,
        color_scheme: TerminalTheme | None = None,
    ) -> None:
        self.background = background
        self.color_scheme = color_scheme
        self.invalidate_count = 0
        self.render_count = 0
        self.notifications_enabled: bool | None = None
        self.notification_calls: list[bool] = []
        self.listeners: list[Callable[[TerminalTheme], None]] = []
        self.unsubscribe_count = 0

    async def query_terminal_background_color(self, timeout_ms: float) -> RgbColor | None:
        return self.background

    async def query_terminal_color_scheme(self, timeout_ms: float) -> TerminalTheme | None:
        return self.color_scheme

    def invalidate(self) -> None:
        self.invalidate_count += 1

    def request_render(self) -> None:
        self.render_count += 1

    def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        self.notifications_enabled = enabled
        self.notification_calls.append(enabled)

    def on_terminal_color_scheme_change(self, listener: Callable[[TerminalTheme], None]) -> Callable[[], None]:
        self.listeners.append(listener)

        def _unsubscribe() -> None:
            self.unsubscribe_count += 1
            if listener in self.listeners:
                self.listeners.remove(listener)

        return _unsubscribe

    def emit_color_scheme(self, terminal_theme: TerminalTheme) -> None:
        for listener in list(self.listeners):
            listener(terminal_theme)


class FakeSettings:
    def __init__(self, theme_setting: str | None = None) -> None:
        self.theme_setting = theme_setting
        self.flush_count = 0
        self.set_calls: list[str] = []

    def get_theme_setting(self) -> str | None:
        return self.theme_setting

    def set_theme(self, theme: str) -> None:
        self.theme_setting = theme
        self.set_calls.append(theme)

    async def flush(self) -> None:
        self.flush_count += 1


def _make(
    theme_setting: str | None = None,
    initial_theme_setting: str | None = None,
    **ui_kwargs: Any,
) -> tuple[InteractiveThemeController, FakeUi, FakeSettings, list[str], list[int]]:
    ui = FakeUi(**ui_kwargs)
    settings = FakeSettings(theme_setting)
    errors: list[str] = []
    changed: list[int] = []
    controller = InteractiveThemeController(
        ui,  # type: ignore[arg-type]
        lambda: settings,  # type: ignore[arg-type,return-value]
        errors.append,
        lambda: changed.append(1),
        initial_theme_setting,
    )
    return controller, ui, settings, errors, changed


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Detection falls back to "dark" with no COLORFGBG hint; make that explicit
    # so an inherited terminal environment cannot flip the expectations.
    monkeypatch.delenv("COLORFGBG", raising=False)
    yield
    init_theme("dark")


# --------------------------------------------------------------------------
# constructor
# --------------------------------------------------------------------------


def test_constructor_resolves_an_auto_pair_against_the_env_detected_background(monkeypatch: pytest.MonkeyPatch):
    """`resolveThemeSetting(getThemeSetting(), detectTerminalBackgroundFromEnv().theme)`."""
    monkeypatch.setenv("COLORFGBG", "0;15")  # white background -> light terminal
    controller, _ui, _settings, _errors, _changed = _make("dark/light")

    assert controller.get_terminal_theme() == "light"
    # `"<light>/<dark>"`, so a light terminal picks the theme named "dark".
    assert current_theme.name == "dark"


def test_constructor_loads_a_plain_setting_verbatim():
    _controller, _ui, _settings, _errors, _changed = _make("light")

    assert current_theme.name == "light"


def test_constructor_falls_back_to_the_default_theme_without_a_setting():
    controller, _ui, _settings, _errors, _changed = _make(None)

    assert controller.get_terminal_theme() == "dark"
    assert current_theme.name == "dark"


def test_constructor_binds_a_terminal_color_scheme_listener():
    _controller, ui, _settings, _errors, _changed = _make("light/dark")

    assert len(ui.listeners) == 1
    # Notifications are only turned on once auto-sync is enabled, which happens
    # in `applyFromSettings`, not in the constructor.
    assert ui.notifications_enabled is None


# --------------------------------------------------------------------------
# apply_from_settings
# --------------------------------------------------------------------------


def test_apply_from_settings_enables_auto_sync_for_an_auto_pair():
    controller, ui, settings, _errors, _changed = _make("light/dark", color_scheme="light")

    asyncio.run(controller.apply_from_settings())

    assert controller.get_terminal_theme() == "light"
    assert current_theme.name == "light"
    assert ui.notifications_enabled is True
    assert settings.set_calls == []
    assert settings.flush_count == 0


def test_apply_from_settings_disables_auto_sync_for_a_plain_setting():
    controller, ui, _settings, _errors, _changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    assert ui.notifications_enabled is True

    _settings_obj = controller.settings_manager
    _settings_obj.theme_setting = "light"  # type: ignore[attr-defined]
    asyncio.run(controller.apply_from_settings())

    assert ui.notifications_enabled is False
    assert current_theme.name == "light"


def test_apply_from_settings_persists_a_high_confidence_detection():
    """No `theme` setting: probe the terminal and write a high-confidence result."""
    controller, _ui, settings, _errors, _changed = _make(None, background=RgbColor(r=255, g=255, b=255))

    asyncio.run(controller.apply_from_settings())

    assert controller.get_terminal_theme() == "light"
    assert current_theme.name == "light"
    assert settings.set_calls == ["light"]
    assert settings.flush_count == 1


def test_apply_from_settings_does_not_persist_a_low_confidence_detection():
    """No OSC 11 answer and no COLORFGBG: `confidence` is "low", so nothing is saved."""
    controller, _ui, settings, _errors, _changed = _make(None, background=None)

    asyncio.run(controller.apply_from_settings())

    assert current_theme.name == "dark"
    assert settings.set_calls == []
    assert settings.flush_count == 0


def test_apply_from_settings_reports_an_unloadable_theme_and_falls_back_to_dark():
    controller, _ui, _settings, errors, _changed = _make("no-such-theme")

    asyncio.run(controller.apply_from_settings())

    assert len(errors) == 1
    assert errors[0].startswith('Failed to load theme "no-such-theme": ')
    assert errors[0].endswith("Fell back to dark theme.")
    assert current_theme.name == "dark"


# --------------------------------------------------------------------------
# auto-sync
# --------------------------------------------------------------------------


def test_a_terminal_color_scheme_change_switches_the_theme_while_auto_sync_is_on():
    controller, ui, _settings, _errors, changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    assert current_theme.name == "dark"
    changed.clear()

    ui.emit_color_scheme("light")

    assert controller.get_terminal_theme() == "light"
    assert current_theme.name == "light"
    assert len(changed) == 1


def test_a_repeat_of_the_active_theme_does_not_re_apply_it():
    controller, ui, _settings, _errors, changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    changed.clear()

    ui.emit_color_scheme("dark")

    assert changed == []


def test_a_terminal_color_scheme_change_is_ignored_while_auto_sync_is_off():
    controller, ui, _settings, _errors, changed = _make("light")
    asyncio.run(controller.apply_from_settings())
    changed.clear()

    ui.emit_color_scheme("dark")

    assert current_theme.name == "light"
    assert changed == []


def test_a_color_scheme_change_turns_auto_sync_off_when_the_setting_stopped_being_a_pair():
    controller, ui, settings, _errors, _changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    assert ui.notifications_enabled is True

    settings.theme_setting = "light"
    ui.emit_color_scheme("light")

    assert ui.notifications_enabled is False
    # The theme is left alone; only the sync is torn down.
    assert current_theme.name == "dark"


def test_disable_auto_sync_is_idempotent():
    controller, ui, _settings, _errors, _changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    ui.notification_calls.clear()

    controller.disable_auto_sync()
    controller.disable_auto_sync()

    assert ui.notification_calls == [False]


# --------------------------------------------------------------------------
# set_theme_name / set_theme_instance / preview
# --------------------------------------------------------------------------


def test_set_theme_name_disables_auto_sync_and_applies_the_theme():
    controller, ui, _settings, _errors, changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    changed.clear()

    result = controller.set_theme_name("light")

    assert result.success is True
    assert current_theme.name == "light"
    assert ui.notifications_enabled is False
    assert len(changed) == 1


def test_set_theme_name_does_not_show_an_error_unless_asked():
    controller, _ui, _settings, errors, _changed = _make("light")

    result = controller.set_theme_name("no-such-theme")

    assert result.success is False
    assert errors == []
    assert current_theme.name == "dark"

    controller.set_theme_name("no-such-theme", True)
    assert len(errors) == 1


def test_set_theme_instance_installs_the_instance_and_disables_auto_sync():
    controller, ui, _settings, _errors, changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    changed.clear()
    instance = load_theme("light")
    instance.name = "in-memory"

    result = controller.set_theme_instance(instance)

    assert result.success is True
    assert current_theme.name == "in-memory"
    assert ui.notifications_enabled is False
    assert len(changed) == 1


def test_preview_resolves_an_auto_pair_and_renders_without_notifying():
    controller, ui, _settings, _errors, changed = _make("light")
    ui.render_count = 0
    changed.clear()

    controller.preview("light/dark")

    # Terminal is dark, so the pair's dark side ("dark") is previewed.
    assert current_theme.name == "dark"
    assert ui.render_count == 1
    # Preview must not run the `onChanged` callback; it is not a commit.
    assert changed == []


def test_preview_of_an_unloadable_theme_does_not_render():
    controller, ui, _settings, _errors, _changed = _make("light")
    ui.render_count = 0

    controller.preview("no-such-theme")

    assert ui.render_count == 0


def test_preview_without_a_resolvable_name_is_a_no_op():
    controller, ui, _settings, _errors, _changed = _make("light")
    ui.render_count = 0
    controller._active_theme_name = None

    controller.preview("a/b/c")

    assert ui.render_count == 0
    assert current_theme.name == "light"


# --------------------------------------------------------------------------
# rebind_tui
# --------------------------------------------------------------------------


def test_rebind_tui_replaces_the_listener_and_restores_the_notification_state():
    controller, ui, _settings, _errors, _changed = _make("light/dark", color_scheme="dark")
    asyncio.run(controller.apply_from_settings())
    assert len(ui.listeners) == 1
    ui.notification_calls.clear()

    controller.rebind_tui()

    assert ui.unsubscribe_count == 1
    assert len(ui.listeners) == 1
    # Re-asserted on the (possibly new) TUI even though the value is unchanged.
    assert ui.notification_calls == [True]

    ui.emit_color_scheme("light")
    assert current_theme.name == "light"


# --------------------------------------------------------------------------
# initial theme setting (--use-theme)
#
# Port of `packages/coding-agent/test/theme-controller.test.ts`.
# --------------------------------------------------------------------------


def _make_with_managers(
    managers: list[FakeSettings],
    initial_theme_setting: str | None = None,
    **ui_kwargs: Any,
) -> tuple[InteractiveThemeController, FakeUi, Callable[[FakeSettings], None]]:
    """A controller whose settings manager can be swapped out, as on session switch."""
    ui = FakeUi(**ui_kwargs)
    current = managers[0]

    def select(manager: FakeSettings) -> None:
        nonlocal current
        current = manager

    controller = InteractiveThemeController(
        ui,  # type: ignore[arg-type]
        lambda: current,  # type: ignore[arg-type,return-value]
        lambda _message: None,
        lambda: None,
        initial_theme_setting,
    )
    return controller, ui, select


def test_uses_the_initial_theme_without_persisting_it():
    controller, ui, settings, _errors, _changed = _make("dark", initial_theme_setting="light")

    assert current_theme.name == "light"
    assert controller.get_theme_selection() == "light"
    asyncio.run(controller.apply_from_settings())

    # No terminal query: the theme was already decided by --use-theme.
    assert ui.background is None
    assert settings.set_calls == []
    assert settings.flush_count == 0


def test_resolves_a_theme_pair_and_follows_terminal_appearance_changes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COLORFGBG", "15;0")
    controller, ui, _settings, _errors, _changed = _make(
        "dark/light", initial_theme_setting="light/dark", color_scheme="light"
    )

    assert current_theme.name == "dark"
    asyncio.run(controller.apply_from_settings())
    assert current_theme.name == "light"
    assert ui.notifications_enabled is True

    ui.emit_color_scheme("dark")
    assert current_theme.name == "dark"


def test_detects_the_current_terminal_appearance_when_selecting_a_theme_pair(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COLORFGBG", "")
    controller, _ui, _settings, _errors, _changed = _make("dark", color_scheme="light")

    assert current_theme.name == "dark"
    asyncio.run(controller.set_theme_setting("light/dark"))
    assert current_theme.name == "light"


def test_an_explicit_selection_replaces_the_initial_theme():
    first = FakeSettings("dark")
    second = FakeSettings("light")
    controller, _ui, select = _make_with_managers([first, second], initial_theme_setting="light")
    asyncio.run(controller.apply_from_settings())

    assert controller.set_theme_name("dark").success is True
    select(second)
    asyncio.run(controller.apply_from_settings())

    assert controller.get_theme_selection() == "dark"
    assert current_theme.name == "dark"


def test_reloads_theme_settings_when_no_initial_theme_was_supplied():
    first = FakeSettings("dark")
    second = FakeSettings("light")
    controller, _ui, select = _make_with_managers([first, second])
    asyncio.run(controller.apply_from_settings())
    assert current_theme.name == "dark"

    first.theme_setting = "light"
    asyncio.run(controller.apply_from_settings())
    assert current_theme.name == "light"

    second.theme_setting = "dark"
    select(second)
    asyncio.run(controller.apply_from_settings())
    assert current_theme.name == "dark"
