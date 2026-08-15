"""Bind theme selection to the interactive TUI and terminal color-scheme changes.

Python port of
`packages/coding-agent/src/modes/interactive/theme/theme-controller.ts`.

The controller owns three things the interactive mode used to do inline:

1. resolving the raw `theme` *setting* (which may be an ``"<light>/<dark>"``
   auto pair) against the terminal background before loading a theme,
2. auto-sync: while an auto pair is configured, `TUI` terminal color-scheme
   notifications are enabled and the theme follows the terminal,
3. persisting a high-confidence background detection back to settings the
   first time the app runs without a `theme` setting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.theme.theme import (
    TerminalAutoThemeDetector,
    TerminalTheme,
    Theme,
    ThemeResult,
    detect_terminal_background_from_env,
    detect_terminal_background_theme,
    detect_terminal_theme_for_auto,
    init_theme,
    parse_auto_theme_setting,
    resolve_theme_setting,
    set_theme,
    set_theme_instance,
)

_DETECTION_TIMEOUT_MS = 100.0


class ThemeControllerUi(TerminalAutoThemeDetector, Protocol):
    """The `TUI` surface the controller needs.

    Typed structurally because the interactive mode hands the controller its
    `create_interactive_tui_reference(...)` proxy, not a `TuiBase` instance.
    """

    def invalidate(self) -> None: ...

    def request_render(self) -> None: ...

    def set_terminal_color_scheme_notifications(self, enabled: bool) -> None: ...

    def on_terminal_color_scheme_change(self, listener: Callable[[TerminalTheme], None]) -> Callable[[], None]: ...


class InteractiveThemeController:
    def __init__(
        self,
        ui: ThemeControllerUi,
        get_settings_manager: Callable[[], SettingsManager],
        show_error: Callable[[str], None],
        on_changed: Callable[[], None],
        initial_theme_setting: str | None = None,
    ) -> None:
        self.ui = ui
        self._get_settings_manager = get_settings_manager
        self._show_error = show_error
        self._on_changed = on_changed
        self._current_theme_setting = initial_theme_setting
        self._terminal_theme: TerminalTheme = detect_terminal_background_from_env().theme
        self._active_theme_name: str | None = resolve_theme_setting(self._theme_setting(), self._terminal_theme)
        self._auto_sync_enabled = False
        self._terminal_color_scheme_unsubscribe: Callable[[], None] | None = None
        init_theme(self._active_theme_name)
        self._bind_terminal_color_scheme_listener()

    @property
    def settings_manager(self) -> SettingsManager:
        """The *current* settings manager.

        Read through a callback rather than captured once: switching sessions
        replaces `AgentSession.settings_manager`, and a captured reference
        would keep reading the discarded session's settings.
        """
        return self._get_settings_manager()

    def _theme_setting(self) -> str | None:
        """The `--use-theme` override if this run has one, else the setting."""
        if self._current_theme_setting is not None:
            return self._current_theme_setting
        return self.settings_manager.get_theme_setting()

    def rebind_tui(self) -> None:
        if self._terminal_color_scheme_unsubscribe is not None:
            self._terminal_color_scheme_unsubscribe()
        self._bind_terminal_color_scheme_listener()
        self.ui.set_terminal_color_scheme_notifications(self._auto_sync_enabled)

    async def apply_from_settings(self) -> None:
        settings_manager = self.settings_manager
        theme_setting = self._theme_setting()
        auto_theme = parse_auto_theme_setting(theme_setting)
        if auto_theme is not None:
            self._terminal_theme = await detect_terminal_theme_for_auto(self.ui, _DETECTION_TIMEOUT_MS)
            self._set_auto_sync(True)
            self._apply_theme_name(
                auto_theme.light_theme if self._terminal_theme == "light" else auto_theme.dark_theme,
                show_error=True,
            )
            return

        self._set_auto_sync(False)
        if theme_setting is not None:
            self._apply_theme_name(theme_setting, show_error=True)
            return

        detection = await detect_terminal_background_theme(self.ui, _DETECTION_TIMEOUT_MS)
        self._terminal_theme = detection.theme
        if not self._apply_theme_name(detection.theme).success:
            return
        if detection.confidence == "high":
            settings_manager.set_theme(detection.theme)
            await settings_manager.flush()

    def get_theme_selection(self) -> str | None:
        return self._theme_setting() or self._active_theme_name

    def set_theme_name(self, theme_name: str, show_error: bool = False) -> ThemeResult:
        self._set_auto_sync(False)
        result = self._apply_theme_name(theme_name, show_error=show_error)
        if result.success:
            self._current_theme_setting = theme_name
        return result

    async def set_theme_setting(self, theme_setting: str) -> None:
        self._current_theme_setting = theme_setting
        await self.apply_from_settings()

    def set_theme_instance(self, theme_instance: Theme) -> ThemeResult:
        self._set_auto_sync(False)
        set_theme_instance(theme_instance)
        self._active_theme_name = "<in-memory>"
        self._notify_changed()
        return ThemeResult(success=True)

    def preview(self, theme_setting_or_name: str) -> None:
        theme_name = resolve_theme_setting(theme_setting_or_name, self._terminal_theme) or self._active_theme_name
        if not theme_name:
            return
        if set_theme(theme_name).success:
            self.ui.invalidate()
            self.ui.request_render()

    def disable_auto_sync(self) -> None:
        self._set_auto_sync(False)

    def get_terminal_theme(self) -> TerminalTheme:
        return self._terminal_theme

    # -- internals ---------------------------------------------------------

    def _apply_theme_name(self, theme_name: str, show_error: bool = False) -> ThemeResult:
        result = set_theme(theme_name)
        self._active_theme_name = theme_name if result.success else "dark"
        self._notify_changed()
        if not result.success and show_error:
            self._show_error(f'Failed to load theme "{theme_name}": {result.error}\nFell back to dark theme.')
        return result

    def _notify_changed(self) -> None:
        self.ui.invalidate()
        self._on_changed()

    def _set_auto_sync(self, enabled: bool) -> None:
        if self._auto_sync_enabled == enabled:
            return
        self._auto_sync_enabled = enabled
        self.ui.set_terminal_color_scheme_notifications(enabled)

    def _bind_terminal_color_scheme_listener(self) -> None:
        self._terminal_color_scheme_unsubscribe = self.ui.on_terminal_color_scheme_change(self._apply_terminal_theme)

    def _apply_terminal_theme(self, terminal_theme: TerminalTheme) -> None:
        if not self._auto_sync_enabled:
            return
        self._terminal_theme = terminal_theme
        auto_theme = parse_auto_theme_setting(self._theme_setting())
        if auto_theme is None:
            self._set_auto_sync(False)
            return
        theme_name = auto_theme.light_theme if terminal_theme == "light" else auto_theme.dark_theme
        if theme_name != self._active_theme_name:
            self._apply_theme_name(theme_name)


__all__ = ["InteractiveThemeController"]
