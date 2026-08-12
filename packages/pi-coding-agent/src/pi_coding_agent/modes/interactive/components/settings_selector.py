"""The `/settings` dialog.

Ported from ``packages/coding-agent/src/modes/interactive/components/settings-selector.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pi_tui.component import Component, Container
from pi_tui.components.select_list import SelectItem, SelectList, SelectListLayoutOptions
from pi_tui.components.settings_list import SettingItem, SettingsList, SettingsListOptions
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.terminal_image import get_capabilities

from ....core.http_dispatcher import HTTP_IDLE_TIMEOUT_CHOICES, format_http_idle_timeout_ms
from ..theme.theme import (
    get_select_list_theme,
    get_settings_list_theme,
    parse_auto_theme_setting,
    theme,
)
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_display_text

SETTINGS_SUBMENU_SELECT_LIST_LAYOUT = SelectListLayoutOptions(min_primary_column_width=12, max_primary_column_width=32)

THINKING_DESCRIPTIONS: dict[str, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Extra-high reasoning (~32k tokens)",
    "max": "Maximum reasoning",
}

DEFAULT_PROJECT_TRUST_LABELS: dict[str, str] = {
    "ask": "Ask",
    "always": "Always trust",
    "never": "Never trust",
}

DEFAULT_PROJECT_TRUST_BY_LABEL: dict[str, str] = {label: value for value, label in DEFAULT_PROJECT_TRUST_LABELS.items()}

AUTOMATIC_THEME_VALUE = "/"


@dataclass
class SettingsConfig:
    auto_compact: bool = True
    show_images: bool = True
    image_width_cells: int = 60
    auto_resize_images: bool = True
    block_images: bool = False
    enable_skill_commands: bool = True
    steering_mode: str = "one-at-a-time"
    follow_up_mode: str = "one-at-a-time"
    transport: str = "auto"
    http_idle_timeout_ms: int = 300_000
    thinking_level: str = "off"
    available_thinking_levels: list[str] = field(default_factory=list)
    current_theme: str = "dark"
    terminal_theme: str = "dark"
    available_themes: list[str] = field(default_factory=list)
    hide_thinking_block: bool = False
    mermaid_rendering_mode: str = "off"
    show_cache_miss_notices: bool = True
    collapse_changelog: bool = False
    enable_install_telemetry: bool = True
    double_escape_action: str = "tree"
    tree_filter_mode: str = "default"
    show_hardware_cursor: bool = False
    editor_padding_x: int = 1
    output_pad: int = 1
    autocomplete_max_visible: int = 10
    quiet_startup: bool = False
    default_project_trust: str = "ask"
    clear_on_shrink: bool = False
    show_terminal_progress: bool = True
    tui_mode: str = "regular"
    fullscreen_exit_output: str = "transcript"
    fullscreen_scrollbar: str = "auto"
    warnings: dict[str, Any] = field(default_factory=dict)


def _noop(*_args: object) -> None:
    return None


@dataclass
class SettingsCallbacks:
    on_auto_compact_change: Callable[[bool], None] = _noop
    on_show_images_change: Callable[[bool], None] = _noop
    on_image_width_cells_change: Callable[[int], None] = _noop
    on_auto_resize_images_change: Callable[[bool], None] = _noop
    on_block_images_change: Callable[[bool], None] = _noop
    on_enable_skill_commands_change: Callable[[bool], None] = _noop
    on_steering_mode_change: Callable[[str], None] = _noop
    on_follow_up_mode_change: Callable[[str], None] = _noop
    on_transport_change: Callable[[str], None] = _noop
    on_http_idle_timeout_ms_change: Callable[[int], None] = _noop
    on_thinking_level_change: Callable[[str], None] = _noop
    on_theme_change: Callable[[str], None] = _noop
    on_theme_preview: Callable[[str], None] | None = None
    on_hide_thinking_block_change: Callable[[bool], None] = _noop
    on_mermaid_rendering_mode_change: Callable[[str], None] = _noop
    on_show_cache_miss_notices_change: Callable[[bool], None] = _noop
    on_collapse_changelog_change: Callable[[bool], None] = _noop
    on_enable_install_telemetry_change: Callable[[bool], None] = _noop
    on_double_escape_action_change: Callable[[str], None] = _noop
    on_tree_filter_mode_change: Callable[[str], None] = _noop
    on_show_hardware_cursor_change: Callable[[bool], None] = _noop
    on_editor_padding_x_change: Callable[[int], None] = _noop
    on_output_pad_change: Callable[[int], None] = _noop
    on_autocomplete_max_visible_change: Callable[[int], None] = _noop
    on_quiet_startup_change: Callable[[bool], None] = _noop
    on_default_project_trust_change: Callable[[str], None] = _noop
    on_clear_on_shrink_change: Callable[[bool], None] = _noop
    on_show_terminal_progress_change: Callable[[bool], None] = _noop
    on_tui_mode_change: Callable[[str], None] = _noop
    on_fullscreen_exit_output_change: Callable[[str], None] = _noop
    on_fullscreen_scrollbar_change: Callable[[str], None] = _noop
    on_warnings_change: Callable[[dict[str, Any]], None] = _noop
    on_cancel: Callable[[], None] = _noop


class WarningSettingsSubmenu(Container):
    def __init__(
        self,
        warnings: dict[str, Any],
        on_change: Callable[[dict[str, Any]], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        self.state = dict(warnings)

        items = [
            SettingItem(
                id="anthropic-extra-usage",
                label="Anthropic extra usage",
                description="Warn when Anthropic subscription auth may use paid extra usage",
                current_value="true" if self.state.get("anthropicExtraUsage", True) else "false",
                values=["true", "false"],
            )
        ]

        def handle_change(item_id: str, new_value: str) -> None:
            if item_id == "anthropic-extra-usage":
                self.state = {**self.state, "anthropicExtraUsage": new_value == "true"}
                on_change(dict(self.state))

        self.settings_list = SettingsList(
            items, min(len(items), 10), get_settings_list_theme(), handle_change, on_cancel
        )
        self.add_child(self.settings_list)

    def handle_input(self, data: str) -> None:
        self.settings_list.handle_input(data)


class SelectSubmenu(Container):
    def __init__(
        self,
        title: str,
        description: str,
        options: list[SelectItem],
        current_value: str,
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
        on_selection_change: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.add_child(Text(theme.bold(theme.fg("accent", title)), 0, 0))
        if description:
            self.add_child(Spacer(1))
            self.add_child(Text(theme.fg("muted", description), 0, 0))
        self.add_child(Spacer(1))

        self.select_list = SelectList(
            options, min(len(options), 10), get_select_list_theme(), SETTINGS_SUBMENU_SELECT_LIST_LAYOUT
        )
        for index, option in enumerate(options):
            if option.value == current_value:
                self.select_list.set_selected_index(index)
                break

        self.select_list.on_select = lambda item: on_select(item.value)
        self.select_list.on_cancel = on_cancel
        if on_selection_change is not None:
            self.select_list.on_selection_change = lambda item: on_selection_change(item.value)

        self.add_child(self.select_list)
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("dim", "  Enter to select · Esc to go back"), 0, 0))

    def handle_input(self, data: str) -> None:
        self.select_list.handle_input(data)


def theme_items(available_themes: list[str]) -> list[SelectItem]:
    return [SelectItem(value=name, label=name) for name in available_themes]


def single_mode_theme_items(available_themes: list[str]) -> list[SelectItem]:
    return [
        SelectItem(
            value=AUTOMATIC_THEME_VALUE,
            label="Automatic",
            description="Use separate themes for light and dark terminal appearance",
        ),
        *theme_items(available_themes),
    ]


def preferred_theme(available_themes: list[str], preferred: str | None, fallback: str) -> str:
    if preferred and preferred in available_themes:
        return preferred
    if fallback in available_themes:
        return fallback
    return available_themes[0] if available_themes else fallback


def default_automatic_themes(current_theme_setting: str, available_themes: list[str]) -> tuple[str, str]:
    auto_theme = parse_auto_theme_setting(current_theme_setting)
    if auto_theme:
        return auto_theme.light_theme, auto_theme.dark_theme

    current_fixed_theme = None if "/" in current_theme_setting else current_theme_setting
    theme_name = preferred_theme(available_themes, current_fixed_theme, "dark")
    return theme_name, theme_name


class ThemeSubmenu(Container):
    def __init__(
        self,
        current_theme_setting: str,
        terminal_theme: str,
        available_themes: list[str],
        callbacks: SettingsCallbacks,
        on_done: Callable[..., None],
    ) -> None:
        super().__init__()
        self.callbacks = callbacks
        self.available_themes = available_themes
        self.terminal_theme = terminal_theme
        self.on_done = on_done
        self.original_theme_setting = current_theme_setting
        self.input_component: Component | None = None

        auto_theme = parse_auto_theme_setting(current_theme_setting)
        self.light_theme, self.dark_theme = default_automatic_themes(current_theme_setting, available_themes)
        fixed_theme = None if (auto_theme or "/" in current_theme_setting) else current_theme_setting
        self.mode = "automatic" if auto_theme else "single"
        self.single_theme = preferred_theme(
            available_themes,
            fixed_theme if fixed_theme else (self._get_active_automatic_theme() if auto_theme else None),
            "dark",
        )

        if self.mode == "automatic":
            self._show_automatic_menu()
        else:
            self._show_single_menu()

    def handle_input(self, data: str) -> None:
        if self.input_component is not None:
            handler = getattr(self.input_component, "handle_input", None)
            if handler is not None:
                handler(data)

    def _set_content(self, render_component: Component, input_component: Component | None = None) -> None:
        self.clear()
        self.add_child(render_component)
        self.input_component = input_component if input_component is not None else render_component

    def _preview(self, value: str) -> None:
        if self.callbacks.on_theme_preview is not None:
            self.callbacks.on_theme_preview(value)

    def _show_single_menu(self) -> None:
        self.mode = "single"

        def select(value: str) -> None:
            if value == AUTOMATIC_THEME_VALUE:
                self.mode = "automatic"
                self._preview(self._get_theme_setting())
                self._show_automatic_menu()
                return
            self.single_theme = value
            self.on_done(value)

        menu = SelectSubmenu(
            "Theme",
            "Select a theme, or choose Automatic to follow terminal appearance.",
            single_mode_theme_items(self.available_themes),
            self.single_theme,
            select,
            self._cancel,
            lambda value: self._preview(
                self._get_automatic_theme_setting() if value == AUTOMATIC_THEME_VALUE else value
            ),
        )
        self._set_content(menu)

    def _show_automatic_menu(self) -> None:
        self.mode = "automatic"
        content = Container()
        content.add_child(Text(theme.bold(theme.fg("accent", "Automatic Theme")), 0, 0))
        content.add_child(Spacer(1))
        content.add_child(Text(theme.fg("muted", "Choose themes for terminal light and dark appearance."), 0, 0))
        content.add_child(Text(theme.fg("muted", "Light/dark detection requires terminal support."), 0, 0))
        content.add_child(Spacer(1))

        def light_submenu(current_value: str, done: Callable[..., None]) -> Component:
            def select(value: str) -> None:
                self.light_theme = value
                self._preview(self._get_theme_setting())
                done(value)

            return self._create_theme_select(
                "Light Theme",
                "Select the theme to use for light terminal appearance",
                current_value,
                done,
                select,
            )

        def dark_submenu(current_value: str, done: Callable[..., None]) -> Component:
            def select(value: str) -> None:
                self.dark_theme = value
                self._preview(self._get_theme_setting())
                done(value)

            return self._create_theme_select(
                "Dark Theme",
                "Select the theme to use for dark terminal appearance",
                current_value,
                done,
                select,
            )

        items = [
            SettingItem(
                id="light-theme",
                label="Light theme",
                description="Theme to use in automatic mode when the terminal is light",
                current_value=self.light_theme,
                submenu=light_submenu,
            ),
            SettingItem(
                id="dark-theme",
                label="Dark theme",
                description="Theme to use in automatic mode when the terminal is dark",
                current_value=self.dark_theme,
                submenu=dark_submenu,
            ),
            SettingItem(
                id="apply",
                label="Apply",
                description="Save and go back",
                current_value="save and go back",
                values=["save and go back"],
            ),
            SettingItem(
                id="single-mode",
                label="Change mode",
                description="Switch to one theme for light and dark",
                current_value="switch to single theme",
                values=["switch to single theme"],
            ),
        ]

        def handle_change(item_id: str, _new_value: str) -> None:
            if item_id == "single-mode":
                self.mode = "single"
                self.single_theme = self._get_active_automatic_theme()
                self._preview(self.single_theme)
                self._show_single_menu()
            elif item_id == "apply":
                self.on_done(self._get_automatic_theme_setting())

        settings_list = SettingsList(items, min(len(items), 10), get_settings_list_theme(), handle_change, self._cancel)
        content.add_child(settings_list)
        self._set_content(content, settings_list)

    def _create_theme_select(
        self,
        title: str,
        description: str,
        current_value: str,
        done: Callable[..., None],
        on_select: Callable[[str], None],
    ) -> SelectSubmenu:
        def cancel() -> None:
            self._preview(self._get_theme_setting())
            done()

        return SelectSubmenu(
            title,
            description,
            theme_items(self.available_themes),
            current_value,
            on_select,
            cancel,
            self._preview,
        )

    def _get_theme_setting(self) -> str:
        return self._get_automatic_theme_setting() if self.mode == "automatic" else self.single_theme

    def _get_active_automatic_theme(self) -> str:
        return self.light_theme if self.terminal_theme == "light" else self.dark_theme

    def _get_automatic_theme_setting(self) -> str:
        return f"{self.light_theme}/{self.dark_theme}"

    def _cancel(self) -> None:
        self._preview(self.original_theme_setting)
        self.on_done()


def _build_settings_items(config: SettingsConfig, callbacks: SettingsCallbacks) -> list[SettingItem]:
    supports_images = bool(get_capabilities().images)
    follow_up_key = key_display_text("app.message.followUp")
    current_warnings = dict(config.warnings)

    def warnings_submenu(_current_value: str, done: Callable[..., None]) -> Component:
        def on_change(warnings: dict[str, Any]) -> None:
            nonlocal current_warnings
            current_warnings = warnings
            callbacks.on_warnings_change(warnings)

        return WarningSettingsSubmenu(current_warnings, on_change, lambda: done())

    def thinking_submenu(current_value: str, done: Callable[..., None]) -> Component:
        def select(value: str) -> None:
            callbacks.on_thinking_level_change(value)
            done(value)

        return SelectSubmenu(
            "Thinking Level",
            "Select reasoning depth for thinking-capable models",
            [
                SelectItem(value=level, label=level, description=THINKING_DESCRIPTIONS.get(level))
                for level in config.available_thinking_levels
            ],
            current_value,
            select,
            lambda: done(),
        )

    def theme_submenu(current_value: str, done: Callable[..., None]) -> Component:
        return ThemeSubmenu(current_value, config.terminal_theme, config.available_themes, callbacks, done)

    items: list[SettingItem] = [
        SettingItem(
            id="autocompact",
            label="Auto-compact",
            description="Automatically compact context when it gets too large",
            current_value="true" if config.auto_compact else "false",
            values=["true", "false"],
        ),
        SettingItem(
            id="steering-mode",
            label="Steering mode",
            description=(
                "Enter while streaming queues steering messages. 'one-at-a-time': deliver one, "
                "wait for response. 'all': deliver all at once."
            ),
            current_value=config.steering_mode,
            values=["one-at-a-time", "all"],
        ),
        SettingItem(
            id="follow-up-mode",
            label="Follow-up mode",
            description=(
                f"{follow_up_key} queues follow-up messages until agent stops. 'one-at-a-time': "
                "deliver one, wait for response. 'all': deliver all at once."
            ),
            current_value=config.follow_up_mode,
            values=["one-at-a-time", "all"],
        ),
        SettingItem(
            id="transport",
            label="Transport",
            description="Preferred transport for providers that support multiple transports",
            current_value=config.transport,
            values=["sse", "websocket", "websocket-cached", "auto"],
        ),
        SettingItem(
            id="http-idle-timeout",
            label="HTTP idle timeout",
            description=(
                "Maximum idle gap while waiting for HTTP headers or body chunks. Disable for local "
                "models that pause longer than five minutes."
            ),
            current_value=format_http_idle_timeout_ms(config.http_idle_timeout_ms),
            values=[choice.label for choice in HTTP_IDLE_TIMEOUT_CHOICES],
        ),
        SettingItem(
            id="hide-thinking",
            label="Hide thinking",
            description="Hide thinking blocks in assistant responses",
            current_value="true" if config.hide_thinking_block else "false",
            values=["true", "false"],
        ),
        SettingItem(
            id="mermaid-rendering",
            label="Mermaid diagrams",
            description="Render Mermaid code blocks as Unicode diagrams",
            current_value=config.mermaid_rendering_mode,
            values=["off", "final", "streaming"],
        ),
        SettingItem(
            id="cache-miss-notices",
            label="Cache miss notices",
            description="Show transcript notices for significant prompt-cache misses",
            current_value="true" if config.show_cache_miss_notices else "false",
            values=["true", "false"],
        ),
        SettingItem(
            id="collapse-changelog",
            label="Collapse changelog",
            description="Show condensed changelog after updates",
            current_value="true" if config.collapse_changelog else "false",
            values=["true", "false"],
        ),
        SettingItem(
            id="quiet-startup",
            label="Quiet startup",
            description="Disable verbose printing at startup",
            current_value="true" if config.quiet_startup else "false",
            values=["true", "false"],
        ),
        SettingItem(
            id="install-telemetry",
            label="Install telemetry",
            description="Send an anonymous version/update ping after changelog-detected updates",
            current_value="true" if config.enable_install_telemetry else "false",
            values=["true", "false"],
        ),
        SettingItem(
            id="default-project-trust",
            label="Default project trust",
            description="Fallback behavior when no extension or saved trust decision decides project trust",
            current_value=DEFAULT_PROJECT_TRUST_LABELS[config.default_project_trust],
            values=list(DEFAULT_PROJECT_TRUST_LABELS.values()),
        ),
        SettingItem(
            id="double-escape-action",
            label="Double-escape action",
            description="Action when pressing Escape twice with empty editor",
            current_value=config.double_escape_action,
            values=["tree", "fork", "none"],
        ),
        SettingItem(
            id="tree-filter-mode",
            label="Tree filter mode",
            description="Default filter when opening /tree",
            current_value=config.tree_filter_mode,
            values=["default", "no-tools", "user-only", "labeled-only", "all"],
        ),
        SettingItem(
            id="warnings",
            label="Warnings",
            description="Enable or disable individual warnings",
            current_value="configure",
            submenu=warnings_submenu,
        ),
        SettingItem(
            id="thinking",
            label="Thinking level",
            description="Reasoning depth for thinking-capable models",
            current_value=config.thinking_level,
            submenu=thinking_submenu,
        ),
        SettingItem(
            id="tui-mode",
            label="TUI mode",
            description="Interface layout; fullscreen mode is experimental",
            current_value=config.tui_mode,
            values=["regular", "fullscreen"],
        ),
        SettingItem(
            id="fullscreen-exit-output",
            label="Fullscreen exit output",
            description="Print the transcript or only a session resume hint when exiting fullscreen mode",
            current_value=config.fullscreen_exit_output,
            values=["transcript", "resume-hint"],
        ),
        SettingItem(
            id="fullscreen-scrollbar",
            label="Fullscreen scrollbar",
            description="Scrollbar behavior in fullscreen mode; has no effect in regular mode",
            current_value=config.fullscreen_scrollbar,
            values=["auto", "always", "hidden"],
        ),
        SettingItem(
            id="theme",
            label="Theme",
            description="Color theme for the interface",
            current_value=config.current_theme,
            submenu=theme_submenu,
        ),
    ]

    if supports_images:
        items.insert(
            1,
            SettingItem(
                id="show-images",
                label="Show images",
                description="Render images inline in terminal",
                current_value="true" if config.show_images else "false",
                values=["true", "false"],
            ),
        )
        items.insert(
            2,
            SettingItem(
                id="image-width-cells",
                label="Image width",
                description="Preferred inline image width in terminal cells",
                current_value=str(config.image_width_cells),
                values=["60", "80", "120"],
            ),
        )

    items.insert(
        3 if supports_images else 1,
        SettingItem(
            id="auto-resize-images",
            label="Auto-resize images",
            description="Resize large images to 2000x2000 max for better model compatibility",
            current_value="true" if config.auto_resize_images else "false",
            values=["true", "false"],
        ),
    )

    def index_of(item_id: str) -> int:
        return next(i for i, item in enumerate(items) if item.id == item_id)

    items.insert(
        index_of("auto-resize-images") + 1,
        SettingItem(
            id="block-images",
            label="Block images",
            description="Prevent images from being sent to LLM providers",
            current_value="true" if config.block_images else "false",
            values=["true", "false"],
        ),
    )
    items.insert(
        index_of("block-images") + 1,
        SettingItem(
            id="skill-commands",
            label="Skill commands",
            description="Register skills as /skill:name commands",
            current_value="true" if config.enable_skill_commands else "false",
            values=["true", "false"],
        ),
    )
    items.insert(
        index_of("skill-commands") + 1,
        SettingItem(
            id="show-hardware-cursor",
            label="Show hardware cursor",
            description="Show the terminal cursor while still positioning it for IME support",
            current_value="true" if config.show_hardware_cursor else "false",
            values=["true", "false"],
        ),
    )
    items.insert(
        index_of("show-hardware-cursor") + 1,
        SettingItem(
            id="editor-padding",
            label="Editor padding",
            description="Horizontal padding for input editor (0-3)",
            current_value=str(config.editor_padding_x),
            values=["0", "1", "2", "3"],
        ),
    )
    items.insert(
        index_of("editor-padding") + 1,
        SettingItem(
            id="output-padding",
            label="Output padding",
            description="Horizontal padding for user messages, assistant messages, and thinking",
            current_value=str(config.output_pad),
            values=["0", "1"],
        ),
    )
    items.insert(
        index_of("output-padding") + 1,
        SettingItem(
            id="autocomplete-max-visible",
            label="Autocomplete max items",
            description="Max visible items in autocomplete dropdown (3-20)",
            current_value=str(config.autocomplete_max_visible),
            values=["3", "5", "7", "10", "15", "20"],
        ),
    )
    items.insert(
        index_of("autocomplete-max-visible") + 1,
        SettingItem(
            id="clear-on-shrink",
            label="Clear on shrink",
            description="Clear empty rows when content shrinks (may cause flicker)",
            current_value="true" if config.clear_on_shrink else "false",
            values=["true", "false"],
        ),
    )
    items.insert(
        index_of("clear-on-shrink") + 1,
        SettingItem(
            id="terminal-progress",
            label="Terminal progress",
            description="Show OSC 9;4 progress indicators in the terminal tab bar",
            current_value="true" if config.show_terminal_progress else "false",
            values=["true", "false"],
        ),
    )

    return items


class SettingsSelectorComponent(Container):
    def __init__(self, config: SettingsConfig, callbacks: SettingsCallbacks) -> None:
        super().__init__()
        items = _build_settings_items(config, callbacks)

        def handle_change(item_id: str, new_value: str) -> None:
            is_true = new_value == "true"
            if item_id == "autocompact":
                callbacks.on_auto_compact_change(is_true)
            elif item_id == "show-images":
                callbacks.on_show_images_change(is_true)
            elif item_id == "image-width-cells":
                callbacks.on_image_width_cells_change(int(new_value))
            elif item_id == "auto-resize-images":
                callbacks.on_auto_resize_images_change(is_true)
            elif item_id == "block-images":
                callbacks.on_block_images_change(is_true)
            elif item_id == "skill-commands":
                callbacks.on_enable_skill_commands_change(is_true)
            elif item_id == "steering-mode":
                callbacks.on_steering_mode_change(new_value)
            elif item_id == "follow-up-mode":
                callbacks.on_follow_up_mode_change(new_value)
            elif item_id == "transport":
                callbacks.on_transport_change(new_value)
            elif item_id == "http-idle-timeout":
                choice = next((c for c in HTTP_IDLE_TIMEOUT_CHOICES if c.label == new_value), None)
                if choice is not None:
                    callbacks.on_http_idle_timeout_ms_change(choice.timeout_ms)
            elif item_id == "hide-thinking":
                callbacks.on_hide_thinking_block_change(is_true)
            elif item_id == "mermaid-rendering":
                callbacks.on_mermaid_rendering_mode_change(new_value)
            elif item_id == "cache-miss-notices":
                callbacks.on_show_cache_miss_notices_change(is_true)
            elif item_id == "collapse-changelog":
                callbacks.on_collapse_changelog_change(is_true)
            elif item_id == "quiet-startup":
                callbacks.on_quiet_startup_change(is_true)
            elif item_id == "install-telemetry":
                callbacks.on_enable_install_telemetry_change(is_true)
            elif item_id == "default-project-trust":
                value = DEFAULT_PROJECT_TRUST_BY_LABEL.get(new_value)
                if value:
                    callbacks.on_default_project_trust_change(value)
            elif item_id == "double-escape-action":
                callbacks.on_double_escape_action_change(new_value)
            elif item_id == "tree-filter-mode":
                callbacks.on_tree_filter_mode_change(new_value)
            elif item_id == "show-hardware-cursor":
                callbacks.on_show_hardware_cursor_change(is_true)
            elif item_id == "editor-padding":
                callbacks.on_editor_padding_x_change(int(new_value))
            elif item_id == "output-padding":
                callbacks.on_output_pad_change(0 if new_value == "0" else 1)
            elif item_id == "autocomplete-max-visible":
                callbacks.on_autocomplete_max_visible_change(int(new_value))
            elif item_id == "clear-on-shrink":
                callbacks.on_clear_on_shrink_change(is_true)
            elif item_id == "terminal-progress":
                callbacks.on_show_terminal_progress_change(is_true)
            elif item_id == "tui-mode":
                callbacks.on_tui_mode_change(new_value)
            elif item_id == "fullscreen-exit-output":
                callbacks.on_fullscreen_exit_output_change(new_value)
            elif item_id == "fullscreen-scrollbar":
                callbacks.on_fullscreen_scrollbar_change(new_value)
            elif item_id == "theme":
                callbacks.on_theme_change(new_value)

        self.add_child(DynamicBorder())
        self.settings_list = SettingsList(
            items,
            10,
            get_settings_list_theme(),
            handle_change,
            callbacks.on_cancel,
            SettingsListOptions(enable_search=True),
        )
        self.add_child(self.settings_list)
        self.add_child(DynamicBorder())

    def get_settings_list(self) -> SettingsList:
        return self.settings_list

    def handle_input(self, data: str) -> None:
        self.settings_list.handle_input(data)


__all__ = [
    "AUTOMATIC_THEME_VALUE",
    "DEFAULT_PROJECT_TRUST_BY_LABEL",
    "DEFAULT_PROJECT_TRUST_LABELS",
    "THINKING_DESCRIPTIONS",
    "SelectSubmenu",
    "SettingsCallbacks",
    "SettingsConfig",
    "SettingsSelectorComponent",
    "ThemeSubmenu",
    "WarningSettingsSubmenu",
    "default_automatic_themes",
    "preferred_theme",
    "single_mode_theme_items",
    "theme_items",
]
