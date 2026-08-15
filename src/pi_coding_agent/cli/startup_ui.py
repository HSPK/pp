"""Small TUI dialogs shown before the main interactive session starts.

Ported from ``packages/coding-agent/src/cli/startup-ui.ts``.

Some questions have to be answered before a session exists: "do you trust this
project folder?", first-time setup, and any prompt an extension raises during
startup. Each of these spins up a throwaway `TuiMainScreen`, shows one
component, waits for an answer, then clears and stops the screen so the real
interactive UI can take over a clean terminal.

Themes are loaded from *global* settings only. Project settings are what the
trust prompt is deciding about, so reading them first would defeat the point.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pi_tui.keybindings import set_keybindings
from pi_tui.tasks import spawn
from pi_tui.terminal import ProcessTerminal
from pi_tui.tui_main_screen import TuiMainScreen

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.config import (
    APP_NAME,
    CONFIG_DIR_NAME,
    ENV_AGENT_DIR,
    PACKAGE_NAME,
    get_agent_dir,
    get_settings_path,
)
from pi_coding_agent.core.experimental import are_experimental_features_enabled
from pi_coding_agent.core.package_manager import PackageManager
from pi_coding_agent.core.settings_manager import SettingsManager, SettingsManagerCreateOptions
from pi_coding_agent.modes.interactive.components.extension_dialogs import (
    ExtensionInputComponent,
    ExtensionSelectorComponent,
)
from pi_coding_agent.modes.interactive.components.first_time_setup import (
    FirstTimeSetupComponent,
    FirstTimeSetupResult,
)
from pi_coding_agent.modes.interactive.theme.theme import (
    Theme,
    detect_terminal_background_from_env,
    detect_terminal_theme_for_auto,
    init_theme,
    load_theme_from_path,
    parse_auto_theme_setting,
    resolve_theme_setting,
    set_registered_themes,
    set_theme,
)

T = TypeVar("T")

OFFICIAL_PACKAGE_NAME = "pp-coding-agent"
OFFICIAL_APP_NAME = "pi"
OFFICIAL_CONFIG_DIR_NAME = ".pi"

_CLEAR_SETTLE_DELAY_S = 0.025


def is_official_distribution(*, package_name: str, app_name: str, config_dir_name: str) -> bool:
    return (
        package_name == OFFICIAL_PACKAGE_NAME
        and app_name == OFFICIAL_APP_NAME
        and config_dir_name == OFFICIAL_CONFIG_DIR_NAME
    )


def load_themes(resources: Sequence[object]) -> list[Theme]:
    """Load enabled theme resources, skipping any that fail to parse.

    A broken theme must not stop a startup prompt from appearing; the normal
    resource loader reports theme diagnostics later in startup.
    """
    themes: list[Theme] = []
    seen: set[str] = set()
    for resource in resources:
        if not getattr(resource, "enabled", False):
            continue
        try:
            loaded = load_theme_from_path(getattr(resource, "path", ""))
        except Exception:
            continue
        name = getattr(loaded, "name", None)
        if name:
            if name in seen:
                continue
            seen.add(name)
        themes.append(loaded)
    return themes


async def load_startup_themes(settings_manager: SettingsManager) -> list[Theme]:
    global_settings_manager = SettingsManager.in_memory(
        settings_manager.get_global_settings(),
        SettingsManagerCreateOptions(project_trusted=False),
    )
    package_manager = PackageManager(
        cwd=os.getcwd(),
        agent_dir=get_agent_dir(),
        settings_manager=global_settings_manager,
    )
    resolved = await package_manager.resolve(lambda _source: "skip")
    return load_themes(resolved.themes)


async def create_startup_tui(settings_manager: SettingsManager) -> TuiMainScreen:
    set_registered_themes(await load_startup_themes(settings_manager))
    terminal_theme = detect_terminal_background_from_env().theme
    init_theme(resolve_theme_setting(settings_manager.get_theme_setting(), terminal_theme) or terminal_theme)
    set_keybindings(KeybindingsManager.create())
    ui = TuiMainScreen(ProcessTerminal(), settings_manager.get_show_hardware_cursor(), get_agent_dir())
    ui.set_clear_on_shrink(settings_manager.get_clear_on_shrink())
    return ui


async def _apply_detected_startup_theme(ui: TuiMainScreen, settings_manager: SettingsManager) -> None:
    theme_setting = settings_manager.get_theme_setting()
    if theme_setting and not parse_auto_theme_setting(theme_setting):
        return
    terminal_theme = await detect_terminal_theme_for_auto(ui=ui, timeout_ms=100)
    set_theme(resolve_theme_setting(theme_setting, terminal_theme) or terminal_theme)
    ui.invalidate()
    ui.request_render()


async def _clear_startup_tui(ui: TuiMainScreen) -> None:
    ui.clear()
    ui.request_render()
    await asyncio.sleep(_CLEAR_SETTLE_DELAY_S)


def should_run_first_time_setup(settings_path: str | None = None) -> bool:
    """First-time setup runs only on an untouched, official, experimental install.

    All of these must hold: this is the official Pi distribution (not a
    fork/rebrand), experimental features are on, the default agent directory is
    in use, and ``settings.json`` does not exist yet.
    """
    if not is_official_distribution(package_name=PACKAGE_NAME, app_name=APP_NAME, config_dir_name=CONFIG_DIR_NAME):
        return False
    if not are_experimental_features_enabled():
        return False
    if os.environ.get(ENV_AGENT_DIR):
        return False
    return not Path(settings_path or get_settings_path()).exists()


async def show_startup_selector(
    settings_manager: SettingsManager, title: str, options: Sequence[tuple[str, T]]
) -> T | None:
    """Show a one-shot selector and return the chosen value (``None`` if cancelled)."""
    ui = await create_startup_tui(settings_manager)
    future: asyncio.Future[T | None] = asyncio.get_running_loop().create_future()
    labels = [label for label, _ in options]
    by_label = dict(options)

    def finish(result: T | None) -> None:
        if future.done():
            return
        future.set_result(result)

    selector = ExtensionSelectorComponent(
        title,
        labels,
        lambda label: finish(by_label.get(label)),
        lambda: finish(None),
        tui=ui,
    )
    ui.add_child(selector)
    ui.set_focus(selector)
    ui.start()
    theme_task = spawn(_apply_detected_startup_theme(ui, settings_manager))
    try:
        return await future
    finally:
        theme_task.cancel()
        await _clear_startup_tui(ui)
        ui.stop()


async def show_startup_input(
    settings_manager: SettingsManager, title: str, placeholder: str | None = None
) -> str | None:
    ui = await create_startup_tui(settings_manager)
    future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()

    def finish(result: str | None) -> None:
        if future.done():
            return
        future.set_result(result)

    component = ExtensionInputComponent(title, placeholder, lambda value: finish(value), lambda: finish(None), tui=ui)
    ui.add_child(component)
    ui.set_focus(component)
    ui.start()
    theme_task = spawn(_apply_detected_startup_theme(ui, settings_manager))
    try:
        return await future
    finally:
        theme_task.cancel()
        component.dispose()
        await _clear_startup_tui(ui)
        ui.stop()


async def show_first_time_setup(settings_manager: SettingsManager) -> None:
    """Show the first-run dialog and persist the answers."""
    ui = await create_startup_tui(settings_manager)
    future: asyncio.Future[FirstTimeSetupResult | None] = asyncio.get_running_loop().create_future()

    def finish(result: FirstTimeSetupResult | None) -> None:
        if future.done():
            return
        future.set_result(result)

    ui.start()
    detected_theme = await detect_terminal_theme_for_auto(ui=ui, timeout_ms=100)
    set_theme(detected_theme)

    def preview(theme_name: str) -> None:
        set_theme(theme_name)
        ui.request_render()

    component = FirstTimeSetupComponent(
        detected_theme=detected_theme,
        on_theme_preview=preview,
        on_submit=finish,
        on_cancel=lambda: finish(None),
    )
    ui.add_child(component)
    ui.set_focus(component)
    ui.request_render()

    try:
        result = await future
        if result is not None:
            settings_manager.set_theme(result.theme)
            settings_manager.set_enable_analytics(result.share_analytics)
            await settings_manager.flush()
    finally:
        await _clear_startup_tui(ui)
        ui.stop()


__all__ = [
    "OFFICIAL_APP_NAME",
    "OFFICIAL_CONFIG_DIR_NAME",
    "OFFICIAL_PACKAGE_NAME",
    "create_startup_tui",
    "is_official_distribution",
    "load_startup_themes",
    "load_themes",
    "should_run_first_time_setup",
    "show_first_time_setup",
    "show_startup_input",
    "show_startup_selector",
]
