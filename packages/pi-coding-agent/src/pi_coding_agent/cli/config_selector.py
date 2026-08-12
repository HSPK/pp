"""TUI config selector for the `config` command.

Python port of `packages/coding-agent/src/cli/config-selector.ts`.

`config` shows the same `ConfigSelectorComponent` the interactive mode uses, on
a throwaway full-screen TUI, and returns once the user closes it. TypeScript
resolves a `Promise` from the component's close/exit callbacks; this awaits an
`asyncio.Future` set by the same callbacks.
"""

from __future__ import annotations

import asyncio
import sys

from pi_tui.keybindings import set_keybindings
from pi_tui.terminal import ProcessTerminal
from pi_tui.tui_main_screen import TuiMainScreen

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.package_manager import ResolvedPaths
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.components.config_selector import (
    ConfigSelectorComponent,
    ConfigWriteScope,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme


async def select_config(
    *,
    resolved_paths: dict[str, ResolvedPaths],
    settings_manager: SettingsManager,
    cwd: str,
    agent_dir: str,
    write_scope: ConfigWriteScope,
    project_mode_available: bool,
) -> None:
    """Show the config selector and return when it closes."""
    # TypeScript passes `enableWatcher = true` here; the theme file watcher is
    # not ported (see theme.py's module docstring), so only the theme is set.
    init_theme(settings_manager.get_theme())
    set_keybindings(KeybindingsManager.create())

    ui = TuiMainScreen(ProcessTerminal(), None, agent_dir)
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def on_close() -> None:
        if future.done():
            return
        ui.stop()
        future.set_result(None)

    def on_exit() -> None:
        ui.stop()
        sys.exit(0)

    selector = ConfigSelectorComponent(
        resolved_paths,
        settings_manager,
        cwd,
        agent_dir,
        on_close,
        on_exit,
        ui.request_render,
        ui.terminal.rows,
        write_scope,
        project_mode_available,
    )

    ui.add_child(selector)
    ui.set_focus(selector.get_resource_list())
    ui.start()
    try:
        await future
    finally:
        ui.stop()


__all__ = ["select_config"]
