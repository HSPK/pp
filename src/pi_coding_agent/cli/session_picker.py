"""TUI session selector for the `--resume` flag.

Python port of `packages/coding-agent/src/cli/session-picker.ts`.

`--resume` without a session argument needs a picker before the main TUI
exists, so this drives the same `SessionSelectorComponent` the `/resume` slash
command uses, on the lightweight startup screen from `startup_ui`.

TypeScript resolves a `Promise` from three callbacks; this awaits an
`asyncio.Future` set by the same three callbacks.
"""

from __future__ import annotations

import asyncio
import sys

from pi_tui.keybindings import set_keybindings

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.components.session_selector import (
    SessionSelectorComponent,
    SessionsLoader,
)

from .startup_ui import create_startup_tui


async def select_session(
    current_sessions_loader: SessionsLoader,
    all_sessions_loader: SessionsLoader,
    settings_manager: SettingsManager,
) -> str | None:
    """Show the session selector. Returns the chosen session path, or `None` if cancelled.

    Exits the process outright when the user asks to quit, matching upstream's
    `process.exit(0)`.
    """
    ui = await create_startup_tui(settings_manager)
    future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()

    keybindings = KeybindingsManager.create()
    set_keybindings(keybindings)

    def finish(result: str | None) -> None:
        if future.done():
            return
        ui.stop()
        future.set_result(result)

    def on_exit() -> None:
        ui.stop()
        sys.exit(0)

    selector = SessionSelectorComponent(
        current_sessions_loader,
        all_sessions_loader,
        finish,
        lambda: finish(None),
        on_exit,
        ui.request_render,
        keybindings=keybindings,
        show_rename_hint=False,
    )

    ui.add_child(selector)
    ui.set_focus(selector.get_session_list())
    ui.start()
    try:
        return await future
    finally:
        ui.stop()


__all__ = ["select_session"]
