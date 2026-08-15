"""Python port of `packages/coding-agent/test/session-selector-rename.test.ts`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pi_tui.keybindings import set_keybindings

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.session_manager import SessionInfo
from pi_coding_agent.modes.interactive.components.session_selector import SessionSelectorComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme

# Kitty keyboard protocol encoding for Ctrl+R
CTRL_R = "\x1b[114;5u"

_EPOCH = datetime.fromtimestamp(0, tz=UTC)


@pytest.fixture(autouse=True)
def _theme_and_keybindings() -> None:
    init_theme("dark")
    # Ensure test isolation: keybindings are a global singleton.
    set_keybindings(KeybindingsManager.create())


def make_session(session_id: str, name: str | None = None) -> SessionInfo:
    return SessionInfo(
        path=f"/tmp/{session_id}.jsonl",
        id=session_id,
        cwd="",
        name=name,
        created=_EPOCH,
        modified=_EPOCH,
        message_count=1,
        first_message="hello",
        all_messages_text="hello",
    )


async def flush_promises() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _build(
    sessions: list[SessionInfo],
    *,
    show_rename_hint: bool,
    rename_session: object = None,
) -> SessionSelectorComponent:
    async def load_current(_on_progress: object = None) -> list[SessionInfo]:
        return sessions

    async def load_all(_on_progress: object = None) -> list[SessionInfo]:
        return []

    return SessionSelectorComponent(
        load_current,
        load_all,
        lambda _path: None,
        lambda: None,
        lambda: None,
        lambda: None,
        KeybindingsManager.create(),
        rename_session=rename_session,
        show_rename_hint=show_rename_hint,
    )


class TestSessionSelectorRename:
    @pytest.mark.asyncio
    async def test_shows_rename_hint_in_interactive_resume_picker(self) -> None:
        selector = _build([make_session("a")], show_rename_hint=True)
        await flush_promises()

        output = "\n".join(selector.render(120))
        assert "ctrl+r" in output
        assert "rename" in output

    @pytest.mark.asyncio
    async def test_does_not_show_rename_hint_in_cli_resume_picker(self) -> None:
        selector = _build([make_session("a")], show_rename_hint=False)
        await flush_promises()

        output = "\n".join(selector.render(120))
        assert "ctrl+r" not in output
        assert "rename" not in output

    @pytest.mark.asyncio
    async def test_enters_rename_mode_on_ctrl_r_and_submits_with_enter(self) -> None:
        sessions = [make_session("a", name="Old")]
        renames: list[tuple[str, str | None]] = []

        async def rename_session(path: str, next_name: str | None) -> None:
            renames.append((path, next_name))

        selector = _build(sessions, show_rename_hint=True, rename_session=rename_session)
        await flush_promises()

        selector.get_session_list().handle_input(CTRL_R)
        await flush_promises()

        output = "\n".join(selector.render(120))
        assert "Rename Session" in output
        assert "Resume Session" not in output

        selector.handle_input("X")
        selector.handle_input("\r")
        await flush_promises()

        assert renames == [(sessions[0].path, "XOld")]
