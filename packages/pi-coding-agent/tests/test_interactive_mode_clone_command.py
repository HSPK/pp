"""Python port of `packages/coding-agent/test/interactive-mode-clone-command.test.ts`.

Calls `InteractiveMode._handle_clone_command` against a stand-in `self`, the way
the TypeScript test calls the prototype method with a fake `this`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


@dataclass
class _Recorder:
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def __call__(self, *args: Any) -> None:
        self.calls.append(args)


class _CloneContext:
    def __init__(self, leaf_id: str | None) -> None:
        self.fork_calls: list[tuple[str, str]] = []
        self.session_manager = _SessionManager(leaf_id)
        self.runtime_host = _RuntimeHost(self.fork_calls)
        self.editor = _Editor()
        self.ui = _Ui()
        self.footer = _Footer()
        self.show_status = _Recorder()
        self.show_error = _Recorder()
        self.rebuild_calls: list[tuple[Any, ...]] = []

    def _rebuild_chat_from_session(self) -> None:
        self.rebuild_calls.append(())


class _SessionManager:
    def __init__(self, leaf_id: str | None) -> None:
        self._leaf_id = leaf_id

    def get_leaf_id(self) -> str | None:
        return self._leaf_id


class _RuntimeHost:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self._calls = calls

    async def fork(self, entry_id: str, position: str = "before") -> dict[str, Any]:
        self._calls.append((entry_id, position))
        return {"cancelled": False}


class _Editor:
    def __init__(self) -> None:
        self.set_text_calls: list[str] = []

    def set_text(self, text: str) -> None:
        self.set_text_calls.append(text)


class _Ui:
    def __init__(self) -> None:
        self.render_calls = 0

    def request_render(self) -> None:
        self.render_calls += 1


class _Footer:
    def __init__(self) -> None:
        self.set_session_calls = 0

    def set_session(self, session: Any) -> None:
        self.set_session_calls += 1


async def test_clones_the_current_leaf_into_a_new_session() -> None:
    context = _CloneContext("leaf-123")

    await InteractiveMode._handle_clone_command(context)

    assert context.fork_calls == [("leaf-123", "at")]
    assert context.rebuild_calls == []
    assert context.editor.set_text_calls == [""]
    assert context.show_status.calls == [("Cloned to new session",)]
    assert context.show_error.calls == []
    assert context.ui.render_calls == 0


async def test_shows_a_status_message_when_there_is_nothing_to_clone() -> None:
    context = _CloneContext(None)

    await InteractiveMode._handle_clone_command(context)

    assert context.fork_calls == []
    assert context.show_status.calls == [("Nothing to clone yet",)]
    assert context.show_error.calls == []
