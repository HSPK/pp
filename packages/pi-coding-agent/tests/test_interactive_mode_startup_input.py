"""Python port of `packages/coding-agent/test/interactive-mode-startup-input.test.ts`.

Like the TypeScript file, this calls `InteractiveMode`'s submit/input methods
against a hand-built stand-in for `self`, so no terminal, session or agent is
constructed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


class _FakeEditor:
    def __init__(self) -> None:
        self.history: list[str] = []
        self.texts: list[str] = []

    def add_to_history(self, text: str) -> None:
        self.history.append(text)

    def set_text(self, text: str) -> None:
        self.texts.append(text)


class _FakeSession:
    is_compacting = False
    is_streaming = False
    is_bash_running = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def prompt(self, text: str, **_kwargs: Any) -> None:
        self.prompts.append(text)


class _SubmitContext:
    """Stand-in for `this` in TypeScript's `setupEditorSubmitHandler.call(context)`."""

    _UNSUPPORTED_COMMANDS = InteractiveMode._UNSUPPORTED_COMMANDS

    _setup_editor_submit_handler = InteractiveMode._setup_editor_submit_handler
    _handle_submit = InteractiveMode._handle_submit
    _handle_slash_command = InteractiveMode._handle_slash_command
    _deliver_input = InteractiveMode._deliver_input
    get_user_input = InteractiveMode.get_user_input

    def __init__(self) -> None:
        self.default_editor: Any = type("_DefaultEditor", (), {"on_submit": None})()
        self.editor = _FakeEditor()
        self.session = _FakeSession()
        self.pending_user_inputs: list[str] = []
        self.flush_calls = 0
        self._on_input_future: asyncio.Future[str] | None = None

    def _flush_pending_bash_components(self) -> None:
        self.flush_calls += 1


async def test_queues_a_normal_prompt_submitted_before_the_input_callback_is_installed() -> None:
    context = _SubmitContext()
    context._setup_editor_submit_handler()

    context.default_editor.on_submit(" early prompt ")
    # `_setup_editor_submit_handler` spawns `_handle_submit`; let it run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert context.pending_user_inputs == ["early prompt"]
    assert context.flush_calls == 1
    assert context.editor.history == ["early prompt"]


async def test_returns_queued_startup_input_before_installing_a_new_input_callback() -> None:
    context = _SubmitContext()
    context.pending_user_inputs = ["queued prompt"]

    assert await context.get_user_input() == "queued prompt"
    assert context._on_input_future is None
    assert context.pending_user_inputs == []
