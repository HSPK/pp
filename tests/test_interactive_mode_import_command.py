"""Python port of `packages/coding-agent/test/interactive-mode-import-command.test.ts`.

Pins `/import` path parsing (quotes, apostrophes, command-token boundaries) and
the import handler's success and file-not-found paths. Called against a stand-in
`self`, mirroring the TypeScript prototype call.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pi_coding_agent.core.agent_session_runtime import SessionImportFileNotFoundError
from pi_coding_agent.core.session_cwd import (
    MissingSessionCwdError,
    SessionCwdIssue,
    format_missing_session_cwd_prompt,
)
from pi_coding_agent.modes.interactive.components.simple_selectors import ConfirmSelectorComponent
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode
from pi_coding_agent.modes.interactive.theme.theme import init_theme

_get_path_command_argument = InteractiveMode._get_path_command_argument


class _RuntimeHost:
    def __init__(self, error: Exception | None = None, *, error_once: bool = False) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._error = error
        self._error_once = error_once

    async def import_from_jsonl(self, input_path: str, cwd_override: str | None = None) -> dict[str, Any]:
        self.calls.append((input_path, cwd_override))
        if self._error is not None:
            error = self._error
            if self._error_once:
                self._error = None
            raise error
        return {"cancelled": False}


class _Footer:
    def set_session(self, session: Any) -> None:
        pass


class _ImportContext:
    """Stand-in `self` for `_handle_import_command`, mirroring TypeScript's context object."""

    def __init__(self, runtime_host: _RuntimeHost, *, confirm: bool = True) -> None:
        self.runtime_host = runtime_host
        self.footer = _Footer()
        self.session = object()
        self.errors: list[str] = []
        self.statuses: list[str] = []
        self.confirms: list[tuple[str, str]] = []
        self._confirm_answer = confirm

    async def _show_confirm(self, title: str, message: str) -> bool:
        self.confirms.append((title, message))
        return self._confirm_answer

    _get_path_command_argument = InteractiveMode._get_path_command_argument
    # The real recovery prompt, as TypeScript puts the real `getPathCommandArgument`
    # on its context object.
    _prompt_for_missing_session_cwd = InteractiveMode._prompt_for_missing_session_cwd

    def show_error(self, message: str) -> None:
        self.errors.append(message)

    def show_status(self, message: str) -> None:
        self.statuses.append(message)

    def _clear_status_indicator(self, kind: str | None = None) -> None:
        pass

    def _rebuild_chat_from_session(self) -> None:
        pass

    def _update_terminal_title(self) -> None:
        pass


def test_strips_quotes_from_import_path_arguments() -> None:
    assert _get_path_command_argument(None, '/import "path/to/session.jsonl"', "/import") == "path/to/session.jsonl"
    assert (
        _get_path_command_argument(None, '/import "path with spaces/session.jsonl"', "/import")
        == "path with spaces/session.jsonl"
    )


def test_preserves_apostrophes_in_unquoted_import_path_arguments() -> None:
    assert _get_path_command_argument(None, "/import john's/session.jsonl", "/import") == "john's/session.jsonl"


def test_enforces_command_token_boundaries() -> None:
    assert _get_path_command_argument(None, "/important /tmp/session.jsonl", "/import") is None
    assert _get_path_command_argument(None, "/exporter out.html", "/export") is None
    assert _get_path_command_argument(None, "/import /tmp/session.jsonl", "/import") == "/tmp/session.jsonl"


async def test_passes_unquoted_path_to_runtime_host_import_from_jsonl() -> None:
    runtime_host = _RuntimeHost()
    context = _ImportContext(runtime_host)

    await InteractiveMode._handle_import_command(context, '/import "path/to/session.jsonl"')

    assert context.confirms == [("Import session", "Replace current session with path/to/session.jsonl?")]
    assert runtime_host.calls == [("path/to/session.jsonl", None)]
    assert context.errors == []
    assert context.statuses == ["Session imported from: path/to/session.jsonl"]


async def test_passes_unquoted_apostrophe_path_unchanged() -> None:
    runtime_host = _RuntimeHost()
    context = _ImportContext(runtime_host)

    await InteractiveMode._handle_import_command(context, "/import john's/session.jsonl")

    assert runtime_host.calls == [("john's/session.jsonl", None)]
    assert context.errors == []
    assert context.statuses == ["Session imported from: john's/session.jsonl"]


async def test_shows_a_non_fatal_error_when_import_path_does_not_exist() -> None:
    runtime_host = _RuntimeHost(SessionImportFileNotFoundError("/tmp/missing-session.jsonl"))
    context = _ImportContext(runtime_host)

    await InteractiveMode._handle_import_command(context, "/import /tmp/missing-session.jsonl")

    assert context.errors == ["Failed to import session: File not found: /tmp/missing-session.jsonl"]
    assert context.statuses == []


async def test_unexpected_import_errors_are_not_swallowed_as_a_soft_error() -> None:
    """Not a separate TypeScript `it(...)`, but the other half of the branch its
    `expect(handleFatalRuntimeError).not.toHaveBeenCalled()` assertion pins:
    that assertion is only meaningful for `SessionImportFileNotFoundError`
    because upstream's `catch` funnels every *other* error to
    `handleFatalRuntimeError`, which shows the message and then exits the
    process -- it never turns an unrecognized error into a quiet
    `showError` the user could miss. This port has no `handleFatalRuntimeError`
    /`process.exit` analogue (that machinery -- `stop()`, the theme-watcher
    shutdown, signal deregistration -- is unported entirely, not specific to
    `/import`), so the closest faithful behavior is to not swallow the error:
    it propagates instead of being downgraded to `show_error`.
    """
    runtime_host = _RuntimeHost(RuntimeError("disk exploded"))
    context = _ImportContext(runtime_host)

    with pytest.raises(RuntimeError, match="disk exploded"):
        await InteractiveMode._handle_import_command(context, "/import /tmp/session.jsonl")

    assert context.errors == []
    assert context.statuses == []


async def test_declining_the_confirmation_cancels_the_import() -> None:
    """Not a separate TypeScript `it(...)`, but the other half of the branch its
    `showExtensionConfirm` assertion pins: upstream returns early with
    "Import cancelled" and never touches `importFromJsonl` when the user says no.
    """
    runtime_host = _RuntimeHost()
    context = _ImportContext(runtime_host, confirm=False)

    await InteractiveMode._handle_import_command(context, "/import /tmp/session.jsonl")

    assert context.confirms == [("Import session", "Replace current session with /tmp/session.jsonl?")]
    assert runtime_host.calls == []
    assert context.statuses == ["Import cancelled"]
    assert context.errors == []


async def test_usage_error_when_no_path_is_given_and_nothing_is_confirmed() -> None:
    runtime_host = _RuntimeHost()
    context = _ImportContext(runtime_host)

    await InteractiveMode._handle_import_command(context, "/import")

    assert context.errors == ["Usage: /import <path.jsonl>"]
    assert context.confirms == []
    assert runtime_host.calls == []


# The cases below have no TypeScript `it(...)`, but they pin the two remaining
# branches of upstream's `handleImportCommand`: the `MissingSessionCwdError`
# recovery (prompt for a cwd, then retry with it) and its cancellation. The
# TypeScript test only wires `promptForMissingSessionCwd` into the context
# without exercising it.


def _missing_cwd_error() -> MissingSessionCwdError:
    return MissingSessionCwdError(
        SessionCwdIssue(sessionCwd="/gone", fallbackCwd="/here", sessionFile="/sessions/a.jsonl")
    )


async def test_retries_the_import_in_the_fallback_cwd_when_the_session_cwd_is_gone() -> None:
    error = _missing_cwd_error()
    runtime_host = _RuntimeHost(error, error_once=True)
    context = _ImportContext(runtime_host)

    await InteractiveMode._handle_import_command(context, "/import /tmp/session.jsonl")

    assert context.confirms == [
        ("Import session", "Replace current session with /tmp/session.jsonl?"),
        ("Session cwd not found", format_missing_session_cwd_prompt(error.issue)),
    ]
    assert runtime_host.calls == [("/tmp/session.jsonl", None), ("/tmp/session.jsonl", "/here")]
    assert context.errors == []
    assert context.statuses == ["Session imported from: /tmp/session.jsonl"]


async def test_declining_the_fallback_cwd_cancels_the_import() -> None:
    runtime_host = _RuntimeHost(_missing_cwd_error(), error_once=True)
    context = _ImportContext(runtime_host)
    context._confirm_answer = True

    answers = iter([True, False])

    async def confirm(title: str, message: str) -> bool:
        context.confirms.append((title, message))
        return next(answers)

    context._show_confirm = confirm  # type: ignore[method-assign]

    await InteractiveMode._handle_import_command(context, "/import /tmp/session.jsonl")

    assert runtime_host.calls == [("/tmp/session.jsonl", None)]
    assert context.statuses == ["Import cancelled"]
    assert context.errors == []


# `_show_confirm` and the component behind it are what the stubbed
# `_show_confirm` above stands in for; these drive the real objects so the stub
# cannot hide a broken dialog.


class _ConfirmHost:
    def __init__(self) -> None:
        self.shown: list[ConfirmSelectorComponent] = []
        self.hidden = 0

    def _show_selector(self, component: ConfirmSelectorComponent) -> None:
        self.shown.append(component)

    def _hide_selector(self) -> None:
        self.hidden += 1

    _show_confirm = InteractiveMode._show_confirm


@pytest.mark.parametrize(("key", "expected"), [("\r", True), ("\x1b", False)])
async def test_show_confirm_resolves_from_the_real_yes_no_dialog(key: str, expected: bool) -> None:
    init_theme("dark")
    host = _ConfirmHost()
    pending = asyncio.ensure_future(host._show_confirm("Import session", "Replace current session with a.jsonl?"))
    await asyncio.sleep(0)

    assert len(host.shown) == 1
    component = host.shown[0]
    component.handle_input(key)

    assert await pending is expected
    assert host.hidden == 1


async def test_confirm_selector_second_option_answers_no() -> None:
    init_theme("dark")
    answers: list[bool] = []
    component = ConfirmSelectorComponent("t", "m", answers.append, lambda: answers.append(False))

    items = component.get_select_list()._items
    assert [item.label for item in items] == ["Yes", "No"]

    component.handle_input("\x1b[B")
    component.handle_input("\r")
    assert answers == [False]
