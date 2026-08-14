"""`pi.*` runtime actions must reach the session in the real CLI.

These actions are baked into an extension's `pi` object as the file loads,
which happens before any session exists. The CLI used to load extensions
without supplying bindings, so every one of them fell back to the no-op
default: `pi.send_user_message()` ran, reported nothing, and did nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pi_coding_agent.core.extensions import SessionRuntimeActions
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import Extension


class _FakeSession:
    """Records what the actions forward, standing in for `AgentSession`."""

    def __init__(self, name: str | None = None) -> None:
        self.user_messages: list[tuple[Any, str | None]] = []
        self.custom_messages: list[tuple[str, Any, bool, str | None]] = []
        self.entries: list[tuple[str, Any]] = []
        self._name = name
        self.active_tools = ["read"]
        self.session_manager = self

    async def send_user_message(self, content: Any, deliver_as: str | None = None) -> None:
        self.user_messages.append((content, deliver_as))

    async def send_custom_message(
        self,
        custom_type: str,
        content: Any,
        display: bool,
        details: Any = None,
        *,
        trigger_turn: bool = False,
        deliver_as: str | None = None,
    ) -> None:
        self.custom_messages.append((custom_type, content, trigger_turn, deliver_as))

    def append_custom_entry(self, custom_type: str, data: Any = None) -> None:
        self.entries.append((custom_type, data))

    def set_session_name(self, name: str) -> None:
        self._name = name

    @property
    def session_name(self) -> str | None:
        return self._name

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        self.active_tools = list(tool_names)

    def get_active_tool_names(self) -> list[str]:
        return self.active_tools


async def _settle() -> None:
    """The message actions are fire-and-forget, so let their tasks run."""
    await asyncio.sleep(0.05)


async def test_actions_are_inert_before_a_session_is_bound() -> None:
    """Extensions run module-level code at load time, before the session exists."""
    actions = SessionRuntimeActions().actions

    actions.send_user_message("too early")
    actions.send_message({"customType": "x", "content": "y", "display": True})
    actions.set_session_name("nobody")
    actions.append_entry("note")
    await _settle()

    assert actions.get_session_name() is None
    assert actions.get_active_tools() == []


async def test_send_user_message_reaches_the_session() -> None:
    holder = SessionRuntimeActions()
    session = _FakeSession()
    holder.bind(session)

    holder.actions.send_user_message("do the thing", {"deliverAs": "followUp"})
    await _settle()

    assert session.user_messages == [("do the thing", "followUp")]


async def test_send_message_forwards_the_custom_message_options() -> None:
    holder = SessionRuntimeActions()
    session = _FakeSession()
    holder.bind(session)

    holder.actions.send_message(
        {"customType": "status", "content": "working", "display": True},
        {"triggerTurn": True, "deliverAs": "steer"},
    )
    await _settle()

    assert session.custom_messages == [("status", "working", True, "steer")]


async def test_the_remaining_actions_reach_the_session() -> None:
    holder = SessionRuntimeActions()
    session = _FakeSession()
    holder.bind(session)
    actions = holder.actions

    actions.append_entry("bookmark", {"at": 1})
    actions.set_session_name("renamed")
    actions.set_active_tools(["bash", "edit"])

    assert session.entries == [("bookmark", {"at": 1})]
    assert actions.get_session_name() == "renamed"
    assert actions.get_active_tools() == ["bash", "edit"]


async def test_rebinding_follows_a_replacement_session() -> None:
    """`/new`, `/import` and `/clone` replace the session under the extension."""
    holder = SessionRuntimeActions()
    first = _FakeSession("first")
    second = _FakeSession("second")

    holder.bind(first)
    holder.actions.send_user_message("to first")
    await _settle()

    holder.bind(second)
    holder.actions.send_user_message("to second")
    await _settle()

    assert [content for content, _ in first.user_messages] == ["to first"]
    assert [content for content, _ in second.user_messages] == ["to second"]
    assert holder.actions.get_session_name() == "second"


async def test_an_extension_calling_pi_send_user_message_reaches_the_session() -> None:
    """The whole path: a loaded extension's `pi` object through to the session."""
    holder = SessionRuntimeActions()
    session = _FakeSession()
    holder.bind(session)

    extension = Extension(path="inline.py", resolved_path="inline.py", handlers={})
    pi = ExtensionAPI(extension, holder.actions)

    pi.send_user_message("from the extension", {"deliverAs": "steer"})
    await _settle()

    assert session.user_messages == [("from the extension", "steer")]


@pytest.mark.parametrize("deliver_as", ["steer", "followUp", None])
async def test_delivery_mode_is_passed_through(deliver_as: str | None) -> None:
    holder = SessionRuntimeActions()
    session = _FakeSession()
    holder.bind(session)

    options = {"deliverAs": deliver_as} if deliver_as is not None else None
    holder.actions.send_user_message("text", options)
    await _settle()

    assert session.user_messages == [("text", deliver_as)]


async def test_the_cli_loads_extensions_with_bound_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression itself: `entry.py` must pass `actions=`.

    Everything above passes with a CLI that never supplies bindings, because
    it builds the holder by hand. This checks the wiring that was missing.
    """
    from pi_coding_agent.cli import entry

    captured: dict[str, Any] = {}

    async def fake_discover(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _StopHere

    monkeypatch.setattr(entry, "discover_and_load_extensions", fake_discover)

    with pytest.raises(_StopHere):
        await entry.build_session_runtime(_MinimalArgs(), ".", ".")

    actions = captured.get("actions")
    assert actions is not None, "entry.py must pass actions= or every pi.* call is a no-op"
    # A default-constructed `ExtensionRuntimeActions` has the no-op lambdas;
    # a bound one carries the holder's methods.
    assert getattr(actions.send_user_message, "__self__", None) is not None


class _StopHere(Exception):
    """Ends `build_session_runtime` once the call under test has been made."""


class _MinimalArgs:
    """Only the attributes reached before extension loading."""

    def __getattr__(self, name: str) -> Any:
        return None


async def test_the_keyword_form_the_examples_use_works() -> None:
    """`examples/extensions/git_merge_and_resolve.py` calls with `deliver_as=`.

    The dict form is what TypeScript's options object ports to; the keyword
    form is what a Python extension author writes, and both docs and the
    shipped example use it. Accepting only one silently drops the delivery
    mode -- or raises TypeError inside a handler, where it is easy to miss.
    """
    holder = SessionRuntimeActions()
    session = _FakeSession()
    holder.bind(session)

    holder.actions.send_user_message("text", deliver_as="followUp")
    holder.actions.send_message(
        {"customType": "s", "content": "c", "display": True},
        deliver_as="steer",
        trigger_turn=True,
    )
    await _settle()

    assert session.user_messages == [("text", "followUp")]
    assert session.custom_messages == [("s", "c", True, "steer")]


# --------------------------------------------------------------------------
# ctx.shutdown()
# --------------------------------------------------------------------------


async def test_ctx_shutdown_is_a_no_op_without_a_registered_handler() -> None:
    """A session cannot know what "shut down" means for its host."""
    from pi_coding_agent.core.extensions.runner import ExtensionRunner

    runner = ExtensionRunner([], cwd=".")
    # No `bind_core`, so nothing is wired: the call must not raise.
    context = runner.create_context()
    context.shutdown()


async def test_ctx_shutdown_calls_the_registered_handler(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "suite"))
    from harness import create_harness

    called: list[bool] = []
    harness = await create_harness(tmp_path)
    try:
        harness.session.set_extension_shutdown_handler(lambda: called.append(True))

        harness.session.extension_runner.create_context().shutdown()

        assert called == [True]
    finally:
        harness.session.dispose()
