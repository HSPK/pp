"""Python port of `packages/coding-agent/test/suite/regressions/5943-session-start-notify.test.ts`.

Three of the seven TypeScript cases are portable; the other four pin behavior of
machinery this port deliberately omits:

* `renders loaded resources before restored messages without stale entries`
  needs `showLoadedResources`; this port never fills
  `loaded_resources_container` (no startup resource banner).
* `renders replacement session state before session_start handlers can notify`
  needs `InteractiveMode.rebindCurrentSession` plus
  `bindExtensions({uiContext})`; neither the rebind method nor the extension
  UI host (`ctx.ui.notify`) exists here, so there is no `apply/render/
  subscribe/bind/notify` ordering to assert.
* `runs the reload render hook before reload session_start handlers can notify`
  needs `AgentSession.reload({beforeSessionStart})`, which this port does not
  have (and `ctx.ui.notify` again).
* `keeps the reload blocker focused until async reload completes` needs the same
  async `session.reload`; `_handle_reload_command` here is synchronous apart
  from settings I/O and never installs a blocker component.

The two "subscribes before replacement session_start handlers send ..." cases
*are* ported: their subject is `AgentSession.bind_extensions()` emitting
`session_start` late enough that a listener subscribed beforehand still sees
the messages the handler sends. The TS versions reach that through
`rebindCurrentSession`, whose `subscribeToAgent` -> `bindCurrentSessionExtensions`
order is exactly "subscribe, then bind"; the Python versions subscribe and then
call `bind_extensions()` directly, which pins the same guarantee.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harness import create_harness, drain_extension_actions
from interactive_harness import make_interactive_mode
from pi_ai.providers.faux import faux_assistant_message
from pi_coding_agent.core.agent_session import AgentSessionEvent
from pi_coding_agent.core.extensions.loader import ExtensionAPI


def _message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part.text for part in content if getattr(part, "type", None) == "text")


def _record_messages(event: AgentSessionEvent, events: list[str]) -> None:
    if event.type not in ("message_start", "message_end"):
        return
    events.append(f"{event.type}:{event.message.role}:{_message_text(event.message)}")


async def test_refreshes_hide_thinking_block_before_rebuilding_chat_during_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    events: list[str] = []

    monkeypatch.setattr(mode.settings_manager, "get_hide_thinking_block", lambda: True)

    original_reload = mode.settings_manager.reload

    async def reload_settings() -> None:
        events.append("reload")
        await original_reload()

    monkeypatch.setattr(mode.settings_manager, "reload", reload_settings)
    monkeypatch.setattr(
        mode, "_rebuild_chat_from_session", lambda: events.append(f"rebuild:{mode.hide_thinking_block}")
    )

    assert mode.hide_thinking_block is False

    await mode._handle_reload_command()

    assert mode.hide_thinking_block is True
    assert events == ["reload", "rebuild:True"]


async def test_subscribes_before_session_start_handlers_send_messages(tmp_path: Path) -> None:
    events: list[str] = []

    def factory(pi: ExtensionAPI) -> None:
        def on_session_start(_event, _ctx) -> None:
            pi.send_message({"customType": "session-start", "content": "custom from start", "display": True})

        pi.on("session_start", on_session_start)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        events.append("subscribe")
        harness.session.subscribe(lambda event: _record_messages(event, events))

        events.append("bind")
        await harness.session.bind_extensions()
        await drain_extension_actions()

        assert events == [
            "subscribe",
            "bind",
            "message_start:custom:custom from start",
            "message_end:custom:custom from start",
        ]
    finally:
        harness.cleanup()


async def test_subscribes_before_session_start_handlers_send_user_messages(tmp_path: Path) -> None:
    events: list[str] = []

    def factory(pi: ExtensionAPI) -> None:
        def on_session_start(_event, _ctx) -> None:
            pi.send_user_message("user from start")

        pi.on("session_start", on_session_start)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    harness.set_responses([faux_assistant_message("assistant from start")])
    try:
        events.append("subscribe")
        harness.session.subscribe(lambda event: _record_messages(event, events))

        events.append("bind")
        await harness.session.bind_extensions()
        await drain_extension_actions()
        await asyncio.wait_for(harness.session.agent.wait_for_idle(), timeout=10)

        assert events[:2] == ["subscribe", "bind"]
        assert "message_start:user:user from start" in events
        assert "message_end:user:user from start" in events
        assert "message_end:assistant:assistant from start" in events
    finally:
        harness.cleanup()
