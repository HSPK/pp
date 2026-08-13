"""Python port of
`packages/coding-agent/test/suite/regressions/startup-session-rebind-duplicate-subscription.test.ts`.

The regression: `InteractiveMode.rebindCurrentSession()` awaits
`bindCurrentSessionExtensions()`. If the session is replaced while that await
is outstanding -- the startup rebind still in flight when `setRebindSession`
fires for a replacement session -- the stale rebind must not resume and
subscribe a second listener to the *new* session. The fix captures
`const session = this.session` up front and bails with
`if (this.session !== session) return;` after the await.

TypeScript calls the method on a hand-built context object
(`prototype.rebindCurrentSession.call(context)`), which is exactly what
Python's unbound-method call does here: `InteractiveMode._rebind_current_session`
is invoked with a `_RebindContext` as `self`. Two shape differences, both
forced by the port:

- `bindCurrentSessionExtensions` has no counterpart. That method exists in
  TypeScript only to hand the extension *UI host* context (dialogs, widgets,
  theme control -- a documented omission, see the README) to the session, so
  the port awaits `session.bind_extensions()` directly. The controllable await
  therefore lives on the session stub rather than on the mode.
- The port's `session` is a property reading `runtime_host.session`, so the
  context exposes it as a property over a mutable attribute rather than a
  plain field.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


class _FakeExtensionRunner:
    """Records the UI host the rebind installs, as `ExtensionRunner` would."""

    def __init__(self) -> None:
        self.ui_contexts: list[tuple[Any, str]] = []

    def set_ui_context(self, ui: Any = None, mode: str = "print") -> None:
        self.ui_contexts.append((ui, mode))


class _BindableSession:
    """Stands in for `AgentSession`, with a `bind_extensions()` the test resolves."""

    def __init__(self) -> None:
        self.bind_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self.extension_runner = _FakeExtensionRunner()

    async def bind_extensions(self) -> None:
        await self.bind_future


class _RebindContext:
    """The `RebindContext` of the TypeScript test, as a `self` for the real method."""

    def __init__(self, startup_session: _BindableSession) -> None:
        self.current_session: _BindableSession = startup_session
        self._unsubscribe: Any = None
        self.compaction_queued_messages: list[Any] = []
        self.subscribe_calls = 0
        self.update_terminal_title_calls = 0
        self.footer = self
        self.ui = self

    @property
    def session(self) -> _BindableSession:
        return self.current_session

    def _apply_runtime_settings(self) -> None: ...

    def _rebuild_chat_from_session(self) -> None: ...

    def _subscribe_to_agent(self) -> None:
        self.subscribe_calls += 1

    def _update_available_provider_count(self) -> None: ...

    def _update_editor_border_color(self) -> None: ...

    def _update_terminal_title(self) -> None:
        self.update_terminal_title_calls += 1

    def set_session(self, session: object) -> None: ...

    def request_render(self) -> None: ...


async def test_does_not_subscribe_from_the_stale_startup_rebind() -> None:
    startup_session = _BindableSession()
    replacement_session = _BindableSession()
    context = _RebindContext(startup_session)

    startup_rebind = asyncio.ensure_future(
        InteractiveMode._rebind_current_session(context, startup_session, render_before_bind=False)
    )
    await asyncio.sleep(0)
    # The startup path subscribes only after its bind resolves.
    assert context.subscribe_calls == 0

    context.current_session = replacement_session
    replacement_rebind = asyncio.ensure_future(
        InteractiveMode._rebind_current_session(context, replacement_session, render_before_bind=True)
    )
    await asyncio.sleep(0)

    assert context.subscribe_calls == 1

    startup_session.bind_future.set_result(None)
    await startup_rebind

    # The stale startup rebind must bail: no second subscription, and none of
    # its post-bind work runs against the replacement session.
    assert context.subscribe_calls == 1
    assert context.update_terminal_title_calls == 0

    replacement_session.bind_future.set_result(None)
    await replacement_rebind

    assert context.subscribe_calls == 1
    assert context.update_terminal_title_calls == 1
