"""Python port of `packages/coding-agent/test/agent-session-runtime-events.test.ts`.

Pins the session-lifecycle extension events `AgentSessionRuntime` emits around
every session replacement: `session_before_switch`, `session_before_fork`,
`session_shutdown` and `session_start`, plus `session_before_*` cancellation.

The TypeScript suite drives a `registerFauxProvider()` model. This port uses
`test_agent_session_runtime`'s scripted stream function -- no network call is
made -- and builds one `Extension` per session (TS re-runs its
`ExtensionFactory` per session for the same effect).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pi_agent.agent import Agent, MutableAgentState
from pi_ai.types import TextContent
from test_agent_session_runtime import (
    TEST_MODEL,
    _make_assistant_message,
    _make_model_runtime,
    _scripted_stream_fn,
    _wait,
)

from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.agent_session_runtime import (
    AgentSessionRuntime,
    create_agent_session_runtime,
)
from pi_coding_agent.core.extensions.types import (
    Extension,
    SessionBeforeForkEvent,
    SessionBeforeForkResult,
    SessionBeforeSwitchEvent,
    SessionBeforeSwitchResult,
    SessionShutdownEvent,
    SessionStartEvent,
)
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionResult
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

ExtensionFactory = Callable[[], Extension]


class _LifecycleRuntimeFactory:
    """Builds a real `AgentSession` per replacement, with a fresh `Extension`."""

    def __init__(
        self,
        tmp_path: Path,
        model_runtime: ModelRuntime,
        extension_factory: ExtensionFactory,
    ) -> None:
        self._tmp_path = tmp_path
        self._model_runtime = model_runtime
        self._extension_factory = extension_factory
        self.sessions: list[AgentSession] = []

    async def __call__(
        self,
        *,
        cwd: str,
        agent_dir: str,
        session_manager: SessionManager,
        session_start_event: SessionStartEvent | None = None,
        **_ignored: Any,
    ) -> CreateAgentSessionResult:
        agent_dir_path = self._tmp_path / "agent"
        agent_dir_path.mkdir(parents=True, exist_ok=True)
        resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=str(agent_dir_path)))
        resource_loader.reload()
        agent = Agent(
            _scripted_stream_fn(
                [
                    _make_assistant_message([TextContent(text="one")]),
                    _make_assistant_message([TextContent(text="two")]),
                    _make_assistant_message([TextContent(text="three")]),
                ]
            ),
            initial_state=MutableAgentState(model=TEST_MODEL, system_prompt="You are a test assistant."),
        )
        existing = session_manager.build_session_context()
        if existing.messages:
            agent.state.messages = existing.messages
        session = AgentSession(
            agent=agent,
            session_manager=session_manager,
            settings_manager=SettingsManager.in_memory(),
            cwd=cwd,
            resource_loader=resource_loader,
            model_runtime=self._model_runtime,
            base_tools_override={},
            extensions=[self._extension_factory()],
            session_start_event=session_start_event,
        )
        self.sessions.append(session)
        return CreateAgentSessionResult(session=session, model_fallback_message=None)


async def create_runtime_host(
    tmp_path: Path, extension_factory: ExtensionFactory
) -> tuple[AgentSessionRuntime, _LifecycleRuntimeFactory]:
    model_runtime = await _make_model_runtime(tmp_path)
    factory = _LifecycleRuntimeFactory(tmp_path, model_runtime, extension_factory)
    session_manager = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"))
    runtime = await _wait(
        create_agent_session_runtime(
            factory, cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=session_manager
        )
    )
    await _wait(runtime.session.bind_extensions())
    return runtime, factory


def _handlers_extension(handlers: dict[str, list[Any]]) -> Extension:
    return Extension(path="lifecycle-ext.py", resolved_path="/test/lifecycle-ext.py", handlers=handlers)


class TestAgentSessionRuntimeSessionLifecycleEvents:
    async def test_emits_session_before_switch_and_session_start_for_new_and_resume_flows(self, tmp_path: Path) -> None:
        events: list[Any] = []

        def make_extension() -> Extension:
            return _handlers_extension(
                {
                    "session_before_switch": [lambda event, ctx: events.append(event)],
                    "session_shutdown": [lambda event, ctx: events.append(event)],
                    "session_start": [lambda event, ctx: events.append(event)],
                }
            )

        runtime, _factory = await create_runtime_host(tmp_path, make_extension)
        try:
            assert events == [SessionStartEvent(reason="startup")]
            events.clear()

            await _wait(runtime.session.prompt("hello"))
            original_session_file = runtime.session.session_file
            assert original_session_file

            new_session_result = await _wait(runtime.new_session())
            assert new_session_result["cancelled"] is False
            await _wait(runtime.session.bind_extensions())
            second_session_file = runtime.session.session_file
            assert events == [
                SessionBeforeSwitchEvent(reason="new", target_session_file=None),
                SessionShutdownEvent(reason="new", target_session_file=second_session_file),
                SessionStartEvent(reason="new", previous_session_file=original_session_file),
            ]

            events.clear()
            assert second_session_file

            switch_result = await _wait(runtime.switch_session(original_session_file))
            assert switch_result["cancelled"] is False
            await _wait(runtime.session.bind_extensions())
            assert events == [
                SessionBeforeSwitchEvent(reason="resume", target_session_file=original_session_file),
                SessionShutdownEvent(reason="resume", target_session_file=original_session_file),
                SessionStartEvent(reason="resume", previous_session_file=second_session_file),
            ]
        finally:
            await _wait(runtime.dispose())

    async def test_honors_session_before_switch_cancellation(self, tmp_path: Path) -> None:
        events: list[Any] = []

        def cancel_switch(event, ctx) -> SessionBeforeSwitchResult:
            events.append(event)
            return SessionBeforeSwitchResult(cancel=True)

        def make_extension() -> Extension:
            return _handlers_extension(
                {
                    "session_before_switch": [cancel_switch],
                    "session_start": [lambda event, ctx: events.append(event)],
                }
            )

        runtime, _factory = await create_runtime_host(tmp_path, make_extension)
        try:
            assert events == [SessionStartEvent(reason="startup")]
            events.clear()

            await _wait(runtime.session.prompt("hello"))
            original_session_file = runtime.session.session_file

            result = await _wait(runtime.new_session())
            assert result == {"cancelled": True}
            assert runtime.session.session_file == original_session_file
            assert events == [SessionBeforeSwitchEvent(reason="new", target_session_file=None)]
        finally:
            await _wait(runtime.dispose())

    async def test_runs_before_session_invalidate_after_session_shutdown_and_before_rebind_session(
        self, tmp_path: Path
    ) -> None:
        phases: list[str] = []

        def make_extension() -> Extension:
            return _handlers_extension({"session_shutdown": [lambda event, ctx: phases.append("session_shutdown")]})

        runtime, _factory = await create_runtime_host(tmp_path, make_extension)
        old_session = runtime.session
        try:

            def before_invalidate() -> None:
                phases.append("before_session_invalidate")
                # The outgoing session is still usable at this point: that is
                # the whole reason the callback is synchronous.
                assert old_session.extension_runner.create_context().cwd == old_session.session_manager.get_cwd()

            async def on_rebind(_session: AgentSession) -> None:
                phases.append("rebind_session")

            runtime.set_before_session_invalidate(before_invalidate)
            runtime.set_rebind_session(on_rebind)

            await _wait(runtime.new_session())

            assert phases == ["session_shutdown", "before_session_invalidate", "rebind_session"]
            # TS additionally asserts that `oldSession.extensionRunner.createContext()`
            # now throws "This extension ctx is stale after session replacement or
            # reload...". That staleness guard steers authors towards `withSession`,
            # which belongs to the unported extension UI host; this port's runner has
            # no stale state, so the context stays usable. See
            # `agent_session_runtime.py`'s module docstring.
            runtime.set_before_session_invalidate(None)
            runtime.set_rebind_session(None)
        finally:
            await _wait(runtime.dispose())

    async def test_emits_session_before_fork_and_session_start_and_honors_cancellation(self, tmp_path: Path) -> None:
        events: list[Any] = []
        cancel_next_fork = {"value": False}

        def on_before_fork(event, ctx) -> SessionBeforeForkResult | None:
            events.append(event)
            if cancel_next_fork["value"]:
                cancel_next_fork["value"] = False
                return SessionBeforeForkResult(cancel=True)
            return None

        def make_extension() -> Extension:
            return _handlers_extension(
                {
                    "session_before_fork": [on_before_fork],
                    "session_shutdown": [lambda event, ctx: events.append(event)],
                    "session_start": [lambda event, ctx: events.append(event)],
                }
            )

        runtime, _factory = await create_runtime_host(tmp_path, make_extension)
        try:
            assert events == [SessionStartEvent(reason="startup")]
            events.clear()

            await _wait(runtime.session.prompt("hello"))
            user_message = runtime.session.get_user_messages_for_forking()[0]
            previous_session_file = runtime.session.session_file

            success_result = await _wait(runtime.fork(user_message["entryId"]))
            assert success_result["cancelled"] is False
            assert success_result["selected_text"] == "hello"
            await _wait(runtime.session.bind_extensions())
            assert events == [
                SessionBeforeForkEvent(entry_id=user_message["entryId"], position="before"),
                SessionShutdownEvent(reason="fork", target_session_file=runtime.session.session_file),
                SessionStartEvent(reason="fork", previous_session_file=previous_session_file),
            ]

            events.clear()
            cancel_next_fork["value"] = True
            cancel_result = await _wait(runtime.fork(user_message["entryId"]))
            assert cancel_result == {"cancelled": True}
            assert events == [SessionBeforeForkEvent(entry_id=user_message["entryId"], position="before")]

            events.clear()
            cancel_next_fork["value"] = True
            cancel_at_result = await _wait(runtime.fork("missing-entry", position="at"))
            assert cancel_at_result == {"cancelled": True}
            assert events == [SessionBeforeForkEvent(entry_id="missing-entry", position="at")]
        finally:
            await _wait(runtime.dispose())
