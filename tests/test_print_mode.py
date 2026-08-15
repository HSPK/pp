"""Python port of `packages/coding-agent/test/print-mode.test.ts`.

The TypeScript test asserts that `runPrintMode` fires the extension runner's
`session_shutdown` event exactly once. It does so against a hand-built fake
runtime host whose `dispose()` records the emit itself, so it never proves the
real host emits anything.

This port drives the real objects instead: a real `AgentSessionRuntime` built
by `create_agent_session_runtime`, wrapping a real `AgentSession` with a real
`ExtensionRunner` holding a real `Extension`. Only the model is scripted (no
network call, no provider request). That makes the same assertions load-bearing
-- `session_shutdown` has to travel from `AgentSessionRuntime.dispose()`
through the runner to a handler -- and it additionally pins `session_start`,
which TypeScript's `rebindSession()` emits via `session.bindExtensions()` at
the top of `runPrintMode`.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

from pi_agent.agent import Agent, MutableAgentState
from pi_ai.types import AssistantMessage, ImageContent, TextContent
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
    SessionShutdownEvent,
    SessionStartEvent,
)
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionResult
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.print_mode import PrintModeOptions, run_print_mode


class _RecordingRuntimeFactory:
    """Builds a real `AgentSession` per replacement, with a real recording `Extension`."""

    def __init__(self, tmp_path: Path, model_runtime: ModelRuntime, response: AssistantMessage) -> None:
        self._tmp_path = tmp_path
        self._model_runtime = model_runtime
        self._response = response
        self.events: list[Any] = []
        self.sessions: list[AgentSession] = []

    def _extension(self) -> Extension:
        async def on_session_start(event: SessionStartEvent, _ctx: object) -> None:
            self.events.append(event)

        async def on_session_shutdown(event: SessionShutdownEvent, _ctx: object) -> None:
            self.events.append(event)

        return Extension(
            path="print-mode-ext.py",
            resolved_path="/test/print-mode-ext.py",
            handlers={
                "session_start": [on_session_start],
                "session_shutdown": [on_session_shutdown],
            },
        )

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
            _scripted_stream_fn([self._response]),
            initial_state=MutableAgentState(model=TEST_MODEL, system_prompt="You are a test assistant."),
        )
        session = AgentSession(
            agent=agent,
            session_manager=session_manager,
            settings_manager=SettingsManager.in_memory(),
            cwd=cwd,
            resource_loader=resource_loader,
            model_runtime=self._model_runtime,
            base_tools_override={},
            extensions=[self._extension()],
            session_start_event=session_start_event,
        )
        self.sessions.append(session)
        return CreateAgentSessionResult(session=session, model_fallback_message=None)


async def _create_runtime_host(
    tmp_path: Path, response: AssistantMessage
) -> tuple[AgentSessionRuntime, _RecordingRuntimeFactory]:
    model_runtime = await _make_model_runtime(tmp_path)
    factory = _RecordingRuntimeFactory(tmp_path, model_runtime, response)
    session_manager = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"))
    runtime = await _wait(
        create_agent_session_runtime(
            factory, cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=session_manager
        )
    )
    return runtime, factory


def _shutdown_events(factory: _RecordingRuntimeFactory) -> list[SessionShutdownEvent]:
    return [event for event in factory.events if isinstance(event, SessionShutdownEvent)]


def _start_events(factory: _RecordingRuntimeFactory) -> list[SessionStartEvent]:
    return [event for event in factory.events if isinstance(event, SessionStartEvent)]


def _prompt_recorder(session: AgentSession) -> tuple[list[tuple[str, Any]], Callable[[], None]]:
    """Records the arguments `run_print_mode` passes to the *real* `session.prompt`.

    The wrapper delegates to the real coroutine, so the prompt still runs
    through the real agent loop -- unlike TS's `vi.fn(async () => {})`, which
    replaces it.
    """
    calls: list[tuple[str, Any]] = []
    original = session.prompt

    async def prompt(message: str, images: Any = None) -> None:
        calls.append((message, images))
        await original(message, images=images)

    session.prompt = prompt  # type: ignore[method-assign]
    return calls, lambda: setattr(session, "prompt", original)


async def test_emits_session_shutdown_in_text_mode(tmp_path: Path) -> None:
    runtime_host, factory = await _create_runtime_host(tmp_path, _make_assistant_message([TextContent(text="done")]))
    calls, _restore = _prompt_recorder(runtime_host.session)
    images = [ImageContent(mime_type="image/png", data="abc")]

    exit_code = await _wait(
        run_print_mode(runtime_host, PrintModeOptions(mode="text", initial_message="Say done", initial_images=images))
    )

    assert exit_code == 0
    assert calls == [("Say done", images)]
    assert len(_start_events(factory)) == 1
    assert [event.reason for event in _shutdown_events(factory)] == ["quit"]


async def test_emits_session_shutdown_in_json_mode(tmp_path: Path) -> None:
    runtime_host, factory = await _create_runtime_host(tmp_path, _make_assistant_message([TextContent(text="done")]))
    calls, _restore = _prompt_recorder(runtime_host.session)

    exit_code = await _wait(run_print_mode(runtime_host, PrintModeOptions(mode="json", messages=["hello"])))

    assert exit_code == 0
    assert calls == [("hello", None)]
    assert len(_start_events(factory)) == 1
    assert [event.reason for event in _shutdown_events(factory)] == ["quit"]


async def test_emits_session_shutdown_and_returns_non_zero_on_assistant_error(tmp_path: Path) -> None:
    runtime_host, factory = await _create_runtime_host(
        tmp_path, _make_assistant_message([], stop_reason="error", error_message="provider failure")
    )

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        exit_code = await _wait(run_print_mode(runtime_host, PrintModeOptions(mode="text", messages=["hello"])))

    assert exit_code == 1
    assert stderr.getvalue().strip() == "provider failure"
    assert len(_start_events(factory)) == 1
    assert [event.reason for event in _shutdown_events(factory)] == ["quit"]


async def test_rebinds_to_the_replacement_session(tmp_path: Path) -> None:
    """TS registers `rebindSession` with the host so a mid-run session replacement re-attaches.

    `print-mode.ts` calls `runtimeHost.setRebindSession(...)`, and its
    `rebindSession` reassigns `session = runtimeHost.session` before binding
    extensions and re-subscribing. Without both halves the mode keeps prompting
    and reading the disposed session, and the replacement never gets its
    `session_start`.
    """
    runtime_host, factory = await _create_runtime_host(tmp_path, _make_assistant_message([TextContent(text="done")]))

    first_session = runtime_host.session
    original_prompt = first_session.prompt
    replaced = False

    async def prompt_then_replace(message: str, images: Any = None) -> None:
        nonlocal replaced
        await original_prompt(message, images=images)
        if not replaced:
            replaced = True
            await runtime_host.new_session()

    first_session.prompt = prompt_then_replace  # type: ignore[method-assign]

    exit_code = await _wait(run_print_mode(runtime_host, PrintModeOptions(mode="text", messages=["first", "second"])))

    assert exit_code == 0
    assert len(factory.sessions) == 2
    assert runtime_host.session is factory.sessions[1]
    # `session_start` for the startup session, then for the replacement: the
    # second one can only come from the registered rebind callback calling
    # `bind_extensions()` on the new session.
    assert [event.reason for event in _start_events(factory)] == ["startup", "new"]
    assert [event.reason for event in _shutdown_events(factory)] == ["new", "quit"]
    # The second prompt has to land on the replacement session, which proves
    # `rebind_session` reassigned the mode's local `session`.
    assert [
        part.text
        for message in factory.sessions[1].state.messages
        if message.role == "user"
        for part in message.content
    ] == ["second"]
