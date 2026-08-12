"""Tests for `pi_coding_agent.core.agent_session_runtime`.

Ported from `packages/coding-agent/test/*agent-session-runtime*` (TS's
`AgentSessionRuntime` session-lifecycle wrapper), adapted to this port's
narrower surface: there is no `AgentSessionServices` split, so the runtime
factory here builds an `AgentSession` directly. The extension
`session_before_switch`/`session_before_fork` cancellation hooks *are* ported
and are covered by `test_agent_session_runtime_events.py`.

The highest-value test in this module is
`test_end_to_end_real_server_real_runtime_real_client_scripted_model`: it
starts a REAL `pi_server` (over a Unix socket in `tmp_path`), driven by the
REAL `PiAgentSessionRuntime` adapter wrapping a REAL `AgentSession`, connected
to the REAL `pi_client.PiClient`. The only thing faked is the model response
(a scripted `stream_fn`, mirroring `packages/pi-agent/tests/conftest.py` and
`test_agent_session.py`'s `scripted_stream_fn` pattern) -- there is no real
HTTP/network I/O anywhere in this module, and every awaited call is bounded
by `asyncio.wait_for`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pi_agent.agent import Agent, MutableAgentState
from pi_agent.types import AgentTool, AgentToolResult, MessageEndEvent, MessageUpdateEvent
from pi_ai.providers import openai_compatible_provider
from pi_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_client import PiClient, PiClientOptions, create_unix_transport_factory
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.agent_session_runtime import (
    AgentSessionRuntime,
    PiAgentSessionRuntime,
    PiAgentSessionRuntimeService,
    SessionImportFileNotFoundError,
    create_agent_session_runtime,
)
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import (
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
    create_agent_session,
)
from pi_coding_agent.core.session_cwd import MissingSessionCwdError
from pi_coding_agent.core.session_manager import NewSessionOptions, SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_server.errors import PiServerError
from pi_server.transports.unix import UnixServerOptions, create_unix_server
from pi_server.types import CreateSessionOptions

TIMEOUT = 5.0

TEST_MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="test",
    base_url="https://fake.example.com",
    context_window=1000,
    max_tokens=100,
)


async def _wait(awaitable: Any, timeout: float = TIMEOUT) -> Any:
    return await asyncio.wait_for(awaitable, timeout=timeout)


def _make_assistant_message(
    content: list[Any], stop_reason: str = "stop", error_message: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=content,
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _replay_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    """Emit the protocol event sequence that produces `message` (mirrors `test_agent_session.py`)."""
    stream = AssistantMessageEventStream()
    # Intermediate events carry a "pending" partial (distinct from `message`,
    # whose `stop_reason` is already final) so `PiAgentSessionRuntime` can be
    # observed transitioning through its "streaming" status, matching how a
    # real provider's partial differs from the final message.
    partial = dataclasses.replace(message, stop_reason="pending")
    stream.push(StartEvent(partial=partial))
    for index, block in enumerate(message.content):
        if block.type == "text":
            stream.push(TextStartEvent(content_index=index, partial=partial))
            stream.push(TextDeltaEvent(content_index=index, delta=block.text, partial=partial))
            stream.push(TextEndEvent(content_index=index, content=block.text, partial=partial))
        elif block.type == "toolCall":
            stream.push(ToolCallStartEvent(content_index=index, partial=partial))
            stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=partial))
    if message.stop_reason in ("error", "aborted"):
        stream.push(ErrorEvent(reason=message.stop_reason, error=message))
    else:
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


def _scripted_stream_fn(responses: list[AssistantMessage]) -> Callable[..., Any]:
    remaining = list(responses)

    def stream_fn(model: Any, context: Any, options: Any = None) -> Any:
        if not remaining:
            raise AssertionError("stream_fn called more times than there are scripted responses")
        return _replay_stream(remaining.pop(0))

    return stream_fn


_SECOND_MODEL = Model(
    id="second-model",
    name="Second Test Model",
    api="openai-completions",
    context_window=1000,
    max_tokens=100,
    cost=ModelCost(input=0, output=0),
    reasoning=True,
    thinking_level_map={"minimal": "low", "low": "low", "medium": "medium", "high": "high"},
)


def _fake_provider(extra_models: list[Model] | None = None) -> object:
    return openai_compatible_provider(
        provider_id="test",
        name="Fake Test Provider",
        base_url="https://fake.example.com",
        env_vars=["FAKE_TEST_API_KEY"],
        models=[
            Model(
                id="test-model",
                name="Test Model",
                api="openai-completions",
                context_window=1000,
                max_tokens=100,
                cost=ModelCost(input=0, output=0),
            ),
            *(extra_models or []),
        ],
    )


async def _make_service(tmp_path: Path) -> PiAgentSessionRuntimeService:
    """A real `PiAgentSessionRuntimeService`, with a fake provider registered so
    `create_agent_session`'s model resolution has something to select (no
    network I/O: `login` only writes a local credential file)."""
    model_runtime = await _wait(ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[_fake_provider()]))
    await _wait(model_runtime.login("test", "fake-key"))
    return PiAgentSessionRuntimeService(
        agent_dir=str(tmp_path / "agent"), default_cwd=str(tmp_path), model_runtime=model_runtime
    )


def _make_echo_tool() -> AgentTool:
    async def execute(tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None) -> AgentToolResult:
        text = params.get("text", "") if isinstance(params, dict) else ""
        return AgentToolResult(content=[TextContent(text=f"echo:{text}")])

    return AgentTool(
        name="echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        label="Echo",
        execute=execute,
    )


async def _build_real_session(
    tmp_path: Path,
    responses: list[AssistantMessage],
    *,
    session_id: str = "session-1",
    tool: AgentTool | None = None,
    extra_provider_models: list[Model] | None = None,
) -> AgentSession:
    """A real `AgentSession` wired to a scripted `stream_fn`, sandboxed under `tmp_path`.

    Registers `_fake_provider()` and logs in (writes a local credential file
    only, no network I/O) so `AgentSession.prompt`'s auth check passes.
    """
    model_runtime = await _wait(
        ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[_fake_provider(extra_provider_models)])
    )
    await _wait(model_runtime.login("test", "fake-key"))
    session_manager = SessionManager.in_memory(str(tmp_path), NewSessionOptions(id=session_id))
    settings_manager = SettingsManager.in_memory()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=str(tmp_path), agent_dir=str(agent_dir)))
    resource_loader.reload()

    agent = Agent(
        _scripted_stream_fn(responses),
        initial_state=MutableAgentState(model=TEST_MODEL, system_prompt="You are a test assistant."),
    )
    echo_tool = tool or _make_echo_tool()
    return AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=settings_manager,
        cwd=str(tmp_path),
        resource_loader=resource_loader,
        model_runtime=model_runtime,
        custom_tools={"echo": echo_tool},
        initial_active_tool_names=["echo"],
        base_tools_override={},
    )


# ---------------------------------------------------------------------------
# PiAgentSessionRuntime: unit tests against a directly-constructed session
# ---------------------------------------------------------------------------


async def test_snapshot_has_the_shape_required_by_pi_server(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [], session_id="session-1")
    runtime = PiAgentSessionRuntime(session)
    try:
        snapshot = runtime.snapshot()
        assert snapshot["id"] == "session-1"
        assert snapshot["phase"] == "idle"
        assert snapshot["transcript"] == []
        assert snapshot["queuedSteer"] == []
        assert snapshot["queuedSteerCount"] == 0
        assert snapshot["model"] == {"provider": "test", "id": "test-model"}
    finally:
        await _wait(runtime.dispose())


async def test_prompt_builds_a_user_and_assistant_transcript_item(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [_make_assistant_message([TextContent(text="hi there")])])
    runtime = PiAgentSessionRuntime(session)
    try:
        await _wait(runtime.prompt(_prompt_input("hello")))
        snapshot = runtime.snapshot()
        assert snapshot["phase"] == "idle"
        roles = [item["role"] for item in snapshot["transcript"]]
        assert roles == ["user", "assistant"]
        assert snapshot["transcript"][0]["content"] == [{"type": "text", "text": "hello"}]
        assert snapshot["transcript"][1]["status"] == "complete"
        assert snapshot["transcript"][1]["content"] == [{"type": "text", "text": "hi there"}]
    finally:
        await _wait(runtime.dispose())


async def test_prompt_while_busy_raises_a_protocol_error(tmp_path: Path) -> None:
    # Gate the tool's `execute` on a real await so the turn is genuinely
    # in-flight (not finished) when the second `prompt()` call races it --
    # a scripted stream_fn resolves so fast there's otherwise no reliable
    # window to observe `phase == "turn"`.
    release_event = asyncio.Event()

    async def gated_execute(
        tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None
    ) -> AgentToolResult:
        await release_event.wait()
        return AgentToolResult(content=[TextContent(text="done")])

    gated_tool = AgentTool(
        name="echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        label="Echo",
        execute=gated_execute,
    )
    tool_call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    responses = [
        _make_assistant_message([tool_call], stop_reason="toolUse"),
        _make_assistant_message([TextContent(text="ok")]),
    ]
    session = await _build_real_session(tmp_path, responses, tool=gated_tool)
    runtime = PiAgentSessionRuntime(session)
    try:
        prompt_task = asyncio.ensure_future(runtime.prompt(_prompt_input("first")))
        async with asyncio.timeout(TIMEOUT):
            while runtime.get_phase() != "turn":
                await asyncio.sleep(0.005)
        with pytest.raises(PiServerError, match="already running"):
            await _wait(runtime.prompt(_prompt_input("second")))
        release_event.set()
        await _wait(prompt_task)
    finally:
        await _wait(runtime.dispose())


async def test_tool_call_and_result_appear_in_the_transcript(tmp_path: Path) -> None:
    tool_call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    responses = [
        _make_assistant_message([tool_call], stop_reason="toolUse"),
        _make_assistant_message([TextContent(text="done")]),
    ]
    session = await _build_real_session(tmp_path, responses)
    runtime = PiAgentSessionRuntime(session)
    progress_events: list[dict[str, Any]] = []
    runtime.subscribe(lambda event: progress_events.append(event) if event.type == "progress" else None)
    try:
        await _wait(runtime.prompt(_prompt_input("please echo")))
        snapshot = runtime.snapshot()
        tool_items = [item for item in snapshot["transcript"] if item["role"] == "tool"]
        assert len(tool_items) == 1
        assert tool_items[0]["toolName"] == "echo"
        assert tool_items[0]["status"] == "complete"
        assert tool_items[0]["content"] == [{"type": "text", "text": "echo:hi"}]

        item_started_tool_events = [
            event
            for event in progress_events
            if event.progress["type"] == "item_started" and event.progress["item"]["role"] == "tool"
        ]
        assert len(item_started_tool_events) == 1
        assert item_started_tool_events[0].progress["item"]["status"] == "running"
    finally:
        await _wait(runtime.dispose())


async def test_streamed_assistant_deltas_are_reported_as_progress(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [_make_assistant_message([TextContent(text="hello world")])])
    runtime = PiAgentSessionRuntime(session)
    progress_events: list[dict[str, Any]] = []
    runtime.subscribe(lambda event: progress_events.append(event.progress) if event.type == "progress" else None)
    try:
        await _wait(runtime.prompt(_prompt_input("hi")))
        deltas = [event for event in progress_events if event["type"] == "assistant_delta"]
        assert any(event["kind"] == "text" and event["delta"] == "hello world" for event in deltas)
        started = [event for event in progress_events if event["type"] == "item_started"]
        assert any(item["item"].get("status") == "streaming" for item in started)
        finished = [event for event in progress_events if event["type"] == "item_finished"]
        assert any(item["item"]["status"] == "complete" for item in finished)
    finally:
        await _wait(runtime.dispose())


async def test_abort_settles_a_running_prompt(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [_make_assistant_message([TextContent(text="ok")])])
    runtime = PiAgentSessionRuntime(session)
    try:
        prompt_task = asyncio.ensure_future(runtime.prompt(_prompt_input("hello")))
        await _wait(runtime.abort())
        await _wait(prompt_task)
        assert runtime.get_phase() == "idle"
    finally:
        await _wait(runtime.dispose())


async def test_set_model_rejects_an_unknown_model(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [])
    runtime = PiAgentSessionRuntime(session)
    try:
        with pytest.raises(PiServerError, match="Unknown model"):
            await _wait(runtime.set_model({"provider": "nope", "id": "nope"}))
    finally:
        await _wait(runtime.dispose())


def _prompt_input(text: str) -> Any:
    from pi_server.types import PromptInput

    return PromptInput(text=text)


# ---------------------------------------------------------------------------
# PiAgentSessionRuntimeService: unit tests, no network (never calls .prompt())
# ---------------------------------------------------------------------------


async def test_service_create_session_uses_the_server_assigned_id(tmp_path: Path) -> None:
    service = await _make_service(tmp_path)
    runtime = await _wait(service.create_session(CreateSessionOptions(id="abc-123", cwd=str(tmp_path), name="hello")))
    try:
        snapshot = runtime.snapshot()
        assert snapshot["id"] == "abc-123"
        assert snapshot["name"] == "hello"
        assert snapshot["cwd"] == str(tmp_path)
    finally:
        await _wait(runtime.dispose())


async def test_service_list_sessions_reflects_created_sessions(tmp_path: Path) -> None:
    """`list_sessions` reads from disk; a session isn't flushed there until it
    has a completed assistant message (`SessionManager._persist_entry`'s
    `has_assistant` gate -- a session that never receives a response is never
    written). So this drives one full (scripted, no-network) prompt through a
    directly-built `AgentSession` to produce a real on-disk session file, then
    asserts the SAME `PiAgentSessionRuntimeService`'s `list_sessions` (its
    read path) finds it -- rather than routing the prompt through the
    service's own `create_session` (which would need a real network call)."""
    session_dir = tmp_path / "sessions"
    model_runtime = await _wait(ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[_fake_provider()]))
    await _wait(model_runtime.login("test", "fake-key"))
    session_manager = SessionManager.create(
        str(tmp_path), session_dir=str(session_dir), options=NewSessionOptions(id="session-xyz")
    )
    settings_manager = SettingsManager.in_memory()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=str(tmp_path), agent_dir=str(agent_dir)))
    resource_loader.reload()
    agent = Agent(
        _scripted_stream_fn([_make_assistant_message([TextContent(text="hi")])]),
        initial_state=MutableAgentState(model=TEST_MODEL, system_prompt="You are a test assistant."),
    )
    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=settings_manager,
        cwd=str(tmp_path),
        resource_loader=resource_loader,
        model_runtime=model_runtime,
        base_tools_override={},
    )
    await _wait(session.prompt("hello"))

    service = PiAgentSessionRuntimeService(
        agent_dir=str(agent_dir), default_cwd=str(tmp_path), model_runtime=model_runtime, session_dir=str(session_dir)
    )
    sessions = await _wait(service.list_sessions())
    assert any(entry["id"] == "session-xyz" for entry in sessions)


async def test_service_open_session_rejects_an_already_live_session(tmp_path: Path) -> None:
    service = await _make_service(tmp_path)
    runtime = await _wait(service.create_session(CreateSessionOptions(id="session-open", cwd=str(tmp_path))))
    try:
        with pytest.raises(PiServerError, match="is locked"):
            await _wait(service.open_session("session-open"))
    finally:
        await _wait(runtime.dispose())


async def test_service_open_session_reports_not_found_for_unknown_ids(tmp_path: Path) -> None:
    service = await _make_service(tmp_path)
    with pytest.raises(PiServerError, match="Unknown session"):
        await _wait(service.open_session("no-such-session"))


async def test_service_list_models_reports_the_configured_snapshot(tmp_path: Path) -> None:
    service = await _make_service(tmp_path)
    models = await _wait(service.list_models())
    assert isinstance(models, list)
    assert any(entry["provider"] == "test" and entry["id"] == "test-model" for entry in models)


# ---------------------------------------------------------------------------
# End-to-end: real pi_server + real PiAgentSessionRuntime + real pi_client
# ---------------------------------------------------------------------------


class _E2EService:
    """Minimal `PiServerService` wrapping REAL `AgentSession`s built with a scripted model.

    `create_session` bypasses `sdk.create_agent_session`/`PiAgentSessionRuntimeService`
    (which would call `ModelRuntime.stream_simple` -- real network I/O) and instead
    builds a real `AgentSession` directly, exactly as `_build_real_session` does above,
    so the model is scripted rather than a real provider. The runtime it returns
    (`PiAgentSessionRuntime`) is the same real adapter `PiAgentSessionRuntimeService` uses.
    """

    def __init__(self, tmp_path: Path, responses_by_prompt_count: dict[int, list[AssistantMessage]]) -> None:
        self._tmp_path = tmp_path
        self._responses_by_prompt_count = responses_by_prompt_count
        self._runtimes: dict[str, PiAgentSessionRuntime] = {}

    async def list_sessions(self) -> list[dict[str, Any]]:
        return []

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def create_session(self, options: CreateSessionOptions) -> PiAgentSessionRuntime:
        responses = self._responses_by_prompt_count.get(len(self._runtimes), [])
        session = await _build_real_session(self._tmp_path, responses, session_id=options.id)
        runtime = PiAgentSessionRuntime(
            session, name=options.name, on_dispose=lambda: self._runtimes.pop(options.id, None)
        )
        self._runtimes[options.id] = runtime
        return runtime

    async def open_session(self, session_id: str) -> PiAgentSessionRuntime:
        raise PiServerError("not_found", f"Unknown session: {session_id}")


async def test_end_to_end_real_server_real_runtime_real_client_scripted_model(tmp_path: Path) -> None:
    """The highest-value test: proves the real RPC stack works end to end.

    Real `pi_server` (Unix socket) <-> real `PiAgentSessionRuntime` (wrapping a
    real `AgentSession`) <-> real `pi_client`. Covers: create a session,
    prompt it, receive streamed transcript progress on the client, a tool call
    and its result appearing in the transcript, and the final assistant
    message -- with the model provided by a scripted `stream_fn`.
    """
    tool_call = ToolCall(id="call-1", name="echo", arguments={"text": "world"})
    responses = {
        0: [
            _make_assistant_message([tool_call], stop_reason="toolUse"),
            _make_assistant_message([TextContent(text="all done")]),
        ]
    }
    service = _E2EService(tmp_path, responses)
    # A Unix socket path is capped at 107 bytes. `tmp_path` already spends most
    # of that on the pytest/xdist-worker prefix, so the socket gets its own
    # short directory instead of overflowing under a parallel run.
    socket_dir = tempfile.mkdtemp(prefix="pi-t-")
    socket_path = str(Path(socket_dir) / "e2e.sock")
    server_errors: list[Exception] = []
    server = create_unix_server(service, UnixServerOptions(path=socket_path, on_error=server_errors.append))
    await _wait(server.start())
    client = PiClient(PiClientOptions(transport_factory=create_unix_transport_factory(socket_path)))
    try:
        await _wait(client.connect())
        assert client.connected is True

        handle = await _wait(client.create_session(cwd=str(tmp_path)))
        assert handle.snapshot is not None
        assert handle.snapshot["phase"] == "idle"

        progress_events: list[dict[str, Any]] = []
        unsubscribe = handle.on_event(lambda event: progress_events.append(event))

        session = await _wait(handle.prompt("please echo hello"), timeout=10.0)

        # Progress events are best-effort/fire-and-forget relative to the
        # command response (`TRANSCRIPT_PROGRESS_SCHEMA`'s docstring: "snapshots
        # remain authoritative"), so a straggler can still be in flight when
        # `prompt()` resolves. Poll briefly for the final delta, mirroring
        # `test_integration.py`'s `while not any(...): await asyncio.sleep(0.01)`.
        async def _assistant_deltas() -> list[dict[str, Any]]:
            return [
                event["progress"]
                for event in progress_events
                if event.get("type") == "session_progress" and event["progress"]["type"] == "assistant_delta"
            ]

        async with asyncio.timeout(TIMEOUT):
            while not any(delta["delta"] == "all done" for delta in await _assistant_deltas()):
                await asyncio.sleep(0.01)

        # Streamed transcript progress reached the client.
        assert any(event.get("type") == "session_progress" for event in progress_events)

        # A tool call and its result appear in the final transcript.
        tool_items = [item for item in session["transcript"] if item["role"] == "tool"]
        assert len(tool_items) == 1
        assert tool_items[0]["toolName"] == "echo"
        assert tool_items[0]["status"] == "complete"
        assert tool_items[0]["content"] == [{"type": "text", "text": "echo:world"}]

        # The final assistant message appears, complete.
        assistant_items = [item for item in session["transcript"] if item["role"] == "assistant"]
        assert assistant_items[-1]["status"] == "complete"
        assert assistant_items[-1]["content"] == [{"type": "text", "text": "all done"}]
        assert session["phase"] == "idle"

        unsubscribe()
    finally:
        await _wait(client.dispose())
        await _wait(server.close())
        shutil.rmtree(socket_dir, ignore_errors=True)
    if server_errors:
        raise server_errors[0]


# ---------------------------------------------------------------------------
# AgentSessionRuntime: the ported session-lifecycle wrapper
#
# Mirrors `packages/coding-agent/test/suite/agent-session-runtime.test.ts`
# ("settles the active response before session replacement", "reports why an
# unflushed session cannot be forked", "duplicates the current active branch
# ... when forking at the current position", "throws when forking with an
# invalid entry id", "updates the runtime session cwd on cross-cwd session
# replacement", "restores model and thinking state from the destination
# session"). That file's three extension-event cases
# (`session_before_switch`/`session_before_fork` emission and cancellation)
# live in `test_agent_session_runtime_events.py`, and its "persists message_end
# assistant replacements to the session manager" case is pinned by
# `test_agent_session.py::test_message_end_extension_hook_replacement_is_persisted`.
# ---------------------------------------------------------------------------


async def _make_model_runtime(tmp_path: Path) -> ModelRuntime:
    """A `ModelRuntime` over the fake provider, credentialed under `tmp_path` (never `$HOME`)."""
    model_runtime = await _wait(ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[_fake_provider()]))
    await _wait(model_runtime.login("test", "fake-key"))
    return model_runtime


def _build_session_for(
    tmp_path: Path,
    model_runtime: ModelRuntime,
    session_manager: SessionManager,
    responses: list[AssistantMessage],
    *,
    cwd: str | None = None,
    tool: AgentTool | None = None,
    settings: dict[str, Any] | None = None,
) -> AgentSession:
    """A real `AgentSession` over `session_manager`, driven by a scripted `stream_fn`.

    Seeds `agent.state.messages` from the session manager the way
    `sdk.create_agent_session` does, so a resumed/forked session manager
    produces a session with the restored transcript.
    """
    session_cwd = cwd or session_manager.get_cwd()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=session_cwd, agent_dir=str(agent_dir)))
    resource_loader.reload()
    agent = Agent(
        _scripted_stream_fn(responses),
        initial_state=MutableAgentState(model=TEST_MODEL, system_prompt="You are a test assistant."),
    )
    existing = session_manager.build_session_context()
    if existing.messages:
        agent.state.messages = existing.messages
    return AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=SettingsManager.in_memory(settings),
        cwd=session_cwd,
        resource_loader=resource_loader,
        model_runtime=model_runtime,
        custom_tools={"echo": tool or _make_echo_tool()},
        initial_active_tool_names=["echo"],
        base_tools_override={},
    )


class _RecordingRuntimeFactory:
    """A `CreateAgentSessionRuntimeFactory` that records every call and builds a real session.

    Stands in for `sdk.create_agent_session` (which would resolve a real
    provider) while keeping everything the runtime itself touches -- the
    `AgentSession`, its `SessionManager`, the model runtime -- real.
    """

    def __init__(
        self,
        tmp_path: Path,
        model_runtime: ModelRuntime,
        *,
        responses: list[AssistantMessage] | None = None,
        responses_per_session: list[list[AssistantMessage]] | None = None,
        tool: AgentTool | None = None,
        model_fallback_message: str | None = None,
    ) -> None:
        self._tmp_path = tmp_path
        self._model_runtime = model_runtime
        self._responses = responses or []
        # One scripted response list per created session, when a test needs the
        # replacement session to answer differently from the first one.
        self._responses_per_session = list(responses_per_session or [])
        self._tool = tool
        self._model_fallback_message = model_fallback_message
        self.calls: list[dict[str, Any]] = []
        self.sessions: list[AgentSession] = []

    async def __call__(
        self, *, cwd: str, agent_dir: str, session_manager: SessionManager, **_ignored: Any
    ) -> CreateAgentSessionResult:
        self.calls.append({"cwd": cwd, "agent_dir": agent_dir, "session_manager": session_manager})
        responses = self._responses_per_session.pop(0) if self._responses_per_session else list(self._responses)
        session = _build_session_for(
            self._tmp_path, self._model_runtime, session_manager, responses, cwd=cwd, tool=self._tool
        )
        self.sessions.append(session)
        return CreateAgentSessionResult(session=session, model_fallback_message=self._model_fallback_message)


async def _make_runtime(
    tmp_path: Path,
    *,
    persisted: bool = False,
    responses: list[AssistantMessage] | None = None,
    responses_per_session: list[list[AssistantMessage]] | None = None,
    tool: AgentTool | None = None,
    model_fallback_message: str | None = None,
) -> tuple[AgentSessionRuntime, _RecordingRuntimeFactory]:
    model_runtime = await _make_model_runtime(tmp_path)
    factory = _RecordingRuntimeFactory(
        tmp_path,
        model_runtime,
        responses=responses,
        responses_per_session=responses_per_session,
        tool=tool,
        model_fallback_message=model_fallback_message,
    )
    session_manager = (
        SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"))
        if persisted
        else SessionManager.in_memory(str(tmp_path))
    )
    runtime = await _wait(
        create_agent_session_runtime(
            factory, cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=session_manager
        )
    )
    return runtime, factory


def _user_texts(session: AgentSession) -> list[str]:
    texts: list[str] = []
    for message in session.messages:
        if getattr(message, "role", None) != "user":
            continue
        content = message.content
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.append("".join(part.text for part in content if getattr(part, "type", None) == "text"))
    return texts


async def test_create_agent_session_runtime_applies_the_factory_result(tmp_path: Path) -> None:
    runtime, factory = await _make_runtime(tmp_path, model_fallback_message="using fallback model")
    try:
        assert runtime.session is factory.sessions[0]
        assert runtime.agent_dir == str(tmp_path / "agent")
        assert runtime.cwd == str(tmp_path)
        assert runtime.model_fallback_message == "using fallback model"
        assert factory.calls[0]["cwd"] == str(tmp_path)
        assert factory.calls[0]["agent_dir"] == str(tmp_path / "agent")
    finally:
        await _wait(runtime.dispose())


async def test_create_agent_session_runtime_rejects_a_session_whose_cwd_is_gone(tmp_path: Path) -> None:
    """`assert_session_cwd_exists` runs before the factory: a stored cwd that no
    longer exists is reported instead of silently starting somewhere else."""
    missing_cwd = tmp_path / "gone"
    missing_cwd.mkdir()
    session_dir = tmp_path / "sessions"
    session_manager = SessionManager.create(str(missing_cwd), session_dir=str(session_dir))
    model_runtime = await _make_model_runtime(tmp_path)
    session = _build_session_for(
        tmp_path, model_runtime, session_manager, [_make_assistant_message([TextContent(text="hi")])]
    )
    await _wait(session.prompt("hello"))
    session_file = session.session_file
    assert session_file is not None
    session.dispose()
    missing_cwd.rmdir()

    factory = _RecordingRuntimeFactory(tmp_path, model_runtime)
    reopened = SessionManager.open(session_file)
    with pytest.raises(MissingSessionCwdError):
        await _wait(
            create_agent_session_runtime(
                factory, cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=reopened
            )
        )
    assert factory.calls == []


async def test_new_session_replaces_an_in_memory_session(tmp_path: Path) -> None:
    runtime, factory = await _make_runtime(tmp_path, responses=[_make_assistant_message([TextContent(text="one")])])
    try:
        first_session = runtime.session
        await _wait(first_session.prompt("hello"))
        assert _user_texts(first_session) == ["hello"]

        result = await _wait(runtime.new_session())

        assert result == {"cancelled": False}
        assert runtime.session is not first_session
        assert runtime.session.messages == []
        assert runtime.session.session_manager.is_persisted() is False
        assert len(factory.calls) == 2
        # The outgoing session was disposed: its listeners are gone.
        assert first_session._event_listeners == []
    finally:
        await _wait(runtime.dispose())


async def test_new_session_records_the_parent_session_on_a_persisted_session(tmp_path: Path) -> None:
    runtime, factory = await _make_runtime(
        tmp_path, persisted=True, responses=[_make_assistant_message([TextContent(text="one")])]
    )
    try:
        previous_file = runtime.session.session_file
        assert previous_file is not None

        await _wait(runtime.new_session(parent_session=previous_file))

        session_manager = runtime.session.session_manager
        assert session_manager.is_persisted() is True
        assert session_manager.get_session_file() != previous_file
        header = session_manager.get_header()
        assert header is not None
        assert header.parent_session == previous_file
        assert factory.calls[-1]["cwd"] == str(tmp_path)
    finally:
        await _wait(runtime.dispose())


async def test_switch_session_restores_a_persisted_session_file(tmp_path: Path) -> None:
    runtime, factory = await _make_runtime(
        tmp_path,
        persisted=True,
        responses=[
            _make_assistant_message([TextContent(text="one")]),
            _make_assistant_message([TextContent(text="two")]),
        ],
    )
    try:
        await _wait(runtime.session.prompt("first prompt"))
        first_file = runtime.session.session_file
        assert first_file is not None
        assert Path(first_file).exists()

        await _wait(runtime.new_session())
        assert runtime.session.session_file != first_file

        result = await _wait(runtime.switch_session(first_file))

        assert result == {"cancelled": False}
        assert runtime.session.session_file == first_file
        assert _user_texts(runtime.session) == ["first prompt"]
        assert factory.calls[-1]["session_manager"].get_cwd() == str(tmp_path)
    finally:
        await _wait(runtime.dispose())


async def test_switch_session_settles_the_outgoing_response_before_replacement(tmp_path: Path) -> None:
    """Port of TS's "settles the active response before session replacement".

    A tool blocked until abort is interrupted by the switch; the outgoing
    session must still persist the tool result rather than leaving the call
    dangling forever.
    """
    tool_started = asyncio.Event()

    async def blocking_execute(
        tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None
    ) -> AgentToolResult:
        tool_started.set()
        if signal is not None:
            await signal.wait()
        return AgentToolResult(content=[TextContent(text="tool aborted")])

    blocking_tool = AgentTool(
        name="echo",
        description="Blocks until aborted",
        parameters={"type": "object", "properties": {}},
        label="Block",
        execute=blocking_execute,
    )
    responses_per_session = [
        # First session: a plain text answer, so its file is flushed to disk.
        [_make_assistant_message([TextContent(text="one")])],
        # Outgoing session: a tool call that blocks until the switch aborts it.
        [
            _make_assistant_message([ToolCall(id="call-1", name="echo", arguments={})], stop_reason="toolUse"),
            _make_assistant_message([TextContent(text="after abort")]),
        ],
        # Session restored by the switch.
        [_make_assistant_message([TextContent(text="restored")])],
    ]
    runtime, _factory = await _make_runtime(
        tmp_path, persisted=True, responses_per_session=responses_per_session, tool=blocking_tool
    )
    try:
        await _wait(runtime.session.prompt("first prompt"))
        first_file = runtime.session.session_file
        assert first_file is not None

        await _wait(runtime.new_session())
        outgoing = runtime.session
        prompt_task = asyncio.ensure_future(outgoing.prompt("start blocking tool"))
        await _wait(tool_started.wait())

        result = await _wait(runtime.switch_session(first_file))
        await _wait(prompt_task)

        assert result == {"cancelled": False}
        assert runtime.session.session_file == first_file
        outgoing_file = outgoing.session_file
        assert outgoing_file is not None
        roles = [
            entry.message.role for entry in SessionManager.open(outgoing_file).get_entries() if entry.type == "message"
        ]
        assert "toolResult" in roles
    finally:
        await _wait(runtime.dispose())


async def test_fork_rejects_an_unknown_entry_id(tmp_path: Path) -> None:
    runtime, _factory = await _make_runtime(tmp_path)
    try:
        with pytest.raises(ValueError, match="Invalid entry ID for forking"):
            await _wait(runtime.fork("missing-entry"))
    finally:
        await _wait(runtime.dispose())


async def test_fork_before_rejects_an_entry_that_is_not_a_user_message(tmp_path: Path) -> None:
    runtime, _factory = await _make_runtime(tmp_path, responses=[_make_assistant_message([TextContent(text="one")])])
    try:
        await _wait(runtime.session.prompt("hello"))
        leaf_id = runtime.session.session_manager.get_leaf_id()
        assert leaf_id is not None

        with pytest.raises(ValueError, match="Invalid entry ID for forking"):
            await _wait(runtime.fork(leaf_id, position="before"))
    finally:
        await _wait(runtime.dispose())


async def test_fork_at_duplicates_the_active_branch_in_memory(tmp_path: Path) -> None:
    """Port of TS's "duplicates the current active branch in-memory when forking
    at the current position"."""
    runtime, factory = await _make_runtime(
        tmp_path,
        responses=[
            _make_assistant_message([TextContent(text="one")]),
            _make_assistant_message([TextContent(text="two")]),
        ],
    )
    try:
        await _wait(runtime.session.prompt("hello"))
        await _wait(runtime.session.prompt("again"))
        before_roles = [getattr(m, "role", None) for m in runtime.session.messages]
        before_users = _user_texts(runtime.session)
        leaf_id = runtime.session.session_manager.get_leaf_id()
        assert leaf_id is not None
        assert runtime.session.session_file is None

        result = await _wait(runtime.fork(leaf_id, position="at"))

        assert result == {"cancelled": False, "selected_text": None}
        assert runtime.session.session_file is None
        assert [getattr(m, "role", None) for m in runtime.session.messages] == before_roles
        assert _user_texts(runtime.session) == before_users
        assert len(factory.calls) == 2
    finally:
        await _wait(runtime.dispose())


async def test_fork_before_a_user_message_drops_it_and_returns_its_text(tmp_path: Path) -> None:
    runtime, _factory = await _make_runtime(
        tmp_path,
        responses=[
            _make_assistant_message([TextContent(text="one")]),
            _make_assistant_message([TextContent(text="two")]),
        ],
    )
    try:
        await _wait(runtime.session.prompt("hello"))
        await _wait(runtime.session.prompt("again"))
        entries = runtime.session.session_manager.get_entries()
        second_user = [entry for entry in entries if entry.type == "message" and entry.message.role == "user"][1]

        result = await _wait(runtime.fork(second_user.id, position="before"))

        assert result["cancelled"] is False
        assert result["selected_text"] == "again"
        assert _user_texts(runtime.session) == ["hello"]
    finally:
        await _wait(runtime.dispose())


async def test_fork_before_the_first_user_message_starts_a_child_session(tmp_path: Path) -> None:
    """With no parent entry to branch from, the fork becomes a fresh session
    whose header points at the outgoing session file."""
    runtime, _factory = await _make_runtime(
        tmp_path, persisted=True, responses=[_make_assistant_message([TextContent(text="one")])]
    )
    try:
        await _wait(runtime.session.prompt("hello"))
        previous_file = runtime.session.session_file
        assert previous_file is not None
        first_user = next(
            entry
            for entry in runtime.session.session_manager.get_entries()
            if entry.type == "message" and entry.message.role == "user"
        )
        assert first_user.parent_id is None

        result = await _wait(runtime.fork(first_user.id, position="before"))

        assert result["selected_text"] == "hello"
        assert runtime.session.messages == []
        header = runtime.session.session_manager.get_header()
        assert header is not None
        assert header.parent_session == previous_file
    finally:
        await _wait(runtime.dispose())


async def test_fork_at_duplicates_the_active_branch_into_a_new_persisted_file(tmp_path: Path) -> None:
    runtime, _factory = await _make_runtime(
        tmp_path, persisted=True, responses=[_make_assistant_message([TextContent(text="one")])]
    )
    try:
        await _wait(runtime.session.prompt("hello"))
        previous_file = runtime.session.session_file
        leaf_id = runtime.session.session_manager.get_leaf_id()
        assert previous_file is not None and leaf_id is not None

        result = await _wait(runtime.fork(leaf_id, position="at"))

        assert result == {"cancelled": False, "selected_text": None}
        forked_file = runtime.session.session_file
        assert forked_file is not None
        assert forked_file != previous_file
        assert Path(forked_file).exists()
        assert _user_texts(runtime.session) == ["hello"]
    finally:
        await _wait(runtime.dispose())


async def test_fork_reports_why_an_unflushed_persisted_session_cannot_be_forked(tmp_path: Path) -> None:
    """Port of TS's "reports why an unflushed session cannot be forked": a session
    file only exists once an assistant response has been written."""
    runtime, _factory = await _make_runtime(tmp_path, persisted=True)
    try:
        session_manager = runtime.session.session_manager
        entry_id = session_manager.append_message(UserMessage(content=[TextContent(text="hello")], timestamp=now_ms()))
        assert session_manager.get_session_file() is not None
        assert not Path(session_manager.get_session_file() or "").exists()

        with pytest.raises(ValueError, match="has not been saved yet"):
            await _wait(runtime.fork(entry_id, position="at"))
    finally:
        await _wait(runtime.dispose())


async def test_import_from_jsonl_rejects_a_missing_file(tmp_path: Path) -> None:
    runtime, _factory = await _make_runtime(tmp_path, persisted=True)
    try:
        with pytest.raises(SessionImportFileNotFoundError) as excinfo:
            await _wait(runtime.import_from_jsonl(str(tmp_path / "nope.jsonl")))
        assert excinfo.value.file_path == str(tmp_path / "nope.jsonl")
    finally:
        await _wait(runtime.dispose())


async def test_import_from_jsonl_copies_the_file_into_the_session_dir_and_switches(tmp_path: Path) -> None:
    exported_dir = tmp_path / "exported"
    exported_dir.mkdir()
    model_runtime = await _make_model_runtime(tmp_path)
    source_manager = SessionManager.create(str(tmp_path), session_dir=str(exported_dir))
    source_session = _build_session_for(
        tmp_path, model_runtime, source_manager, [_make_assistant_message([TextContent(text="one")])]
    )
    await _wait(source_session.prompt("imported prompt"))
    source_file = source_session.session_file
    assert source_file is not None
    source_session.dispose()

    runtime, factory = await _make_runtime(tmp_path, persisted=True)
    try:
        result = await _wait(runtime.import_from_jsonl(source_file))

        assert result == {"cancelled": False}
        destination = tmp_path / "sessions" / Path(source_file).name
        assert destination.exists()
        assert runtime.session.session_file == str(destination)
        assert _user_texts(runtime.session) == ["imported prompt"]
        assert factory.calls[-1]["session_manager"].get_cwd() == str(tmp_path)
    finally:
        await _wait(runtime.dispose())


async def test_import_from_jsonl_skips_the_copy_when_the_file_is_already_in_the_session_dir(tmp_path: Path) -> None:
    model_runtime = await _make_model_runtime(tmp_path)
    session_dir = tmp_path / "sessions"
    source_manager = SessionManager.create(str(tmp_path), session_dir=str(session_dir))
    source_session = _build_session_for(
        tmp_path, model_runtime, source_manager, [_make_assistant_message([TextContent(text="one")])]
    )
    await _wait(source_session.prompt("already here"))
    source_file = source_session.session_file
    assert source_file is not None
    source_session.dispose()
    before_mtime = Path(source_file).stat().st_mtime_ns

    runtime, _factory = await _make_runtime(tmp_path, persisted=True)
    try:
        result = await _wait(runtime.import_from_jsonl(source_file))

        assert result == {"cancelled": False}
        assert runtime.session.session_file == source_file
        assert Path(source_file).stat().st_mtime_ns == before_mtime
        assert _user_texts(runtime.session) == ["already here"]
    finally:
        await _wait(runtime.dispose())


async def test_import_from_jsonl_honours_a_cwd_override(tmp_path: Path) -> None:
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    exported_dir = tmp_path / "exported"
    exported_dir.mkdir()
    model_runtime = await _make_model_runtime(tmp_path)
    source_manager = SessionManager.create(str(tmp_path), session_dir=str(exported_dir))
    source_session = _build_session_for(
        tmp_path, model_runtime, source_manager, [_make_assistant_message([TextContent(text="one")])]
    )
    await _wait(source_session.prompt("hello"))
    source_file = source_session.session_file
    assert source_file is not None
    source_session.dispose()

    runtime, factory = await _make_runtime(tmp_path, persisted=True)
    try:
        await _wait(runtime.import_from_jsonl(source_file, cwd_override=str(other_cwd)))

        assert runtime.cwd == str(other_cwd)
        assert factory.calls[-1]["cwd"] == str(other_cwd)
    finally:
        await _wait(runtime.dispose())


async def test_runtime_dispose_disposes_the_current_session(tmp_path: Path) -> None:
    runtime, _factory = await _make_runtime(tmp_path)
    session = runtime.session
    session.subscribe(lambda event: None)
    assert session._event_listeners != []

    await _wait(runtime.dispose())

    assert session._event_listeners == []


# ---------------------------------------------------------------------------
# PiAgentSessionRuntime: resumed transcripts, phases, steering and errors
# ---------------------------------------------------------------------------


async def test_transcript_is_seeded_from_a_resumed_session(tmp_path: Path) -> None:
    """A runtime built over a session that already has messages rebuilds the wire
    transcript from them, instead of starting empty."""
    model_runtime = await _make_model_runtime(tmp_path)
    session_dir = tmp_path / "sessions"
    original_manager = SessionManager.create(str(tmp_path), session_dir=str(session_dir))
    original = _build_session_for(
        tmp_path,
        model_runtime,
        original_manager,
        [
            _make_assistant_message(
                [ToolCall(id="call-1", name="echo", arguments={"text": "hi"})], stop_reason="toolUse"
            ),
            _make_assistant_message([TextContent(text="done")]),
        ],
    )
    await _wait(original.prompt("please echo"))
    session_file = original.session_file
    assert session_file is not None
    original.dispose()

    resumed = _build_session_for(tmp_path, model_runtime, SessionManager.open(session_file), [])
    runtime = PiAgentSessionRuntime(resumed)
    try:
        transcript = runtime.snapshot()["transcript"]
        assert [item["role"] for item in transcript] == ["user", "assistant", "tool", "assistant"]
        tool_item = transcript[2]
        assert tool_item["toolName"] == "echo"
        assert tool_item["input"] == {"text": "hi"}
        assert tool_item["content"] == [{"type": "text", "text": "echo:hi"}]
        # Ids are allocated in transcript order.
        assert [item["id"] for item in transcript] == ["item-1", "item-2", "item-3", "item-4"]
    finally:
        await _wait(runtime.dispose())


async def test_seeded_tool_result_without_its_call_synthesizes_one(tmp_path: Path) -> None:
    """A persisted tool result whose assistant tool call is no longer in context
    still produces a transcript item, with empty arguments."""
    model_runtime = await _make_model_runtime(tmp_path)
    session_manager = SessionManager.in_memory(str(tmp_path))
    session_manager.append_message(UserMessage(content="orphan check", timestamp=now_ms()))
    session_manager.append_message(
        ToolResultMessage(
            tool_call_id="call-orphan",
            tool_name="echo",
            content=[TextContent(text="echo:orphan")],
            usage=Usage(),
            is_error=False,
            timestamp=now_ms(),
        )
    )
    session = _build_session_for(tmp_path, model_runtime, session_manager, [])
    runtime = PiAgentSessionRuntime(session)
    try:
        transcript = runtime.snapshot()["transcript"]
        assert [item["role"] for item in transcript] == ["user", "tool"]
        assert transcript[1]["toolCallId"] == "call-orphan"
        assert transcript[1]["input"] == {}
    finally:
        await _wait(runtime.dispose())


def _gated_tool(release: asyncio.Event, started: asyncio.Event) -> AgentTool:
    async def execute(tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None) -> AgentToolResult:
        started.set()
        await release.wait()
        return AgentToolResult(content=[TextContent(text="done")])

    return AgentTool(
        name="echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        label="Echo",
        execute=execute,
    )


async def _busy_runtime(tmp_path: Path) -> tuple[PiAgentSessionRuntime, asyncio.Event, Any]:
    """A runtime with an in-flight prompt, blocked inside a tool until the returned event is set."""
    release = asyncio.Event()
    started = asyncio.Event()
    responses = [
        _make_assistant_message([ToolCall(id="call-1", name="echo", arguments={"text": "hi"})], stop_reason="toolUse"),
        _make_assistant_message([TextContent(text="ok")]),
    ]
    session = await _build_real_session(tmp_path, responses, tool=_gated_tool(release, started))
    runtime = PiAgentSessionRuntime(session)
    prompt_task = asyncio.ensure_future(runtime.prompt(_prompt_input("first")))
    await _wait(started.wait())
    return runtime, release, prompt_task


async def test_steer_requires_an_active_prompt(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [])
    runtime = PiAgentSessionRuntime(session)
    try:
        with pytest.raises(PiServerError, match="no active prompt"):
            await _wait(runtime.steer(_steer_input("hurry up")))
    finally:
        await _wait(runtime.dispose())


async def test_steer_queues_a_message_reported_by_the_snapshot(tmp_path: Path) -> None:
    runtime, release, prompt_task = await _busy_runtime(tmp_path)
    try:
        await _wait(runtime.steer(_steer_input("hurry up")))

        snapshot = runtime.snapshot()
        assert snapshot["queuedSteerCount"] == 1
        assert snapshot["queuedSteer"] == [
            {
                "id": "queued-steer-0",
                "role": "user",
                "content": [{"type": "text", "text": "hurry up"}],
                "timestamp": snapshot["updatedAt"],
            }
        ]
        assert snapshot["phase"] == "turn"
    finally:
        release.set()
        await _wait(prompt_task)
        await _wait(runtime.dispose())


async def test_set_model_and_set_thinking_are_rejected_while_busy(tmp_path: Path) -> None:
    runtime, release, prompt_task = await _busy_runtime(tmp_path)
    try:
        with pytest.raises(PiServerError, match="busy"):
            await _wait(runtime.set_model({"provider": "test", "id": "test-model"}))
        with pytest.raises(PiServerError, match="busy"):
            await _wait(runtime.set_thinking("medium"))
    finally:
        release.set()
        await _wait(prompt_task)
        await _wait(runtime.dispose())


async def test_set_model_switches_the_session_model(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [], extra_provider_models=[_SECOND_MODEL])
    runtime = PiAgentSessionRuntime(session)
    try:
        await _wait(runtime.set_model({"provider": "test", "id": "second-model"}))

        assert runtime.snapshot()["model"] == {"provider": "test", "id": "second-model"}
        assert session.model is not None and session.model.id == "second-model"
    finally:
        await _wait(runtime.dispose())


async def test_set_thinking_updates_the_session_thinking_level(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [], extra_provider_models=[_SECOND_MODEL])
    runtime = PiAgentSessionRuntime(session)
    events: list[Any] = []
    runtime.subscribe(events.append)
    try:
        await _wait(runtime.set_model({"provider": "test", "id": "second-model"}))
        await _wait(runtime.set_thinking("medium"))

        assert runtime.snapshot()["thinkingLevel"] == "medium"
        # `thinking_level_changed` re-broadcasts the snapshot to subscribers.
        assert any(event.type == "snapshot" for event in events)
    finally:
        await _wait(runtime.dispose())


async def test_phase_reports_compaction_while_a_compaction_runs(tmp_path: Path) -> None:
    model_runtime = await _make_model_runtime(tmp_path)
    session_manager = SessionManager.in_memory(str(tmp_path))
    session = _build_session_for(
        tmp_path,
        model_runtime,
        session_manager,
        [
            _make_assistant_message([TextContent(text="SUMMARY OF HISTORY")]),
            _make_assistant_message([TextContent(text="TURN PREFIX")]),
        ],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 100}},
    )
    session_manager.append_message(UserMessage(content="first question " * 20, timestamp=1))
    session_manager.append_message(_make_assistant_message([TextContent(text="first answer " * 20)]))
    session_manager.append_message(UserMessage(content="second question " * 20, timestamp=3))
    session_manager.append_message(_make_assistant_message([TextContent(text="second answer " * 20)]))
    session.agent.state.messages = session_manager.build_session_context().messages

    runtime = PiAgentSessionRuntime(session)
    phases: list[str] = []
    session.subscribe(lambda event: phases.append(runtime.get_phase()) if event.type == "compaction_start" else None)
    try:
        await _wait(session.compact(), timeout=10.0)

        assert phases == ["compaction"]
        assert runtime.get_phase() == "idle"
    finally:
        await _wait(runtime.dispose())


async def test_phase_reports_retry_while_a_provider_error_is_retried(tmp_path: Path) -> None:
    model_runtime = await _make_model_runtime(tmp_path)
    session = _build_session_for(
        tmp_path,
        model_runtime,
        SessionManager.in_memory(str(tmp_path)),
        [
            _make_assistant_message([], stop_reason="error", error_message="503 service unavailable"),
            _make_assistant_message([TextContent(text="recovered")]),
        ],
        settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 0}},
    )
    runtime = PiAgentSessionRuntime(session)
    phases: list[str] = []
    session.subscribe(lambda event: phases.append(runtime.get_phase()) if event.type == "auto_retry_start" else None)
    try:
        await _wait(runtime.prompt(_prompt_input("hi")), timeout=10.0)

        assert phases == ["retry"]
        assert runtime.get_phase() == "idle"
    finally:
        await _wait(runtime.dispose())


async def test_protocol_errors_raised_while_handling_a_session_event_reach_listeners(tmp_path: Path) -> None:
    """`_on_session_event` converts a `PiServerError` raised by transcript
    building into an `error` runtime event rather than letting it escape into
    the session's event dispatch."""
    session = await _build_real_session(tmp_path, [_make_assistant_message([TextContent(text="hi")])])
    runtime = PiAgentSessionRuntime(session)
    errors: list[Any] = []
    runtime.subscribe(lambda event: errors.append(event) if event.type == "error" else None)

    def explode(_message: Any) -> None:
        raise PiServerError("internal", "transcript build failed")

    runtime._on_message_start = explode  # type: ignore[method-assign]
    try:
        await _wait(runtime.prompt(_prompt_input("hello")))

        # Both the user and the assistant `message_start` dispatch through the
        # same guard, so each reports one error event.
        assert [event.error.code for event in errors] == ["internal", "internal"]
        assert errors[0].error.message == "transcript build failed"
    finally:
        await _wait(runtime.dispose())


async def test_message_update_without_a_streaming_assistant_is_ignored(tmp_path: Path) -> None:
    session = await _build_real_session(tmp_path, [])
    runtime = PiAgentSessionRuntime(session)
    progress: list[Any] = []
    runtime.subscribe(lambda event: progress.append(event) if event.type == "progress" else None)
    try:
        message = _make_assistant_message([TextContent(text="orphan")])
        runtime._on_session_event(
            MessageUpdateEvent(
                message=message,
                assistant_message_event=TextDeltaEvent(content_index=0, delta="orphan", partial=message),
            )
        )

        assert progress == []
    finally:
        await _wait(runtime.dispose())


async def test_tool_result_without_a_running_item_is_appended_to_the_transcript(tmp_path: Path) -> None:
    """A tool result that never went through `tool_execution_start` (so it has
    neither a running item nor a recorded call) still lands in the transcript."""
    session = await _build_real_session(tmp_path, [])
    runtime = PiAgentSessionRuntime(session)
    progress: list[Any] = []
    runtime.subscribe(lambda event: progress.append(event.progress) if event.type == "progress" else None)
    try:
        runtime._on_session_event(
            MessageEndEvent(
                message=ToolResultMessage(
                    tool_call_id="call-standalone",
                    tool_name="echo",
                    content=[TextContent(text="echo:standalone")],
                    usage=Usage(),
                    is_error=False,
                    timestamp=now_ms(),
                )
            )
        )

        transcript = runtime.snapshot()["transcript"]
        assert [item["role"] for item in transcript] == ["tool"]
        assert transcript[0]["input"] == {}
        assert transcript[0]["status"] == "complete"
        assert [event["type"] for event in progress] == ["item_finished"]
    finally:
        await _wait(runtime.dispose())


def _steer_input(text: str) -> Any:
    from pi_server.types import SteerInput

    return SteerInput(text=text)


# ---------------------------------------------------------------------------
# PiAgentSessionRuntimeService: model refs and reopening persisted sessions
# ---------------------------------------------------------------------------


async def test_service_create_session_rejects_an_unknown_model_ref(tmp_path: Path) -> None:
    service = await _make_service(tmp_path)
    with pytest.raises(PiServerError, match="Unknown model"):
        await _wait(
            service.create_session(
                CreateSessionOptions(id="bad-model", cwd=str(tmp_path), model={"provider": "nope", "id": "nope"})
            )
        )


async def test_service_create_session_uses_the_requested_model(tmp_path: Path) -> None:
    service = await _make_service(tmp_path)
    runtime = await _wait(
        service.create_session(
            CreateSessionOptions(
                id="with-model", cwd=str(tmp_path), model={"provider": "test", "id": "test-model"}, thinking_level="off"
            )
        )
    )
    try:
        assert runtime.snapshot()["model"] == {"provider": "test", "id": "test-model"}
    finally:
        await _wait(runtime.dispose())


async def test_service_open_session_resumes_a_persisted_session(tmp_path: Path) -> None:
    """`open_session` finds the session on disk, rebuilds a runtime over it, and
    releases the lock again when that runtime is disposed."""
    session_dir = tmp_path / "sessions"
    model_runtime = await _make_model_runtime(tmp_path)
    session_manager = SessionManager.create(
        str(tmp_path), session_dir=str(session_dir), options=NewSessionOptions(id="reopen-me")
    )
    session_manager.append_session_info("saved session")
    session = _build_session_for(
        tmp_path, model_runtime, session_manager, [_make_assistant_message([TextContent(text="hi")])]
    )
    await _wait(session.prompt("first prompt"))
    session.dispose()

    service = PiAgentSessionRuntimeService(
        agent_dir=str(tmp_path / "agent"),
        default_cwd=str(tmp_path),
        model_runtime=model_runtime,
        session_dir=str(session_dir),
    )
    runtime = await _wait(service.open_session("reopen-me"))
    try:
        snapshot = runtime.snapshot()
        assert snapshot["id"] == "reopen-me"
        assert snapshot["name"] == "saved session"
        assert snapshot["cwd"] == str(tmp_path)
        assert [item["role"] for item in snapshot["transcript"]] == ["user", "assistant"]
        with pytest.raises(PiServerError, match="is locked"):
            await _wait(service.open_session("reopen-me"))
    finally:
        await _wait(runtime.dispose())

    # Disposing released the lock, so the session can be opened again.
    reopened = await _wait(service.open_session("reopen-me"))
    await _wait(reopened.dispose())


async def test_fork_before_the_first_message_of_an_in_memory_session_starts_over(tmp_path: Path) -> None:
    """In-memory sessions branch the session manager they already own; with no
    parent entry the fork resets it to a fresh session instead. The forked user
    message carries plain-string content, which `_extract_user_message_text`
    returns unchanged."""
    runtime, _factory = await _make_runtime(tmp_path)
    try:
        session_manager = runtime.session.session_manager
        entry_id = session_manager.append_message(UserMessage(content="plain string prompt", timestamp=now_ms()))
        assert session_manager.get_entry(entry_id) is not None

        result = await _wait(runtime.fork(entry_id, position="before"))

        assert result == {"cancelled": False, "selected_text": "plain string prompt"}
        assert runtime.session.session_manager.get_entries() == []
        assert runtime.session.messages == []
    finally:
        await _wait(runtime.dispose())


async def test_switch_session_updates_the_runtime_cwd_on_cross_cwd_replacement(tmp_path: Path) -> None:
    """Port of TS's "updates the runtime session cwd on cross-cwd session replacement".

    Switching to a session recorded under a different working directory must
    move the runtime there too, not just the session manager.
    """
    first_dir = tmp_path / "cwd-a"
    second_dir = tmp_path / "cwd-b"
    first_dir.mkdir()
    second_dir.mkdir()
    session_dir = tmp_path / "sessions"

    model_runtime = await _make_model_runtime(tmp_path)

    # A session belonging to `second_dir`, persisted to disk.
    other_manager = SessionManager.create(str(second_dir), session_dir=str(session_dir))
    other_session = _build_session_for(
        tmp_path,
        model_runtime,
        other_manager,
        [_make_assistant_message([TextContent(text="other")])],
        cwd=str(second_dir),
    )
    await _wait(other_session.prompt("other"))
    other_session_file = other_session.session_file
    assert other_session_file is not None
    other_session.dispose()

    factory = _RecordingRuntimeFactory(tmp_path, model_runtime)
    runtime = await _wait(
        create_agent_session_runtime(
            factory,
            cwd=str(first_dir),
            agent_dir=str(tmp_path / "agent"),
            session_manager=SessionManager.create(str(first_dir), session_dir=str(session_dir)),
        )
    )
    try:
        assert runtime.cwd == str(first_dir)

        await _wait(runtime.switch_session(other_session_file))

        assert Path(runtime.session.session_manager.get_cwd()).resolve() == second_dir.resolve()
        assert Path(runtime.cwd).resolve() == second_dir.resolve()
        assert factory.calls[-1]["cwd"] == str(second_dir)
    finally:
        await _wait(runtime.dispose())


async def test_switch_session_restores_model_and_thinking_state_from_the_destination(tmp_path: Path) -> None:
    """Port of TS's "restores model and thinking state from the destination session".

    TS drives this through the real `createAgentSessionFromServices`, so the
    factory here is the real `create_agent_session` rather than
    `_RecordingRuntimeFactory` -- a fake factory that always builds
    `TEST_MODEL` could not observe the restore at all.
    """
    session_dir = tmp_path / "sessions"
    agent_dir = tmp_path / "agent"
    cwd = str(tmp_path)

    model_runtime = await _wait(
        ModelRuntime.create(agent_dir=agent_dir, providers=[_fake_provider(extra_models=[_SECOND_MODEL])])
    )
    await _wait(model_runtime.login("test", "fake-key"))
    second_model = model_runtime.get_model("test", _SECOND_MODEL.id)
    assert second_model is not None

    async def factory(
        *, cwd: str, agent_dir: str, session_manager: SessionManager, **_ignored: Any
    ) -> CreateAgentSessionResult:
        return await create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                agent_dir=agent_dir,
                model_runtime=model_runtime,
                session_manager=session_manager,
                no_tools="all",
            )
        )

    # Session B: switched to `second-model` with thinking off, then persisted.
    other_manager = SessionManager.create(cwd, session_dir=str(session_dir))
    other = await _wait(
        create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                agent_dir=str(agent_dir),
                model_runtime=model_runtime,
                session_manager=other_manager,
                no_tools="all",
            )
        )
    )
    await _wait(other.session.set_model(second_model))
    other.session.set_thinking_level("off")
    other_manager.append_message(UserMessage(content="hello", timestamp=now_ms()))
    other_manager.append_message(
        dataclasses.replace(_make_assistant_message([TextContent(text="hi")]), model=_SECOND_MODEL.id)
    )
    other_session_file = other.session.session_file
    assert other_session_file is not None
    other.session.dispose()
    assert Path(other_session_file).exists()

    runtime = await _wait(
        create_agent_session_runtime(
            factory,
            cwd=cwd,
            agent_dir=str(agent_dir),
            session_manager=SessionManager.create(cwd, session_dir=str(session_dir)),
        )
    )
    try:
        await _wait(runtime.switch_session(other_session_file))

        assert runtime.session.model is not None
        assert runtime.session.model.id == _SECOND_MODEL.id
        assert runtime.session.thinking_level == "off"
    finally:
        await _wait(runtime.dispose())
