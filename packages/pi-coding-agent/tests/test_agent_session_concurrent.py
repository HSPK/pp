"""Python port of `packages/coding-agent/test/agent-session-concurrent.test.ts`.

Tests for the `AgentSession` concurrent prompt guard.

Two deliberate substitutions, both forced by this repo's "no real sleeps" rule
(the TypeScript uses `setTimeout(10)`, `setTimeout(25)`, `setTimeout(40)` and
`setTimeout(100)` throughout):

- "wait a tick for isStreaming to be set" becomes `_wait_until(...)`, which
  yields to the event loop with zero-delay sleeps until the condition holds.
- The "block until aborted" stream function awaits `signal.wait()` instead of
  re-checking `abortSignal.aborted` every 5 ms.

Neither changes what is asserted.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pi_agent.agent import Agent, MutableAgentState
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.auth.types import Credential
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    StartEvent,
    TextContent,
    ToolCall,
    Usage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_coding_agent.core.agent_session import AgentSession, QueueUpdateEvent
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.extensions.types import InputEventResult
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

sys.path.insert(0, str(Path(__file__).resolve().parent / "suite"))

from harness import (
    _make_extension_actions,
    _SessionRef,
    build_extensions,
    create_test_resource_loader,
)

TIMEOUT = 5.0


def create_assistant_message(text: str) -> AssistantMessage:
    """Port of the TypeScript `createAssistantMessage`."""
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="mock",
        content=[TextContent(text=text)],
        usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost()),
        stop_reason="stop",
        timestamp=now_ms(),
    )


def _tool_use_message(content: list[Any], stop_reason: str) -> AssistantMessage:
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="mock",
        content=content,
        usage=Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=Cost()),
        stop_reason=stop_reason,
        timestamp=now_ms(),
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = TIMEOUT) -> None:
    """Zero-delay replacement for the TypeScript `setTimeout(resolve, 10)` waits."""

    async def _loop() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_loop(), timeout=timeout)


def _signal_of(options: Any, kwargs: dict[str, Any]) -> Any:
    return getattr(options, "signal", None) if options is not None else kwargs.get("signal")


def _spawn_stream(body: Callable[[], Any]) -> None:
    """TS uses `queueMicrotask`; the closest asyncio equivalent is a task."""
    asyncio.get_running_loop().create_task(body())


def _blocking_stream():
    """Stream function that emits `start`, then blocks until aborted."""

    def stream_fn(_model: Any, _context: Any = None, options: Any = None, **kwargs: Any):
        signal = _signal_of(options, kwargs)
        stream = AssistantMessageEventStream()

        async def run() -> None:
            stream.push(StartEvent(partial=create_assistant_message("")))
            if signal is not None:
                await signal.wait()
            aborted = create_assistant_message("Aborted")
            stream.push(ErrorEvent(reason="aborted", error=aborted))

        _spawn_stream(run)
        return stream

    return stream_fn


async def _make_model_runtime(temp_dir: Path) -> ModelRuntime:
    auth_storage = AuthStorage.create(str(temp_dir / "auth.json"))
    model_runtime = await ModelRuntime.create(
        agent_dir=str(temp_dir),
        credentials=auth_storage,
        models_path=str(temp_dir / "models.json"),
    )
    # TS: `authStorage.modify("anthropic", ...)`. This port has no `modify()`.
    await auth_storage.set("anthropic", Credential(type="api_key", key="test-key"))
    return model_runtime


async def _create_session(
    temp_dir: Path,
    stream_fn: Any,
    *,
    tools: list[AgentTool] | None = None,
    base_tools_override: dict[str, AgentTool] | None = None,
    extension_factories: list[Any] | None = None,
    session_ref: _SessionRef | None = None,
) -> tuple[AgentSession, SessionManager]:
    """Port of the TypeScript `createSession()` helper (inlined per-test there)."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    model_runtime = await _make_model_runtime(temp_dir)
    model = model_runtime.get_model("anthropic", "claude-sonnet-4-5")
    assert model is not None

    agent = Agent(
        stream_fn,
        initial_state=MutableAgentState(model=model, system_prompt="Test"),
        get_api_key=lambda _provider: "test-key",
    )
    agent.state.tools = list(tools or [])

    session_manager = SessionManager.in_memory()
    settings_manager = SettingsManager.create(str(temp_dir), str(temp_dir))

    ref = session_ref or _SessionRef()
    extensions = []
    if extension_factories:
        extensions = await build_extensions(extension_factories, str(temp_dir), _make_extension_actions(ref))

    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=settings_manager,
        cwd=str(temp_dir),
        model_runtime=model_runtime,
        resource_loader=create_test_resource_loader(temp_dir),
        base_tools_override=base_tools_override,
        extensions=extensions,
    )
    ref.session = session
    return session, session_manager


async def _drain(prompt_task: asyncio.Future[Any]) -> None:
    """TS: `await firstPrompt.catch(() => {})`."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(prompt_task, timeout=TIMEOUT)


async def test_should_throw_when_prompt_called_while_streaming(tmp_path: Path) -> None:
    session, _ = await _create_session(tmp_path / "s", _blocking_stream())
    try:
        first_prompt = asyncio.ensure_future(session.prompt("First message"))
        await _wait_until(lambda: session.is_streaming)

        assert session.is_streaming is True

        with pytest.raises(Exception) as excinfo:
            await session.prompt("Second message")
        # TS's message names the TypeScript parameter (`streamingBehavior`); this
        # port names the same parameter `streaming_behavior`, so the message
        # tracks the Python signature.
        assert (
            "Agent is already processing. Specify streaming_behavior ('steer' or 'followUp') to queue the message."
            in str(excinfo.value)
        )

        await session.abort()
        await _drain(first_prompt)
    finally:
        session.dispose()


async def test_should_allow_steer_while_streaming(tmp_path: Path) -> None:
    session, _ = await _create_session(tmp_path / "s", _blocking_stream())
    try:
        first_prompt = asyncio.ensure_future(session.prompt("First message"))
        await _wait_until(lambda: session.is_streaming)

        # TS: `expect(() => session.steer(...)).not.toThrow()`; `steer` is async here.
        await session.steer("Steering message")
        assert session.pending_message_count == 1

        await session.abort()
        await _drain(first_prompt)
    finally:
        session.dispose()


async def test_should_allow_follow_up_while_streaming(tmp_path: Path) -> None:
    session, _ = await _create_session(tmp_path / "s", _blocking_stream())
    try:
        first_prompt = asyncio.ensure_future(session.prompt("First message"))
        await _wait_until(lambda: session.is_streaming)

        await session.follow_up("Follow-up message")
        assert session.pending_message_count == 1

        await session.abort()
        await _drain(first_prompt)
    finally:
        session.dispose()


async def test_should_queue_extension_origin_steering_messages_while_streaming(tmp_path: Path) -> None:
    seen: dict[str, Any] = {"steering_message": False, "input_source": None, "pi": None}

    def stream_fn(_model: Any, context: Any = None, options: Any = None, **kwargs: Any):
        signal = _signal_of(options, kwargs)
        stream = AssistantMessageEventStream()

        user_texts: list[str] = []
        for message in getattr(context, "messages", []) or []:
            if getattr(message, "role", None) != "user":
                continue
            content = message.content
            if isinstance(content, str):
                user_texts.append(content)
            else:
                user_texts.append(
                    "\n".join(
                        part.text
                        for part in content
                        if isinstance(part, TextContent | ImageContent) and getattr(part, "type", None) == "text"
                    )
                )

        async def run() -> None:
            if "Steer from extension" in user_texts:
                seen["steering_message"] = True
                stream.push(StartEvent(partial=create_assistant_message("")))
                stream.push(DoneEvent(reason="stop", message=create_assistant_message("Steered")))
                return

            stream.push(StartEvent(partial=create_assistant_message("")))
            if signal is not None:
                await signal.wait()
            stream.push(ErrorEvent(reason="aborted", error=create_assistant_message("Aborted")))

        _spawn_stream(run)
        return stream

    def capture_api(pi: Any) -> None:
        # TS stashes `pi` on `globalThis.testExtensionApi`; a dict is the direct
        # equivalent without mutating module globals.
        seen["pi"] = pi

    def watch_input(pi: Any) -> None:
        async def on_input(event: Any, _ctx: Any = None) -> None:
            seen["input_source"] = event.source

        pi.on("input", on_input)

    session, _ = await _create_session(tmp_path / "s", stream_fn, extension_factories=[capture_api, watch_input])

    queue_events: list[QueueUpdateEvent] = []

    def on_event(event: Any) -> None:
        if isinstance(event, QueueUpdateEvent):
            queue_events.append(event)

    session.subscribe(on_event)

    try:
        first_prompt = asyncio.ensure_future(session.prompt("First message"))
        await _wait_until(lambda: session.is_streaming)
        assert session.is_streaming is True

        pi = seen["pi"]
        assert pi is not None

        pi.send_user_message("Steer from extension", {"deliverAs": "steer"})
        await _wait_until(lambda: session.pending_message_count == 1)

        assert session.pending_message_count == 1
        assert "Steer from extension" in session.get_steering_messages()
        assert seen["input_source"] == "extension"
        assert any("Steer from extension" in event.steering for event in queue_events) is True

        await session.abort()
        await _drain(first_prompt)

        assert seen["steering_message"] is True
    finally:
        session.dispose()


async def test_should_allow_prompt_after_previous_completes(tmp_path: Path) -> None:
    def stream_fn(_model: Any, _context: Any = None, options: Any = None, **kwargs: Any):
        stream = AssistantMessageEventStream()

        async def run() -> None:
            stream.push(StartEvent(partial=create_assistant_message("")))
            stream.push(DoneEvent(reason="stop", message=create_assistant_message("Done")))

        _spawn_stream(run)
        return stream

    session, _ = await _create_session(tmp_path / "s", stream_fn)
    try:
        await session.prompt("First message")

        assert session.is_streaming is False

        # TS: `await expect(session.prompt(...)).resolves.not.toThrow()`.
        await session.prompt("Second message")
    finally:
        session.dispose()


def _dummy_tool() -> AgentTool:
    async def execute(_tool_call_id: str, params: Any, *_args: Any, **_kwargs: Any) -> AgentToolResult:
        q = str(params["q"]) if isinstance(params, dict) and "q" in params else ""
        return AgentToolResult(content=[TextContent(text=f"result:{q}")], details={})

    return AgentTool(
        name="dummy",
        label="dummy",
        description="Dummy tool",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        execute=execute,
    )


def _tool_calling_stream(tool_calls: list[ToolCall], *, leading_text: str | None = None):
    """First turn emits `tool_calls`; once a toolResult is in context, a plain reply."""

    def stream_fn(_model: Any, context: Any = None, options: Any = None, **kwargs: Any):
        stream = AssistantMessageEventStream()
        messages = getattr(context, "messages", []) or []
        has_tool_result = any(getattr(m, "role", None) == "toolResult" for m in messages)

        async def run() -> None:
            if has_tool_result:
                message = _tool_use_message([TextContent(text="done")], "stop")
                stream.push(StartEvent(partial=_tool_use_message([], "stop")))
                stream.push(DoneEvent(reason="stop", message=message))
                return

            content: list[Any] = []
            if leading_text is not None:
                content.append(TextContent(text=leading_text))
            content.extend(tool_calls)
            message = _tool_use_message(content, "toolUse")
            stream.push(StartEvent(partial=_tool_use_message([], "toolUse")))
            stream.push(DoneEvent(reason="toolUse", message=message))

        _spawn_stream(run)
        return stream

    return stream_fn


class _StubExtensionRunner:
    """Stand-in for `session._extension_runner`, matching TS's object literal.

    Every method is declared with the real `ExtensionRunner` method's name and
    sync/async shape, so the stub cannot be easier to satisfy than the real
    runner.
    """

    def __init__(
        self,
        *,
        tool_call_handlers: bool = False,
        on_emit_tool_call: Callable[[Any], None] | None = None,
        on_emit_message_end: Callable[[Any], Any] | None = None,
    ) -> None:
        self._tool_call_handlers = tool_call_handlers
        self._on_emit_tool_call = on_emit_tool_call
        self._on_emit_message_end = on_emit_message_end

    def has_handlers(self, event_type: str) -> bool:
        return self._tool_call_handlers and event_type == "tool_call"

    async def emit(self, _event: Any) -> None:
        return None

    async def emit_message_end(self, event: Any) -> None:
        if self._on_emit_message_end is not None:
            await self._on_emit_message_end(event)
        return None

    async def emit_tool_call(self, event: Any) -> None:
        if self._on_emit_tool_call is not None:
            self._on_emit_tool_call(event)
        return None

    async def emit_tool_result(self, _event: Any) -> None:
        return None

    async def emit_input(
        self, _text: Any, _images: Any, _source: str, _streaming_behavior: str | None = None
    ) -> InputEventResult:
        return InputEventResult(action="continue")

    async def emit_before_agent_start(
        self, _prompt: Any, _images: Any, _system_prompt: Any, _system_prompt_options: Any
    ) -> None:
        return None

    async def emit_context(self, messages: list[Any]) -> list[Any]:
        return messages

    def emit_error(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def get_command(self, _name: str) -> None:
        return None

    def get_registered_commands(self) -> list[Any]:
        return []

    def get_all_registered_tools(self) -> list[Any]:
        return []

    def create_command_context(self) -> Any:
        return None

    def bind_core(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def unsubscribe_events(self) -> None:
        return None

    def invalidate(self, _message: str | None = None) -> None:
        return None


def _persisted_roles(session_manager: SessionManager) -> list[str]:
    return [entry.message.role for entry in session_manager.get_entries() if entry.type == "message"]


async def test_should_wait_for_queued_agent_events_before_emitting_tool_call(tmp_path: Path) -> None:
    tool = _dummy_tool()
    stream_fn = _tool_calling_stream(
        [
            ToolCall(id="toolu_1", name="dummy", arguments={"q": "x"}),
            ToolCall(id="toolu_2", name="dummy", arguments={"q": "y"}),
        ]
    )
    session, session_manager = await _create_session(
        tmp_path / "s", stream_fn, tools=[tool], base_tools_override={"dummy": tool}
    )
    try:
        snapshots: list[list[str]] = []
        session._extension_runner = _StubExtensionRunner(
            tool_call_handlers=True,
            on_emit_tool_call=lambda _event: snapshots.append(_persisted_roles(session_manager)),
        )

        await session.prompt("hi")
        await session.agent.wait_for_idle()

        assert snapshots == [
            ["user", "assistant"],
            ["user", "assistant"],
        ]
    finally:
        session.dispose()


async def test_should_persist_message_end_events_in_order_with_slow_extension_handlers(tmp_path: Path) -> None:
    tool = _dummy_tool()
    stream_fn = _tool_calling_stream(
        [ToolCall(id="toolu_1", name="dummy", arguments={"q": "x"})], leading_text="calling tool"
    )
    session, session_manager = await _create_session(
        tmp_path / "s", stream_fn, tools=[tool], base_tools_override={"dummy": tool}
    )
    try:

        async def slow_message_end(event: Any) -> None:
            message = getattr(event, "message", None)
            if getattr(event, "type", None) == "message_end" and getattr(message, "role", None) == "assistant":
                # TS sleeps 40 ms here to let the agent loop run ahead. Yielding
                # repeatedly gives the loop the same chance to interleave with
                # no wall-clock wait (see this module's docstring).
                for _ in range(100):
                    await asyncio.sleep(0)

        session._extension_runner = _StubExtensionRunner(on_emit_message_end=slow_message_end)

        await session.prompt("hi")
        await session.agent.wait_for_idle()
        for _ in range(200):
            await asyncio.sleep(0)

        assert _persisted_roles(session_manager) == ["user", "assistant", "toolResult", "assistant"]
    finally:
        session.dispose()
