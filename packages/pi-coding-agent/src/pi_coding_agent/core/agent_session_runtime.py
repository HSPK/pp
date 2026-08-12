"""Session-lifecycle wrapper and the real `pi_server` session-runtime adapter.

Python port of `packages/coding-agent/src/core/agent-session-runtime.ts`, plus
a concrete implementation of `pi_server`'s injectable `PiSessionRuntime` /
`PiServerService` boundary (see `pi_server/__init__.py`'s module docstring),
which the TypeScript monorepo never actually wired up either -- only its
in-memory `TestServerService` fake existed there.

**Session-lifecycle extension events are wired.** `switch_session` /
`new_session` / `import_from_jsonl` emit `session_before_switch` and `fork`
emits `session_before_fork` before doing any work; a handler returning
`cancel=True` aborts the replacement and the method returns
`{"cancelled": True}` without touching the current session.
`_teardown_current` emits `session_shutdown` (with the incoming session file
as `target_session_file`) after settling the outgoing turn and before
disposing it, and `dispose()` emits it with `reason="quit"`. Each replacement
passes a `session_start_event` to the runtime factory, which the new
`AgentSession` emits from `bind_extensions()`. A host registers
`set_rebind_session` to be handed the replacement session (print mode does),
and `set_before_session_invalidate` for synchronous teardown that must run
after `session_shutdown` but before the outgoing session is disposed.

**Dropped: stale-context invalidation.** TypeScript makes a captured
`ctx`/`pi` object raise "This extension ctx is stale after session replacement
or reload." once its session has been replaced. That guard exists to steer
extension authors towards `withSession`, which in turn belongs to the
extension UI host (a documented omission -- see the README); the runner here
has no "stale" state to enter.

**Dropped: `AgentSessionServices` / cwd-bound service recreation.** TS bundles
`ModelRuntime` / `SettingsManager` / `ResourceLoader` / extension state into a
per-cwd `AgentSessionServices` object that `agent-session-services.ts`
recreates on every session replacement (a new cwd can mean new project
settings, a new resource loader, etc). This port's `create_agent_session`
(see `sdk.py`) already accepts those pieces as plain constructor arguments,
so `CreateAgentSessionRuntimeFactory` closures over whatever "services" a
caller needs directly -- there is no separate services object to track, and
`AgentSessionRuntime.cwd` simply reads `session.session_manager.get_cwd()`.

**New: `PiAgentSessionRuntime` / `PiAgentSessionRuntimeService`.** These
implement `pi_server.types.PiSessionRuntime` / `PiServerService` by wrapping a
real `AgentSession`. `PiAgentSessionRuntime` subscribes to the session's
event stream and incrementally builds the wire transcript using
`pi_server.protocol.to_protocol_user_message` /
`to_protocol_assistant_message` / `to_protocol_tool_result_message`:

- `message_start` (role `user`) -> nothing yet; the full item is built and
  broadcast at `message_end` (mirrors `AgentSession._handle_agent_event`'s own
  persistence timing, and user messages never stream).
- `message_start` (role `assistant`) -> allocate a transcript id, store it as
  the session's single "currently streaming assistant" slot (only one
  assistant response streams at a time per session -- see `agent_loop.py`'s
  message-identity docs), and broadcast a `status: "streaming"` item via
  `item_started`.
- `message_update` (assistant) -> map `text_delta` / `thinking_delta` /
  `toolcall_delta` stream events to `assistant_delta` progress events keyed
  by that same id; the other `AssistantMessageEvent` variants (`*_start`,
  `*_end`, `done`, `error`) carry no incremental wire shape and are dropped.
  `toolcall_delta` is JSON-fragment text (`ToolCall.arguments` accumulated
  as a string before parsing) which has no non-string schema slot, so it is
  forwarded as fragment text rather than parsed.
- `message_end` (assistant) -> finalize via `to_protocol_assistant_message`
  and broadcast `item_finished`; clear the streaming slot; remember every
  `toolCall` content part's `ToolCall` by id so the matching tool result can
  be validated and re-attach its `arguments`/`name` (protocol's
  `ToolTranscriptOptions.call`).
- `tool_execution_start` -> allocate a transcript id for the call, look up
  the `ToolCall` recorded above (or synthesize one, if the raw agent event
  reordered relative to `message_end` -- doesn't happen in-process today but
  costs nothing to guard), and broadcast a `status: "running"` tool item via
  `item_started`. `tool_execution_update` is dropped: the tool schema has no
  incremental-output slot.
- `message_end` (role `toolResult`) -> finalize via
  `to_protocol_tool_result_message` using the tracked `ToolCall`, and
  broadcast `item_finished`.

`snapshot()`'s `queuedSteer` is synthesized from
`AgentSession.get_steering_messages()` (`list[str]`, no ids/timestamps
tracked) with per-call ids/timestamps -- a documented lossy simplification,
since nothing durable needs those queued-item identities to be stable across
snapshots.

`get_phase()` approximates `SESSION_PHASE_SCHEMA`'s `"compaction"` /
`"branch_summary"` distinction as always `"compaction"`: `AgentSession`
exposes a single `is_compacting` flag covering both auto-compaction, manual
compaction, and branch summarization, with no public way to tell them apart.
"""

from __future__ import annotations

import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any, Literal

from pi_agent.types import ToolCall
from pi_ai.types import Model, now_ms
from pi_server.errors import PiServerError
from pi_server.protocol import (
    AssistantTranscriptOptions,
    ToolTranscriptOptions,
    UserTranscriptOptions,
    to_protocol_assistant_message,
    to_protocol_json_value,
    to_protocol_model_metadata,
    to_protocol_tool_result_message,
    to_protocol_user_message,
)
from pi_server.types import (
    CreateSessionOptions,
    ErrorRuntimeEvent,
    PiSessionRuntimeEvent,
    ProgressRuntimeEvent,
    PromptInput,
    SnapshotRuntimeEvent,
    SteerInput,
)

from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.extensions.types import (
    SessionBeforeForkEvent,
    SessionBeforeForkResult,
    SessionBeforeSwitchEvent,
    SessionBeforeSwitchResult,
    SessionShutdownEvent,
    SessionStartEvent,
)
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, CreateAgentSessionResult, create_agent_session
from pi_coding_agent.core.session_cwd import assert_session_cwd_exists
from pi_coding_agent.core.session_manager import NewSessionOptions, SessionManager, get_default_session_dir
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.utils.paths import resolve_path

# ============================================================================
# `AgentSessionRuntime`: session-lifecycle wrapper (ported, extension-stripped)
# ============================================================================

CreateAgentSessionRuntimeFactory = Callable[..., Awaitable[CreateAgentSessionResult]]
"""`(*, cwd, agent_dir, session_manager, **_ignored) -> CreateAgentSessionResult`.

Closes over whatever fixed inputs (model runtime, settings manager) a caller
needs; extra keyword arguments (e.g. a dropped `session_start_event`) are
accepted and ignored so callers can pass TS-shaped kwargs without failing.
"""


class SessionImportFileNotFoundError(Exception):
    """Raised by `import_from_jsonl` when the input path does not exist."""

    def __init__(self, file_path: str) -> None:
        super().__init__(f"File not found: {file_path}")
        self.file_path = file_path


def _extract_user_message_text(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if getattr(part, "type", None) == "text")


class AgentSessionRuntime:
    """Owns the current `AgentSession`. Replacement methods tear down, then create and apply the next one.

    If creation fails, the error propagates to the caller (matching TS); the
    caller is responsible for user-facing error handling.
    """

    def __init__(
        self,
        session: AgentSession,
        agent_dir: str,
        create_runtime: CreateAgentSessionRuntimeFactory,
        model_fallback_message: str | None = None,
    ) -> None:
        self._session = session
        self._agent_dir = agent_dir
        self._create_runtime = create_runtime
        self._model_fallback_message = model_fallback_message
        self._rebind_session: Callable[[AgentSession], Awaitable[None]] | None = None
        self._before_session_invalidate: Callable[[], None] | None = None

    @property
    def session(self) -> AgentSession:
        return self._session

    def set_rebind_session(self, rebind_session: Callable[[AgentSession], Awaitable[None]] | None) -> None:
        """Register the host callback that re-attaches to a replacement session.

        Port of `setRebindSession`. A mode registers this so that when an
        extension replaces the session (`ctx.new_session()`, `ctx.fork()`,
        `ctx.switch_session()`) the mode calls `bind_extensions()` on the new
        session and moves its event subscription over. Without it the mode
        keeps listening to the disposed session.
        """
        self._rebind_session = rebind_session

    def set_before_session_invalidate(self, before_session_invalidate: Callable[[], None] | None) -> None:
        """Register a synchronous callback run after `session_shutdown` and before disposal.

        Port of `setBeforeSessionInvalidate`. It is deliberately synchronous:
        host-owned teardown here must not yield to the event loop, or the old
        session would already be disposed when it ran.
        """
        self._before_session_invalidate = before_session_invalidate

    @property
    def agent_dir(self) -> str:
        return self._agent_dir

    @property
    def cwd(self) -> str:
        return self._session.session_manager.get_cwd()

    @property
    def model_fallback_message(self) -> str | None:
        return self._model_fallback_message

    async def _emit_before_switch(
        self, reason: Literal["new", "resume"], target_session_file: str | None = None
    ) -> bool:
        """Emit `session_before_switch`; returns `True` when an extension cancelled."""
        runner = self._session.extension_runner
        if not runner.has_handlers("session_before_switch"):
            return False
        result = await runner.emit(SessionBeforeSwitchEvent(reason=reason, target_session_file=target_session_file))
        return isinstance(result, SessionBeforeSwitchResult) and result.cancel is True

    async def _emit_before_fork(self, entry_id: str, position: Literal["before", "at"]) -> bool:
        """Emit `session_before_fork`; returns `True` when an extension cancelled."""
        runner = self._session.extension_runner
        if not runner.has_handlers("session_before_fork"):
            return False
        result = await runner.emit(SessionBeforeForkEvent(entry_id=entry_id, position=position))
        return isinstance(result, SessionBeforeForkResult) and result.cancel is True

    async def _teardown_current(
        self,
        reason: Literal["quit", "reload", "new", "resume", "fork"],
        target_session_file: str | None = None,
    ) -> None:
        # Settle any active response first so the aborted turn (including tool
        # results) is persisted to the outgoing session before it is replaced.
        await self._session.abort()
        runner = self._session.extension_runner
        if runner.has_handlers("session_shutdown"):
            await runner.emit(SessionShutdownEvent(reason=reason, target_session_file=target_session_file))
        if self._before_session_invalidate is not None:
            self._before_session_invalidate()
        self._session.dispose()

    def _apply(self, result: CreateAgentSessionResult) -> None:
        self._session = result.session
        self._model_fallback_message = result.model_fallback_message

    async def _finish_session_replacement(self) -> None:
        """Port of `finishSessionReplacement`: hand the new session to the host."""
        if self._rebind_session is not None:
            await self._rebind_session(self._session)

    async def switch_session(self, session_path: str, cwd_override: str | None = None) -> dict[str, bool]:
        """Switch to a persisted session file at `session_path`.

        Returns `{"cancelled": True}` when an extension cancels via
        `session_before_switch`.
        """
        if await self._emit_before_switch("resume", session_path):
            return {"cancelled": True}

        previous_session_file = self._session.session_file
        session_manager = SessionManager.open(session_path, cwd_override=cwd_override)
        assert_session_cwd_exists(session_manager, self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self._agent_dir,
                session_manager=session_manager,
                session_start_event=SessionStartEvent(reason="resume", previous_session_file=previous_session_file),
            )
        )
        await self._finish_session_replacement()
        return {"cancelled": False}

    async def new_session(self, parent_session: str | None = None) -> dict[str, bool]:
        """Start a fresh session.

        Returns `{"cancelled": True}` when an extension cancels via
        `session_before_switch`.
        """
        if await self._emit_before_switch("new"):
            return {"cancelled": True}

        previous_session_file = self._session.session_file
        session_dir = self._session.session_manager.get_session_dir()
        session_manager = (
            SessionManager.create(self.cwd, session_dir)
            if self._session.session_manager.is_persisted()
            else SessionManager.in_memory(self.cwd)
        )
        if parent_session:
            session_manager.new_session(NewSessionOptions(parent_session=parent_session))

        await self._teardown_current("new", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=self.cwd,
                agent_dir=self._agent_dir,
                session_manager=session_manager,
                session_start_event=SessionStartEvent(reason="new", previous_session_file=previous_session_file),
            )
        )
        await self._finish_session_replacement()
        return {"cancelled": False}

    async def fork(self, entry_id: str, position: Literal["before", "at"] = "before") -> dict[str, Any]:
        """Fork the session tree at (`position="at"`) or before (`position="before"`) `entry_id`.

        Returns `{"cancelled": True}` when an extension cancels via
        `session_before_fork`.
        """
        if await self._emit_before_fork(entry_id, position):
            return {"cancelled": True}

        selected_entry = self._session.session_manager.get_entry(entry_id)
        if selected_entry is None:
            raise ValueError("Invalid entry ID for forking")

        selected_text: str | None = None
        if position == "at":
            target_leaf_id: str | None = selected_entry.id
        else:
            if selected_entry.type != "message" or selected_entry.message.role != "user":
                raise ValueError("Invalid entry ID for forking")
            target_leaf_id = selected_entry.parent_id
            selected_text = _extract_user_message_text(selected_entry.message.content)

        # Captured before the session-manager mutations below: the in-memory
        # branch reuses the *current* session manager, so `session_file` can
        # change underneath us.
        previous_session_file = self._session.session_file

        if self._session.session_manager.is_persisted():
            current_session_file = self._session.session_file
            if not current_session_file:
                raise ValueError("Persisted session is missing a session file")
            session_dir = self._session.session_manager.get_session_dir()
            if not target_leaf_id:
                session_manager = SessionManager.create(self.cwd, session_dir)
                session_manager.new_session(NewSessionOptions(parent_session=current_session_file))
            else:
                if not Path(current_session_file).exists():
                    raise ValueError(
                        "This session has not been saved yet. Wait for the first assistant response "
                        "before cloning or forking it."
                    )
                session_manager = SessionManager.open(current_session_file, session_dir)
                forked_session_path = session_manager.create_branched_session(target_leaf_id)
                if not forked_session_path:
                    raise ValueError("Failed to create forked session")
        else:
            session_manager = self._session.session_manager
            if not target_leaf_id:
                session_manager.new_session(NewSessionOptions(parent_session=self._session.session_file))
            else:
                session_manager.create_branched_session(target_leaf_id)

        await self._teardown_current("fork", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self._agent_dir,
                session_manager=session_manager,
                session_start_event=SessionStartEvent(reason="fork", previous_session_file=previous_session_file),
            )
        )
        await self._finish_session_replacement()
        return {"cancelled": False, "selected_text": selected_text}

    async def import_from_jsonl(self, input_path: str, cwd_override: str | None = None) -> dict[str, bool]:
        """Import a session JSONL file and switch runtime state to the imported session.

        Returns `{"cancelled": True}` when an extension cancels via
        `session_before_switch`. Raises `SessionImportFileNotFoundError` when
        `input_path` does not exist.
        """
        resolved_path = resolve_path(input_path)
        if not Path(resolved_path).exists():
            raise SessionImportFileNotFoundError(resolved_path)

        session_dir = self._session.session_manager.get_session_dir()
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        destination_path = str(Path(session_dir) / Path(resolved_path).name)
        if await self._emit_before_switch("resume", destination_path):
            return {"cancelled": True}

        previous_session_file = self._session.session_file
        if str(Path(destination_path).resolve()) != resolved_path:
            copyfile(resolved_path, destination_path)

        session_manager = SessionManager.open(destination_path, session_dir, cwd_override)
        assert_session_cwd_exists(session_manager, self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self._agent_dir,
                session_manager=session_manager,
                session_start_event=SessionStartEvent(reason="resume", previous_session_file=previous_session_file),
            )
        )
        await self._finish_session_replacement()
        return {"cancelled": False}

    async def dispose(self) -> None:
        runner = self._session.extension_runner
        if runner.has_handlers("session_shutdown"):
            await runner.emit(SessionShutdownEvent(reason="quit"))
        if self._before_session_invalidate is not None:
            self._before_session_invalidate()
        self._session.dispose()


async def create_agent_session_runtime(
    create_runtime: CreateAgentSessionRuntimeFactory,
    *,
    cwd: str,
    agent_dir: str,
    session_manager: SessionManager,
) -> AgentSessionRuntime:
    """Create the initial runtime from a runtime factory and initial session target.

    The same factory is stored on the returned `AgentSessionRuntime` and reused for
    later new/switch/fork/import flows.
    """
    assert_session_cwd_exists(session_manager, cwd)
    result = await create_runtime(cwd=cwd, agent_dir=agent_dir, session_manager=session_manager)
    return AgentSessionRuntime(result.session, agent_dir, create_runtime, result.model_fallback_message)


# ============================================================================
# `PiAgentSessionRuntime` / `PiAgentSessionRuntimeService`: the real
# `pi_server.types.PiSessionRuntime` / `PiServerService` adapter
# ============================================================================

_RUNNING_TOOL_ITEM_STATUS: Literal["running"] = "running"
_DELTA_KIND_BY_EVENT_TYPE = {"text_delta": "text", "thinking_delta": "thinking", "toolcall_delta": "toolCall"}


@dataclass
class _RunningToolItem:
    id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class PiAgentSessionRuntime:
    """`PiSessionRuntime` wrapping one live `AgentSession` for its whole lifetime.

    Unlike `AgentSessionRuntime` (which replaces its wrapped session on
    `/new`, `/resume`, `/fork`), a `PiSessionRuntime` maps 1:1 to a single
    durable session for as long as it is acquired; `pi_server`'s
    `LiveSessionManager` re-acquires a fresh runtime (via
    `PiServerService.create_session` / `open_session`) for every session
    switch instead of asking one runtime to replace its session in place.
    """

    def __init__(
        self, session: AgentSession, *, name: str | None = None, on_dispose: Callable[[], None] | None = None
    ) -> None:
        self._session = session
        self._name = name
        self._on_dispose = on_dispose
        self._id_counter = itertools.count(1)
        self._transcript: list[dict[str, Any]] = []
        self._streaming_assistant_id: str | None = None
        self._pending_tool_calls: dict[str, ToolCall] = {}
        self._running_tool_items: dict[str, _RunningToolItem] = {}
        self._revision = 0
        self._created_at = now_ms()
        self._updated_at = self._created_at
        self._listeners: set[Callable[[PiSessionRuntimeEvent], None]] = set()
        self._error: PiServerError | None = None
        self._unsubscribe = session.subscribe(self._on_session_event)
        self._seed_transcript_from_existing_messages()

    # -- Construction ---------------------------------------------------

    def _next_id(self) -> str:
        return f"item-{next(self._id_counter)}"

    def _seed_transcript_from_existing_messages(self) -> None:
        """Build the initial transcript from a resumed/forked session's persisted messages."""
        for message in self._session.messages:
            role = getattr(message, "role", None)
            if role == "user":
                self._transcript.append(to_protocol_user_message(message, UserTranscriptOptions(id=self._next_id())))
            elif role == "assistant":
                item_id = self._next_id()
                for part in message.content:
                    if getattr(part, "type", None) == "toolCall":
                        self._pending_tool_calls[part.id] = part
                self._transcript.append(to_protocol_assistant_message(message, AssistantTranscriptOptions(id=item_id)))
            elif role == "toolResult":
                call = self._pending_tool_calls.pop(message.tool_call_id, None)
                if call is None:
                    call = ToolCall(id=message.tool_call_id, name=message.tool_name, arguments={})
                self._transcript.append(
                    to_protocol_tool_result_message(message, ToolTranscriptOptions(id=self._next_id(), call=call))
                )
            # `custom`/`bashExecution`/other harness-only roles have no protocol transcript shape; dropped.

    # -- PiSessionRuntime -------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        model = self._session.model
        result: dict[str, Any] = {
            "id": self._session.session_id,
            "cwd": self._session.session_manager.get_cwd(),
            "createdAt": self._created_at,
            "updatedAt": self._updated_at,
            "phase": self.get_phase(),
            "model": {"provider": model.provider, "id": model.id} if model is not None else {"provider": "", "id": ""},
            "thinkingLevel": self._session.thinking_level,
            "attached": False,
            "locked": True,
            "revision": self._revision,
            "transcript": list(self._transcript),
            "queuedSteer": self._build_queued_steer(),
            "queuedSteerCount": len(self._session.get_steering_messages()),
        }
        name = self._name or self._session.session_name
        if name:
            result["name"] = name
        return result

    def _build_queued_steer(self) -> list[dict[str, Any]]:
        queued: list[dict[str, Any]] = []
        for index, text in enumerate(self._session.get_steering_messages()):
            queued.append(
                {
                    "id": f"queued-steer-{index}",
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                    "timestamp": self._updated_at,
                }
            )
        return queued

    def get_phase(self) -> str:
        if self._session.retry_attempt > 0:
            return "retry"
        if self._session.is_compacting:
            return "compaction"
        if self._session.is_streaming:
            return "turn"
        return "idle"

    async def prompt(self, input: PromptInput) -> None:
        if not self._session.is_idle:
            raise PiServerError("busy", "A prompt is already running")
        await self._session.prompt(input.text)

    async def steer(self, input: SteerInput) -> None:
        if self._session.is_idle:
            raise PiServerError("busy", "There is no active prompt to steer")
        await self._session.steer(input.text)

    async def abort(self) -> None:
        await self._session.abort()

    async def set_model(self, model: dict[str, Any]) -> None:
        if not self._session.is_idle:
            raise PiServerError("busy", "Session is busy")
        resolved = self._session.model_runtime.get_model(model["provider"], model["id"])
        if resolved is None:
            raise PiServerError("invalid_request", f"Unknown model: {model.get('provider')}/{model.get('id')}")
        await self._session.set_model(resolved)

    async def set_thinking(self, thinking_level: str) -> None:
        if not self._session.is_idle:
            raise PiServerError("busy", "Session is busy")
        self._session.set_thinking_level(thinking_level)  # type: ignore[arg-type]

    def subscribe(self, listener: Callable[[PiSessionRuntimeEvent], None]) -> Callable[[], None]:
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    async def dispose(self) -> None:
        self._unsubscribe()
        self._session.dispose()
        if self._on_dispose is not None:
            self._on_dispose()

    def _touch(self) -> None:
        self._revision += 1
        self._updated_at = now_ms()

    def _broadcast_snapshot(self) -> None:
        self._touch()
        for listener in list(self._listeners):
            listener(SnapshotRuntimeEvent())

    def _broadcast_progress(self, progress: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(ProgressRuntimeEvent(progress=progress))

    def _on_session_event(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        try:
            if event_type == "message_start":
                self._on_message_start(event.message)
            elif event_type == "message_update":
                self._on_message_update(event)
            elif event_type == "message_end":
                self._on_message_end(event.message)
            elif event_type == "tool_execution_start":
                self._on_tool_execution_start(event)
            elif event_type in ("agent_settled", "queue_update", "thinking_level_changed"):
                self._broadcast_snapshot()
        except PiServerError as error:
            self._error = error
            for listener in list(self._listeners):
                listener(ErrorRuntimeEvent(error=error))

    def _on_message_start(self, message: Any) -> None:
        role = getattr(message, "role", None)
        if role != "assistant":
            return
        item_id = self._next_id()
        self._streaming_assistant_id = item_id
        item = to_protocol_assistant_message(message, AssistantTranscriptOptions(id=item_id))
        self._transcript.append(item)
        self._touch()
        self._broadcast_progress({"type": "item_started", "item": item})

    def _on_message_update(self, event: Any) -> None:
        if self._streaming_assistant_id is None:
            return
        inner = event.assistant_message_event
        kind = _DELTA_KIND_BY_EVENT_TYPE.get(getattr(inner, "type", None))
        if kind is None:
            return
        self._broadcast_progress(
            {
                "type": "assistant_delta",
                "messageId": self._streaming_assistant_id,
                "contentIndex": inner.content_index,
                "kind": kind,
                "delta": inner.delta,
            }
        )

    def _on_message_end(self, message: Any) -> None:
        role = getattr(message, "role", None)
        if role == "user":
            item = to_protocol_user_message(message, UserTranscriptOptions(id=self._next_id()))
            self._transcript.append(item)
            self._touch()
            self._broadcast_progress({"type": "item_started", "item": item})
        elif role == "assistant":
            item_id = self._streaming_assistant_id or self._next_id()
            self._streaming_assistant_id = None
            for part in message.content:
                if getattr(part, "type", None) == "toolCall":
                    self._pending_tool_calls[part.id] = part
            item = to_protocol_assistant_message(message, AssistantTranscriptOptions(id=item_id))
            self._replace_or_append(item_id, item)
            self._touch()
            self._broadcast_progress({"type": "item_finished", "item": item})
        elif role == "toolResult":
            running = self._running_tool_items.pop(message.tool_call_id, None)
            call = self._pending_tool_calls.pop(message.tool_call_id, None)
            if call is None:
                call = ToolCall(id=message.tool_call_id, name=message.tool_name, arguments={})
            item_id = running.id if running is not None else self._next_id()
            item = to_protocol_tool_result_message(message, ToolTranscriptOptions(id=item_id, call=call))
            self._replace_or_append(item_id, item)
            self._touch()
            self._broadcast_progress({"type": "item_finished", "item": item})

    def _on_tool_execution_start(self, event: Any) -> None:
        item_id = self._next_id()
        call = self._pending_tool_calls.get(event.tool_call_id)
        arguments = call.arguments if call is not None else (event.args if isinstance(event.args, dict) else {})
        self._running_tool_items[event.tool_call_id] = _RunningToolItem(
            id=item_id, tool_call_id=event.tool_call_id, tool_name=event.tool_name, arguments=arguments
        )
        item = {
            "id": item_id,
            "role": "tool",
            "toolCallId": event.tool_call_id,
            "toolName": event.tool_name,
            "input": to_protocol_json_value(arguments),
            "content": [],
            "timestamp": now_ms(),
            "status": _RUNNING_TOOL_ITEM_STATUS,
            "isError": False,
        }
        self._transcript.append(item)
        self._touch()
        self._broadcast_progress({"type": "item_started", "item": item})

    def _replace_or_append(self, item_id: str, item: dict[str, Any]) -> None:
        for index, existing in enumerate(self._transcript):
            if existing["id"] == item_id:
                self._transcript[index] = item
                return
        self._transcript.append(item)


class PiAgentSessionRuntimeService:
    """`PiServerService` backed by real, persisted `AgentSession`s under one session directory."""

    def __init__(
        self,
        *,
        agent_dir: str,
        default_cwd: str,
        model_runtime: ModelRuntime,
        settings_manager: SettingsManager | None = None,
        session_dir: str | None = None,
    ) -> None:
        self._agent_dir = agent_dir
        self._default_cwd = default_cwd
        self._model_runtime = model_runtime
        self._settings_manager = settings_manager or SettingsManager.create(default_cwd, agent_dir)
        self._session_dir = session_dir or get_default_session_dir(default_cwd, agent_dir)
        self._runtimes: dict[str, PiAgentSessionRuntime] = {}

    async def list_sessions(self) -> list[dict[str, Any]]:
        sessions = await SessionManager.list_all(session_dir=self._session_dir)
        return [
            {
                "id": info.id,
                "createdAt": int(info.created.timestamp() * 1000),
                "updatedAt": int(info.modified.timestamp() * 1000),
                "cwd": info.cwd,
                **({"sessionName": info.name} if info.name else {}),
            }
            for info in sessions
        ]

    async def list_models(self) -> list[dict[str, Any]]:
        return [
            to_protocol_model_metadata(model, self._model_runtime.has_configured_auth(model.provider))
            for model in self._model_runtime.get_models()
        ]

    async def create_session(self, options: CreateSessionOptions) -> Any:
        cwd = resolve_path(options.cwd) if options.cwd else self._default_cwd
        model = self._resolve_model_ref(options.model)
        session_manager = SessionManager.create(cwd, self._session_dir, NewSessionOptions(id=options.id))
        if options.name:
            session_manager.append_session_info(options.name)
        resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=self._agent_dir))
        resource_loader.reload()

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                agent_dir=self._agent_dir,
                model_runtime=self._model_runtime,
                model=model,
                thinking_level=options.thinking_level,  # type: ignore[arg-type]
                resource_loader=resource_loader,
                session_manager=session_manager,
                settings_manager=self._settings_manager,
            )
        )
        runtime = PiAgentSessionRuntime(
            result.session, name=options.name, on_dispose=lambda: self._runtimes.pop(options.id, None)
        )
        self._runtimes[options.id] = runtime
        return runtime

    async def open_session(self, session_id: str) -> Any:
        existing = self._runtimes.get(session_id)
        if existing is not None:
            raise PiServerError("session_locked", f"Session is locked: {session_id}")
        sessions = await SessionManager.list_all(session_dir=self._session_dir)
        info = next((entry for entry in sessions if entry.id == session_id), None)
        if info is None:
            raise PiServerError("not_found", f"Unknown session: {session_id}")

        session_manager = SessionManager.open(info.path, self._session_dir)
        resource_loader = ResourceLoader(
            ResourceLoaderOptions(cwd=session_manager.get_cwd(), agent_dir=self._agent_dir)
        )
        resource_loader.reload()

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=session_manager.get_cwd(),
                agent_dir=self._agent_dir,
                model_runtime=self._model_runtime,
                resource_loader=resource_loader,
                session_manager=session_manager,
                settings_manager=self._settings_manager,
            )
        )
        runtime = PiAgentSessionRuntime(
            result.session, name=info.name, on_dispose=lambda: self._runtimes.pop(session_id, None)
        )
        self._runtimes[session_id] = runtime
        return runtime

    def _resolve_model_ref(self, model_ref: dict[str, Any] | None) -> Model | None:
        if model_ref is None:
            return None
        model = self._model_runtime.get_model(model_ref["provider"], model_ref["id"])
        if model is None:
            raise PiServerError("invalid_request", f"Unknown model: {model_ref}")
        return model


__all__ = [
    "AgentSessionRuntime",
    "CreateAgentSessionRuntimeFactory",
    "PiAgentSessionRuntime",
    "PiAgentSessionRuntimeService",
    "SessionImportFileNotFoundError",
    "create_agent_session_runtime",
]
