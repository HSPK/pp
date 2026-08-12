"""A reactive, single-session wrapper over `PiClient`.

Python port of `packages/coding-agent/src/client/remote-session.ts`. Wraps one
acquired `SessionHandle` in a small lifecycle state machine (`unbound` /
`ready` / `busy` / `disposed`) plus incrementally-reconstructed transcript
state (`transcript.py`), so a UI only needs to `subscribe()` to one object
rather than juggle `PiClient`/`SessionHandle` directly.

Wire values (`snapshot`, transcript items, model refs, ...) are plain dicts
here, matching this port's convention of validating protocol shapes with
`pi_protocol.schemas` rather than typed wrapper classes.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pi_client import ConnectionState, ConnectionStateChange, PiClient, SessionHandle, Unsubscribe

from .transcript import (
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)

RemoteSessionOperation = Literal["open", "create", "submit", "abort", "set_model", "set_thinking", "reconnect"]


@dataclass(eq=False)
class RemoteSessionLifecycle:
    status: Literal["unbound", "ready", "busy", "disposed"]
    operation: RemoteSessionOperation | None = None


@dataclass
class RemoteSessionState:
    lifecycle: RemoteSessionLifecycle
    snapshot: dict[str, Any] | None
    transcript: list[dict[str, Any]]


@dataclass
class CreateRemoteSessionOptions:
    cwd: str
    model: dict[str, Any] | None = None
    thinking_level: str | None = None


@dataclass
class RemoteSessionOptions:
    on_listener_error: Callable[[Exception], None] | None = None


class RemoteSessionDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Remote session is disposed")


async def _settle_remote_session_disposal(cleanup: list[Awaitable[None]]) -> None:
    results = await asyncio.gather(*cleanup, return_exceptions=True)
    errors = [
        result
        for result in results
        if isinstance(result, BaseException) and not isinstance(result, RemoteSessionDisposedError)
    ]
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup("Failed to dispose remote session", errors)


class RemoteSession:
    """One acquired session lease plus reactive lifecycle/transcript state.

    Construct via `RemoteSession.open(...)`/`RemoteSession.create(...)`, not
    directly (mirrors TS's private constructor + static factory methods)."""

    def __init__(self, client: PiClient, options: RemoteSessionOptions | None = None) -> None:
        self._client = client
        self._on_listener_error = (options or RemoteSessionOptions()).on_listener_error
        self._lifecycle = RemoteSessionLifecycle(status="unbound")
        self._handle: SessionHandle | None = None
        self._transcript: TranscriptState | None = None
        self._unsubscribe_snapshot: Unsubscribe | None = None
        self._unsubscribe_events: Unsubscribe | None = None
        self._listeners: set[Callable[[RemoteSessionState], None]] = set()
        self._pending_attachment_operations: set[asyncio.Future[None]] = set()
        self._active_operation_states: set[RemoteSessionLifecycle] = set()
        self._dispose_future: asyncio.Future[None] | None = None
        self._dispose_signal: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    # -- Read-only state --------------------------------------------------

    @property
    def id(self) -> str | None:
        return self._handle.id if self._handle is not None else None

    @property
    def state(self) -> RemoteSessionState:
        return RemoteSessionState(
            lifecycle=self._lifecycle,
            snapshot=self._transcript.snapshot if self._transcript is not None else None,
            transcript=select_transcript(self._transcript) if self._transcript is not None else [],
        )

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._transcript.snapshot if self._transcript is not None else None

    @property
    def phase(self) -> str | None:
        snapshot = self.snapshot
        return snapshot["phase"] if snapshot is not None else None

    @property
    def operation(self) -> RemoteSessionOperation | None:
        return self._lifecycle.operation if self._lifecycle.status == "busy" else None

    @property
    def models(self) -> list[dict[str, Any]]:
        snapshot = self._client.snapshot
        return snapshot["models"] if snapshot is not None else []

    @property
    def sessions(self) -> list[dict[str, Any]]:
        snapshot = self._client.snapshot
        return snapshot["sessions"] if snapshot is not None else []

    @property
    def connection_state(self) -> ConnectionState:
        return self._client.connection_state

    @property
    def disposed(self) -> bool:
        return self._lifecycle.status == "disposed"

    def subscribe(self, listener: Callable[[RemoteSessionState], None]) -> Unsubscribe:
        self._assert_not_disposed()
        self._listeners.add(listener)
        self._call_listener(listener, self.state)
        return lambda: self._listeners.discard(listener)

    def on_connection_state_change(self, listener: Callable[[ConnectionStateChange], None]) -> Unsubscribe:
        self._assert_not_disposed()
        return self._client.on_connection_state_change(listener)

    # -- Construction -------------------------------------------------------

    @classmethod
    async def open(
        cls, client: PiClient, session_id: str, options: RemoteSessionOptions | None = None
    ) -> RemoteSession:
        session = cls(client, options)
        try:
            await session.open_session(session_id)
            return session
        except Exception:
            await session.dispose()
            raise

    async def open_session(self, session_id: str) -> None:
        if self._handle is not None and self._handle.id == session_id and self._lifecycle.status == "ready":
            return
        await self._replace("open", lambda: self._client.acquire_session(session_id, "exclusive"))

    @classmethod
    async def create(
        cls, client: PiClient, create_options: CreateRemoteSessionOptions, options: RemoteSessionOptions | None = None
    ) -> RemoteSession:
        session = cls(client, options)
        try:
            await session.create_session(create_options)
            return session
        except Exception:
            await session.dispose()
            raise

    async def create_session(self, options: CreateRemoteSessionOptions) -> None:
        await self._replace(
            "create",
            lambda: self._client.create_session(
                cwd=options.cwd, model=options.model, thinking_level=options.thinking_level
            ),
        )

    # -- Operations -----------------------------------------------------

    async def submit(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self._assert_available()
        handle = self._require_handle()
        if self.phase not in ("idle", "turn"):
            raise RuntimeError(f"Session cannot accept input during {self.phase or 'unknown'} phase")

        async def run() -> None:
            if self.phase == "idle":
                await handle.prompt(normalized)
            else:
                await handle.steer(normalized)

        await self._run_operation("submit", run)

    async def abort(self) -> None:
        preempting_submit = self._lifecycle.status == "busy" and self._lifecycle.operation == "submit"
        if preempting_submit:
            self._assert_not_disposed()
        else:
            self._assert_available()
        handle = self._require_handle()
        if self.phase == "idle" and not preempting_submit:
            return

        async def run() -> None:
            await handle.abort()

        await self._run_operation("abort", run, preempt=preempting_submit)

    async def set_model(self, model: dict[str, Any]) -> None:
        async def run() -> None:
            await self._require_handle().set_model(model)

        await self._run_idle_operation("set_model", "change model", run)

    async def set_thinking(self, thinking_level: str) -> None:
        async def run() -> None:
            await self._require_handle().set_thinking(thinking_level)

        await self._run_idle_operation("set_thinking", "change thinking level", run)

    async def reconnect(self) -> None:
        self._assert_available()
        session_id = self._require_handle().id

        async def run() -> None:
            async def attempt() -> None:
                await self._client.reconnect()
                handle = await self._client.acquire_session(session_id, "exclusive")
                await self._assert_not_disposed_after_await(handle)
                self._bind(handle)

            await self._track_attachment_operation(attempt)

        await self._run_operation("reconnect", run)

    def dispose(self) -> Awaitable[None]:
        if self._dispose_future is not None:
            return self._dispose_future
        handle = self._handle
        self._lifecycle = RemoteSessionLifecycle(status="disposed")
        if not self._dispose_signal.done():
            self._dispose_signal.set_result(None)
        self._clear_subscriptions()
        self._handle = None
        self._transcript = None
        cleanup: list[Awaitable[None]] = list(self._pending_attachment_operations)
        if handle is not None:
            cleanup.append(handle.dispose())
        self._dispose_future = asyncio.ensure_future(_settle_remote_session_disposal(cleanup))
        self._notify()
        self._listeners.clear()
        return self._dispose_future

    async def __aenter__(self) -> RemoteSession:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.dispose()

    # -- Internals --------------------------------------------------------

    async def _replace(
        self, operation: Literal["open", "create"], prepare: Callable[[], Awaitable[SessionHandle]]
    ) -> None:
        self._assert_available()
        if self._handle is not None and self.phase != "idle":
            raise RuntimeError(f"Cannot {operation} a session while session is {self.phase or 'unavailable'}")

        async def run() -> None:
            async def attempt() -> None:
                await self._prepare_replacement(operation, prepare)

            await self._track_attachment_operation(attempt)

        await self._run_operation(operation, run)

    async def _track_attachment_operation(self, run: Callable[[], Awaitable[None]]) -> None:
        pending = asyncio.ensure_future(run())
        self._pending_attachment_operations.add(pending)
        try:
            await pending
        finally:
            self._pending_attachment_operations.discard(pending)

    async def _prepare_replacement(
        self, operation: Literal["open", "create"], prepare: Callable[[], Awaitable[SessionHandle]]
    ) -> None:
        previous = self._handle
        next_handle = await prepare()
        await self._assert_not_disposed_after_await(next_handle)
        next_snapshot = next_handle.snapshot
        if next_snapshot is None:
            await self._detach(next_handle)
            raise RuntimeError(f"Session {next_handle.id} did not provide a snapshot")
        if previous is not None and previous.id != next_handle.id and previous.attached and self.phase != "idle":
            await self._detach(next_handle)
            raise RuntimeError(f"Cannot {operation} a session while session is {self.phase or 'unavailable'}")
        if previous is not None and previous.id != next_handle.id and previous.attached:
            try:
                await previous.detach()
            except Exception as error:
                try:
                    await self._detach(next_handle)
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "Failed to replace remote session attachment", [error, cleanup_error]
                    ) from cleanup_error
                raise
        await self._assert_not_disposed_after_await(next_handle)
        self._bind(next_handle, next_snapshot)

    async def _run_idle_operation(
        self,
        operation: Literal["set_model", "set_thinking"],
        description: str,
        run: Callable[[], Awaitable[None]],
    ) -> None:
        self._assert_available()
        self._require_handle()
        if self.phase != "idle":
            raise RuntimeError(f"Cannot {description} while session is {self.phase or 'unavailable'}")
        await self._run_operation(operation, run)

    async def _run_operation(
        self, operation: RemoteSessionOperation, run: Callable[[], Awaitable[None]], preempt: bool = False
    ) -> None:
        if preempt:
            self._assert_not_disposed()
        else:
            self._assert_available()
        previous = self._lifecycle
        busy = RemoteSessionLifecycle(status="busy", operation=operation)
        self._lifecycle = busy
        self._active_operation_states.add(busy)
        self._notify()
        running = asyncio.ensure_future(run())

        async def _raise_on_dispose() -> None:
            await asyncio.shield(self._dispose_signal)
            raise RuntimeError("Remote session is disposed")

        dispose_waiter = asyncio.ensure_future(_raise_on_dispose())
        try:
            # Mirrors `Promise.race`: whichever settles first decides the outcome, but the
            # loser keeps running in the background rather than being cancelled (Promises
            # can't be cancelled in JS; `running` may still be an in-flight server request).
            done, _pending = await asyncio.wait({running, dispose_waiter}, return_when=asyncio.FIRST_COMPLETED)
            if running in done:
                running.result()
            else:

                def _drain(task: asyncio.Task[Any]) -> None:
                    with contextlib.suppress(BaseException):
                        task.exception()

                running.add_done_callback(_drain)
                dispose_waiter.result()
        finally:
            if not dispose_waiter.done():
                dispose_waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await dispose_waiter
            self._active_operation_states.discard(busy)
            if not self.disposed and self._lifecycle is busy:
                if preempt and previous in self._active_operation_states:
                    self._lifecycle = previous
                elif self._handle is not None:
                    self._lifecycle = RemoteSessionLifecycle(status="ready")
                else:
                    self._lifecycle = RemoteSessionLifecycle(status="unbound")
                self._notify()

    def _bind(self, handle: SessionHandle, known_snapshot: dict[str, Any] | None = None) -> None:
        snapshot = known_snapshot if known_snapshot is not None else handle.snapshot
        if snapshot is None:
            raise RuntimeError(f"Session {handle.id} did not provide a snapshot")
        self._clear_subscriptions()
        self._handle = handle
        self._transcript = create_transcript_state(snapshot)
        self._unsubscribe_snapshot = handle.subscribe(self._on_snapshot)
        self._unsubscribe_events = handle.on_event(self._on_event)

    def _on_snapshot(self, next_snapshot: dict[str, Any]) -> None:
        if self._transcript is None:
            return
        self._transcript = apply_transcript_snapshot(self._transcript, next_snapshot)
        self._notify()

    def _on_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "session_removed":
            self._clear_subscriptions()
            self._handle = None
            self._transcript = None
            if self._lifecycle.status != "busy":
                self._lifecycle = RemoteSessionLifecycle(status="unbound")
            self._notify()
            return
        if event["type"] != "session_progress" or self._transcript is None:
            return
        self._transcript = apply_transcript_progress(self._transcript, event["progress"])
        self._notify()

    def _notify(self) -> None:
        state = self.state
        for listener in list(self._listeners):
            self._call_listener(listener, state)

    def _call_listener(self, listener: Callable[[RemoteSessionState], None], state: RemoteSessionState) -> None:
        try:
            listener(state)
        except Exception as error:
            self._report_listener_error(error)

    def _report_listener_error(self, error: Exception) -> None:
        if self._on_listener_error is None:
            return
        with contextlib.suppress(Exception):
            self._on_listener_error(error)

    def _clear_subscriptions(self) -> None:
        if self._unsubscribe_snapshot is not None:
            self._unsubscribe_snapshot()
        if self._unsubscribe_events is not None:
            self._unsubscribe_events()
        self._unsubscribe_snapshot = None
        self._unsubscribe_events = None

    def _require_handle(self) -> SessionHandle:
        if self._handle is None:
            raise RuntimeError("No remote session is attached")
        return self._handle

    def _assert_available(self) -> None:
        self._assert_not_disposed()
        if self._lifecycle.status == "busy":
            raise RuntimeError(f"Remote session is busy with {self._lifecycle.operation}")

    def _assert_not_disposed(self) -> None:
        if self.disposed:
            raise RuntimeError("Remote session is disposed")

    async def _assert_not_disposed_after_await(self, handle: SessionHandle) -> None:
        if not self.disposed:
            return
        await self._detach(handle)
        raise RemoteSessionDisposedError()

    async def _detach(self, handle: SessionHandle) -> None:
        await handle.dispose()


__all__ = [
    "CreateRemoteSessionOptions",
    "RemoteSession",
    "RemoteSessionDisposedError",
    "RemoteSessionLifecycle",
    "RemoteSessionOperation",
    "RemoteSessionOptions",
    "RemoteSessionState",
]
