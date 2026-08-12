"""Top-level RPC client.

Python port of `packages/client/src/client.ts`. `PiClient` owns the wire
connection, the reactive `ClientState`, request/response correlation, and
session lease bookkeeping (shared vs. exclusive attachments, detach-then-
reattach serialization, and lease invalidation on disconnect).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_protocol import ProtocolValidationError, encode_client_message

from .connection import Connection
from .errors import (
    PiClientDisposedError,
    PiDisconnectedError,
    PiServerError,
    PiSessionDetachedError,
    PiSessionOwnershipError,
    to_error,
)
from .session_handle import SessionHandle, SessionLeaseMode
from .state import ClientState
from .types import ConnectionState, ConnectionStateChange, PiClientOptions, Unsubscribe


@dataclass
class _SessionLeaseToken:
    mode: SessionLeaseMode


@dataclass
class _PendingRequest:
    command: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]


_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> asyncio.Task[Any]:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


class PiClient:
    def __init__(self, options: PiClientOptions) -> None:
        self._options = options
        self._state = ClientState(options.on_listener_error)
        self._connection = Connection(
            transport_factory=options.transport_factory,
            max_frame_length=options.max_frame_length,
            on_handshake=self._state.apply_server_snapshot,
            on_message=self._handle_message,
            on_state_change=self._handle_connection_state_change,
        )
        self._pending_requests: dict[str, _PendingRequest] = {}
        self._session_lease_counts: dict[str, int] = {}
        self._exclusive_session_leases: dict[str, _SessionLeaseToken] = {}
        self._session_lease_generations: dict[str, int] = {}
        self._session_attachments: dict[str, asyncio.Future[None]] = {}
        self._session_detachments: dict[str, asyncio.Future[None]] = {}
        self._session_cleanup_required: set[str] = set()
        self._session_reconciliations: dict[str, asyncio.Future[None]] = {}
        self._connection_state_listeners: set[Callable[[ConnectionStateChange], None]] = set()
        self._request_sequence = 0
        self._disposed = False
        self._dispose_future: asyncio.Future[None] | None = None

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection.state

    @property
    def connected(self) -> bool:
        return self._connection.state == "connected"

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._state.snapshot

    @classmethod
    async def open(cls, options: PiClientOptions) -> PiClient:
        """Construct a client and connect it. TypeScript names this static method `connect`;
        it is `open` here because Python cannot overload an instance method and a
        classmethod under the same name."""
        client = cls(options)
        try:
            await client.connect()
            return client
        except Exception:
            await client.dispose()
            raise

    def connect(self) -> Awaitable[dict[str, Any]]:
        if self._disposed:
            future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
            future.set_exception(PiClientDisposedError())
            return future
        if self._connection.state == "disconnected":
            self._state.reset()
        return self._connection.connect()

    def reconnect(self) -> Awaitable[dict[str, Any]]:
        return self.connect()

    def disconnect(self, reason: str = "Client disconnected") -> None:
        self._connection.disconnect(reason)

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        self._assert_not_disposed()
        return self._state.subscribe(listener)

    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        self._assert_not_disposed()
        return self._state.on_event(listener)

    def on_connection_state_change(self, listener: Callable[[ConnectionStateChange], None]) -> Unsubscribe:
        self._assert_not_disposed()
        self._connection_state_listeners.add(listener)
        return lambda: self._connection_state_listeners.discard(listener)

    async def list_sessions(self) -> list[dict[str, Any]]:
        result = await self._request({"command": "list"})
        return result["sessions"]

    async def create_session(
        self,
        cwd: str | None = None,
        name: str | None = None,
        model: dict[str, Any] | None = None,
        thinking_level: str | None = None,
    ) -> SessionHandle:
        command: dict[str, Any] = {"command": "create"}
        if cwd is not None:
            command["cwd"] = cwd
        if name is not None:
            command["name"] = name
        if model is not None:
            command["model"] = model
        if thinking_level is not None:
            command["thinkingLevel"] = thinking_level
        result = await self._request(command)
        token = self._reserve_session_lease(result["session"]["id"], "exclusive")
        return self._create_session_lease(result["session"]["id"], token)

    async def attach_session(self, session_id: str) -> SessionHandle:
        return await self.acquire_session(session_id, "shared")

    async def acquire_session(self, session_id: str, mode: SessionLeaseMode) -> SessionHandle:
        self._assert_not_disposed()
        token = self._reserve_session_lease(session_id, mode)
        try:
            detachment = self._session_detachments.get(session_id)
            if detachment is not None:
                with contextlib.suppress(Exception):
                    await detachment
            reconciled = False
            if session_id in self._session_cleanup_required:
                reconciled = await self._reconcile_session_cleanup(session_id)
            if reconciled or not self._state.is_session_attached(session_id):
                attachment = self._session_attachments.get(session_id)
                if attachment is None:
                    attachment = _spawn(self._attach_session(session_id))
                    self._session_attachments[session_id] = attachment
                try:
                    await attachment
                finally:
                    if self._session_attachments.get(session_id) is attachment:
                        del self._session_attachments[session_id]
            return self._create_session_lease(session_id, token)
        except Exception:
            self._release_session_lease(session_id, token)
            raise

    async def _attach_session(self, session_id: str) -> None:
        previous = self._state.forget_session_snapshot(session_id)
        try:
            await self._request({"command": "attach", "sessionId": session_id})
        except Exception:
            if previous is not None:
                self._state.restore_session_snapshot(previous)
            raise

    def _request(self, command: dict[str, Any]) -> Awaitable[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        if self._disposed:
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            future.set_exception(PiClientDisposedError())
            return future
        if not self.connected:
            future = loop.create_future()
            future.set_exception(PiDisconnectedError())
            return future
        self._request_sequence += 1
        id_ = f"request-{self._request_sequence}"
        future = loop.create_future()
        self._pending_requests[id_] = _PendingRequest(command=command, future=future)
        try:
            frame = encode_client_message(
                {"type": "request", "id": id_, "request": command},
                max_frame_length=self._connection.max_frame_length,
            )
        except Exception as error:
            pending = self._take_pending_request(id_)
            if pending is not None and not pending.future.done():
                pending.future.set_exception(to_error(error))
            return future
        self._connection.send(frame)
        return future

    def _create_session_lease(self, session_id: str, token: _SessionLeaseToken) -> SessionHandle:
        generation = self._session_lease_generations.get(session_id, 0)
        self._session_lease_generations[session_id] = generation
        box: dict[str, Any] = {"state": "active", "release_task": None}

        def refresh_state() -> None:
            if (
                box["state"] in ("active", "releasing")
                and self._session_lease_generations.get(session_id) != generation
            ):
                box["state"] = "invalidated"

        def is_active() -> bool:
            refresh_state()
            return box["state"] == "active" and self._state.is_session_attached(session_id)

        def assert_active() -> None:
            self._assert_not_disposed()
            if not self.connected:
                raise PiDisconnectedError()
            if not is_active():
                raise PiSessionDetachedError(session_id)

        async def release(relinquish_on_failure: bool) -> None:
            refresh_state()
            if box["state"] in ("released", "invalidated"):
                return
            existing_task = box["release_task"]
            if existing_task is not None:
                await existing_task
                return
            assert_active()
            box["state"] = "releasing"

            async def _body() -> None:
                count = self._session_lease_counts.get(session_id, 0)
                if count <= 1:
                    detachment = _spawn(self._detach_request(session_id))
                    self._session_detachments[session_id] = detachment
                    try:
                        await detachment
                        self._release_session_lease(session_id, token)
                    finally:
                        if self._session_detachments.get(session_id) is detachment:
                            del self._session_detachments[session_id]
                else:
                    self._release_session_lease(session_id, token)
                box["state"] = "released"

            async def _wrapped() -> None:
                try:
                    await _body()
                except Exception as error:
                    refresh_state()
                    if box["state"] == "invalidated":
                        return
                    if relinquish_on_failure:
                        self._release_session_lease(session_id, token)
                        self._session_cleanup_required.add(session_id)
                        box["state"] = "released"
                    else:
                        box["state"] = "active"
                        box["release_task"] = None
                    raise error

            task = _spawn(_wrapped())
            box["release_task"] = task
            await task

        def subscribe(listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
            assert_active()

            def guarded(snapshot: dict[str, Any]) -> None:
                if is_active():
                    listener(snapshot)

            return self._state.subscribe_session(session_id, guarded)

        def on_event(listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
            assert_active()

            def guarded(event: dict[str, Any]) -> None:
                if is_active() or event["type"] == "session_removed":
                    listener(event)

            return self._state.on_session_event(session_id, guarded)

        def request(command: dict[str, Any]) -> Awaitable[dict[str, Any]]:
            assert_active()
            return self._request(command)

        callbacks = _Callbacks(
            is_attached=is_active,
            get_snapshot=lambda: self._state.get_session_snapshot(session_id) if is_active() else None,
            subscribe=subscribe,
            on_event=on_event,
            detach=lambda: release(False),
            dispose=lambda: release(True),
            request=request,
        )
        return SessionHandle(session_id, callbacks)

    async def _detach_request(self, session_id: str) -> None:
        await self._request({"command": "detach", "sessionId": session_id})

    def _handle_message(self, message: dict[str, Any]) -> None:
        if message["type"] == "event":
            if message["event"]["type"] == "session_removed":
                self._invalidate_session_leases(message["event"]["sessionId"])
            self._state.apply_event(message["event"])
            return
        pending = self._take_pending_request(message["id"])
        if pending is None:
            self._connection.fail(ProtocolValidationError("Response has no matching request"))
            return
        if not message["ok"]:
            if not pending.future.done():
                pending.future.set_exception(PiServerError(message["error"]))
            return
        if message["result"]["command"] != pending.command["command"]:
            error = ProtocolValidationError(
                f"Response command {message['result']['command']} does not match {pending.command['command']}"
            )
            if not pending.future.done():
                pending.future.set_exception(error)
            self._connection.fail(error)
            return
        self._state.apply_result(message["result"])
        if not pending.future.done():
            pending.future.set_result(message["result"])

    def _handle_connection_state_change(self, change: ConnectionStateChange) -> None:
        if change.state == "disconnected":
            self._state.clear_attachments()
            self._invalidate_all_session_leases()
            self._reject_pending_requests(change.error or PiDisconnectedError())
        self._notify_connection_state_listeners(change)

    def _take_pending_request(self, id_: str) -> _PendingRequest | None:
        return self._pending_requests.pop(id_, None)

    def _reject_pending_requests(self, error: Exception) -> None:
        requests = list(self._pending_requests.values())
        self._pending_requests.clear()
        for request in requests:
            if not request.future.done():
                request.future.set_exception(error)

    def dispose(self) -> Awaitable[None]:
        if self._dispose_future is not None:
            return self._dispose_future
        self._disposed = True
        loop = asyncio.get_event_loop()
        future: asyncio.Future[None] = loop.create_future()
        future.set_result(None)
        self._dispose_future = future
        error = PiClientDisposedError()
        self._reject_pending_requests(error)
        self._connection.disconnect(error)
        self._state.dispose()
        self._invalidate_all_session_leases()
        self._connection_state_listeners.clear()
        return self._dispose_future

    async def __aenter__(self) -> PiClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.dispose()

    def _assert_not_disposed(self) -> None:
        if self._disposed:
            raise PiClientDisposedError()

    async def _reconcile_session_cleanup(self, session_id: str) -> bool:
        if session_id not in self._session_cleanup_required:
            return False
        reconciliation = self._session_reconciliations.get(session_id)
        if reconciliation is None:

            async def _do() -> None:
                try:
                    await self._request({"command": "detach", "sessionId": session_id})
                    self._session_cleanup_required.discard(session_id)
                finally:
                    self._session_reconciliations.pop(session_id, None)

            reconciliation = _spawn(_do())
            self._session_reconciliations[session_id] = reconciliation
        await reconciliation
        return True

    def _reserve_session_lease(self, session_id: str, mode: SessionLeaseMode) -> _SessionLeaseToken:
        count = self._session_lease_counts.get(session_id, 0)
        if mode == "exclusive" and count > 0:
            raise PiSessionOwnershipError(session_id, f"Session {session_id} already has an active lease")
        if mode == "shared" and session_id in self._exclusive_session_leases:
            raise PiSessionOwnershipError(session_id, f"Session {session_id} has an exclusive lease")
        token = _SessionLeaseToken(mode=mode)
        self._session_lease_counts[session_id] = count + 1
        if mode == "exclusive":
            self._exclusive_session_leases[session_id] = token
        return token

    def _release_session_lease(self, session_id: str, token: _SessionLeaseToken) -> None:
        count = self._session_lease_counts.get(session_id, 0)
        if count <= 1:
            self._session_lease_counts.pop(session_id, None)
        else:
            self._session_lease_counts[session_id] = count - 1
        if self._exclusive_session_leases.get(session_id) is token:
            del self._exclusive_session_leases[session_id]

    def _invalidate_session_leases(self, session_id: str) -> None:
        self._session_lease_counts.pop(session_id, None)
        self._exclusive_session_leases.pop(session_id, None)
        self._session_cleanup_required.discard(session_id)
        self._session_lease_generations[session_id] = self._session_lease_generations.get(session_id, 0) + 1

    def _invalidate_all_session_leases(self) -> None:
        for session_id in list(self._session_lease_counts.keys()):
            self._invalidate_session_leases(session_id)
        self._session_cleanup_required.clear()

    def _notify_connection_state_listeners(self, change: ConnectionStateChange) -> None:
        for listener in list(self._connection_state_listeners):
            try:
                listener(change)
            except Exception as error:
                self._report_listener_error(error)

    def _report_listener_error(self, error: BaseException) -> None:
        if self._options.on_listener_error is None:
            return
        with contextlib.suppress(Exception):
            self._options.on_listener_error(to_error(error))


@dataclass
class _Callbacks:
    is_attached: Callable[[], bool]
    get_snapshot: Callable[[], dict[str, Any] | None]
    subscribe: Callable[[Callable[[dict[str, Any]], None]], Unsubscribe]
    on_event: Callable[[Callable[[dict[str, Any]], None]], Unsubscribe]
    detach: Callable[[], Awaitable[None]]
    dispose: Callable[[], Awaitable[None]]
    request: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
