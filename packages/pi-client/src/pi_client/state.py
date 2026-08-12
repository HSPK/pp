"""Client-side reactive state: snapshots, subscriptions, event fan-out.

Python port of `packages/client/src/state.ts`. Snapshots are plain dicts (wire
shape) as validated by `pi_protocol`; revisions gate which updates apply.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from .errors import to_error
from .types import ListenerErrorHandler, Unsubscribe


def _add_mapped_listener(
    listeners_by_id: dict[str, set[Callable[[Any], None]]],
    id_: str,
    listener: Callable[[Any], None],
) -> Unsubscribe:
    listeners = listeners_by_id.get(id_)
    if listeners is None:
        listeners = set()
        listeners_by_id[id_] = listeners
    listeners.add(listener)

    def unsubscribe() -> None:
        listeners.discard(listener)
        if not listeners:
            listeners_by_id.pop(id_, None)

    return unsubscribe


def _get_event_session_id(event: dict[str, Any]) -> str | None:
    if event["type"] == "session_snapshot":
        return event["snapshot"]["id"]
    if event["type"] in ("session_progress", "session_removed"):
        return event["sessionId"]
    return None


class ClientState:
    def __init__(self, on_listener_error: ListenerErrorHandler | None = None) -> None:
        self._session_snapshots: dict[str, dict[str, Any]] = {}
        self._attached_session_ids: set[str] = set()
        self._snapshot_listeners: set[Callable[[dict[str, Any]], None]] = set()
        self._event_listeners: set[Callable[[dict[str, Any]], None]] = set()
        self._session_snapshot_listeners: dict[str, set[Callable[[dict[str, Any]], None]]] = {}
        self._session_event_listeners: dict[str, set[Callable[[dict[str, Any]], None]]] = {}
        self._on_listener_error = on_listener_error
        self._snapshot: dict[str, Any] | None = None

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._snapshot

    def reset(self) -> None:
        self._snapshot = None
        self._session_snapshots.clear()
        self._attached_session_ids.clear()

    def clear_attachments(self) -> None:
        self._attached_session_ids.clear()

    def dispose(self) -> None:
        self.reset()
        self._snapshot_listeners.clear()
        self._event_listeners.clear()
        self._session_snapshot_listeners.clear()
        self._session_event_listeners.clear()

    def get_session_snapshot(self, session_id: str) -> dict[str, Any] | None:
        return self._session_snapshots.get(session_id)

    def is_session_attached(self, session_id: str) -> bool:
        return session_id in self._attached_session_ids

    def forget_session_snapshot(self, session_id: str) -> dict[str, Any] | None:
        return self._session_snapshots.pop(session_id, None)

    def restore_session_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot["id"] not in self._session_snapshots:
            self._session_snapshots[snapshot["id"]] = snapshot

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        self._snapshot_listeners.add(listener)
        return lambda: self._snapshot_listeners.discard(listener)

    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        self._event_listeners.add(listener)
        return lambda: self._event_listeners.discard(listener)

    def subscribe_session(self, session_id: str, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        return _add_mapped_listener(self._session_snapshot_listeners, session_id, listener)

    def on_session_event(self, session_id: str, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        return _add_mapped_listener(self._session_event_listeners, session_id, listener)

    def apply_result(self, result: dict[str, Any]) -> None:
        if result["command"] == "list":
            return
        if result["command"] == "detach":
            session_id = result["sessionId"]
            self._attached_session_ids.discard(session_id)
            snapshot = self._session_snapshots.get(session_id)
            if snapshot is not None:
                self._apply_session_snapshot({**snapshot, "attached": False}, force=True)
            return
        self._apply_session_snapshot(result["session"])

    def apply_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "server_snapshot":
            self.apply_server_snapshot(event["snapshot"])
        if event["type"] == "session_snapshot":
            self._apply_session_snapshot(event["snapshot"])
        if event["type"] == "session_removed":
            self._session_snapshots.pop(event["sessionId"], None)
            self._attached_session_ids.discard(event["sessionId"])
        self._notify(self._event_listeners, event)
        session_id = _get_event_session_id(event)
        if session_id:
            self._notify(self._session_event_listeners.get(session_id), event)

    def apply_server_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self._snapshot is not None and snapshot["revision"] < self._snapshot["revision"]:
            return
        self._snapshot = snapshot
        self._notify(self._snapshot_listeners, snapshot)

    def _apply_session_snapshot(self, snapshot: dict[str, Any], force: bool = False) -> None:
        current = self._session_snapshots.get(snapshot["id"])
        if not force and current is not None and snapshot["revision"] < current["revision"]:
            return
        self._session_snapshots[snapshot["id"]] = snapshot
        if snapshot["attached"]:
            self._attached_session_ids.add(snapshot["id"])
        else:
            self._attached_session_ids.discard(snapshot["id"])
        self._notify(self._session_snapshot_listeners.get(snapshot["id"]), snapshot)

    def _notify(self, listeners: set[Callable[[Any], None]] | None, value: Any) -> None:
        for listener in list(listeners or ()):
            try:
                listener(value)
            except Exception as error:
                self._report_listener_error(error)

    def _report_listener_error(self, error: BaseException) -> None:
        if self._on_listener_error is None:
            return
        with contextlib.suppress(Exception):
            self._on_listener_error(to_error(error))
