"""In-memory `PiServerService` / `PiSessionRuntime` fake for tests.

Python port of `packages/server/src/testing/service.ts`. This fake is the
concrete implementation of the `PiSessionRuntime` / `PiServerService`
injectable boundary (see the `pi_server` package docstring) used by this
package's own tests and by `pi_client`'s client/server integration test. It
never touches a real `pi_agent` session.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from ..errors import PiServerError
from ..types import (
    CreateSessionOptions,
    ErrorRuntimeEvent,
    PiSessionRuntime,
    ProgressRuntimeEvent,
    PromptInput,
    SnapshotRuntimeEvent,
)

TEST_MODEL: dict[str, Any] = {
    "provider": "test",
    "id": "small",
    "name": "Test Small",
    "api": "test-api",
    "reasoning": True,
    "input": ["text", "image"],
    "contextWindow": 16_000,
    "maxTokens": 2_000,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "supportedThinkingLevels": ["off", "medium", "high"],
    "authenticated": True,
}


class Deferred:
    """A resolvable `asyncio.Future`, mirroring the TS testing `Deferred<T>` helper."""

    def __init__(self) -> None:
        self.future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()

    def resolve(self, value: Any = None) -> None:
        if not self.future.done():
            self.future.set_result(value)


class _StoredSession:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot


class _PendingPrompt:
    def __init__(self, input: PromptInput, done: Deferred) -> None:
        self.input = input
        self.done = done


class TestSessionRuntime:
    """`PiSessionRuntime` fake backed by an in-memory `_StoredSession`."""

    __test__ = False  # not a pytest test class despite the name matching TS `TestSessionRuntime`

    def __init__(self, stored: _StoredSession, on_dispose: Any) -> None:
        self.disposed = Deferred()
        self.dispose_count = 0
        self.steers: list[PromptInput] = []
        self._stored = stored
        self._on_dispose = on_dispose
        self._listeners: set[Any] = set()
        self._pending_prompt: _PendingPrompt | None = None

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._stored.snapshot)

    def get_phase(self) -> str:
        return self._stored.snapshot["phase"]

    async def prompt(self, input: PromptInput) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "A prompt is already running")
        done = Deferred()
        self._pending_prompt = _PendingPrompt(input, done)
        revision = self._stored.snapshot["revision"]
        self._update(
            {
                "phase": "turn",
                "transcript": [
                    *self._stored.snapshot["transcript"],
                    {
                        "id": f"user-{revision + 1}",
                        "role": "user",
                        "content": [{"type": "text", "text": input.text}],
                        "timestamp": revision + 1,
                    },
                ],
            }
        )
        outcome = await done.future
        revision = self._stored.snapshot["revision"]
        if outcome == "complete":
            assistant = {
                "id": f"assistant-{revision + 1}",
                "role": "assistant",
                "content": [{"type": "text", "text": f"reply:{input.text}"}],
                "status": "complete",
                "model": self._stored.snapshot["model"],
                "stopReason": "stop",
                "timestamp": revision + 1,
            }
        else:
            assistant = {
                "id": f"assistant-{revision + 1}",
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "status": "aborted",
                "model": self._stored.snapshot["model"],
                "stopReason": "aborted",
                "timestamp": revision + 1,
            }
        self._update({"phase": "idle", "transcript": [*self._stored.snapshot["transcript"], assistant]})
        self._pending_prompt = None

    async def steer(self, input: PromptInput) -> None:
        if self.get_phase() == "idle":
            raise PiServerError("busy", "There is no active prompt to steer")
        self.steers.append(input)
        revision = self._stored.snapshot["revision"]
        self._update(
            {
                "queuedSteerCount": self._stored.snapshot["queuedSteerCount"] + 1,
                "queuedSteer": [
                    *self._stored.snapshot["queuedSteer"],
                    {
                        "id": f"steer-{revision + 1}",
                        "role": "user",
                        "content": [{"type": "text", "text": input.text}],
                        "timestamp": revision + 1,
                    },
                ],
            }
        )

    async def abort(self) -> None:
        if self._pending_prompt is None:
            raise PiServerError("busy", "There is no active prompt to abort")
        self._pending_prompt.done.resolve("aborted")

    async def set_model(self, model: dict[str, Any]) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "Session is busy")
        self._update({"model": model})

    async def set_thinking(self, thinking_level: str) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "Session is busy")
        self._update({"thinkingLevel": thinking_level})

    def subscribe(self, listener: Any) -> Any:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    async def dispose(self) -> None:
        self.dispose_count += 1
        self._on_dispose()
        self.disposed.resolve(None)

    def set_phase(self, phase: str) -> None:
        self._stored.snapshot = {**self._stored.snapshot, "phase": phase}

    def finish_prompt(self) -> None:
        if self._pending_prompt is None:
            raise RuntimeError("No prompt is pending")
        self._pending_prompt.done.resolve("complete")

    def emit_progress(self, progress: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(ProgressRuntimeEvent(progress=progress))

    def emit_error(self, error: PiServerError) -> None:
        for listener in list(self._listeners):
            listener(ErrorRuntimeEvent(error=error))

    def emit_snapshot(self) -> None:
        for listener in list(self._listeners):
            listener(SnapshotRuntimeEvent())

    def _update(self, updates: dict[str, Any]) -> None:
        self._stored.snapshot = {
            **self._stored.snapshot,
            **updates,
            "revision": self._stored.snapshot["revision"] + 1,
            "updatedAt": self._stored.snapshot["updatedAt"] + 1,
        }
        self.emit_snapshot()


class _ListDelay:
    def __init__(self) -> None:
        self.entered = Deferred()
        self.release = Deferred()


class TestServerService:
    """`PiServerService` fake backed by an in-memory session store."""

    __test__ = False  # not a pytest test class despite the name matching TS `TestServerService`

    def __init__(self) -> None:
        self.sessions: dict[str, _StoredSession] = {}
        self.runtimes: dict[str, list[TestSessionRuntime]] = {}
        self.locked: set[str] = set()
        self.last_created_id: str | None = None
        self._next_list_delay: _ListDelay | None = None

    async def list_sessions(self) -> list[dict[str, Any]]:
        delay = self._next_list_delay
        if delay is not None:
            self._next_list_delay = None
            delay.entered.resolve(None)
            await delay.release.future
        return [
            {
                "id": stored.snapshot["id"],
                "createdAt": stored.snapshot["createdAt"],
                "updatedAt": stored.snapshot["updatedAt"],
                "sessionName": stored.snapshot.get("name"),
                "cwd": stored.snapshot["cwd"],
            }
            for stored in self.sessions.values()
        ]

    async def list_models(self) -> list[dict[str, Any]]:
        return [TEST_MODEL]

    async def create_session(self, options: CreateSessionOptions) -> PiSessionRuntime:
        self.last_created_id = options.id
        if options.id in self.sessions:
            raise PiServerError("session_locked", "Session already exists")
        self.seed(options.id, options.name, options.cwd, options.model, options.thinking_level)
        return self._acquire(options.id)

    async def open_session(self, session_id: str) -> PiSessionRuntime:
        if session_id not in self.sessions:
            raise PiServerError("not_found", f"Unknown session: {session_id}")
        if session_id in self.locked:
            raise PiServerError("session_locked", f"Session is locked: {session_id}")
        return self._acquire(session_id)

    def seed(
        self,
        id: str = "session-1",
        name: str | None = None,
        cwd: str | None = None,
        model: dict[str, Any] | None = None,
        thinking_level: str | None = None,
    ) -> None:
        self.sessions[id] = _StoredSession(
            {
                "id": id,
                "name": name if name is not None else f"Session {id}",
                "cwd": cwd if cwd is not None else "/tmp/pi-server-conformance",
                "createdAt": 1,
                "updatedAt": 1,
                "phase": "idle",
                "model": model if model is not None else {"provider": TEST_MODEL["provider"], "id": TEST_MODEL["id"]},
                "thinkingLevel": thinking_level if thinking_level is not None else "off",
                "attached": False,
                "locked": False,
                "revision": 0,
                "transcript": [],
                "queuedSteer": [],
                "queuedSteerCount": 0,
            }
        )

    def delay_next_list(self) -> _ListDelay:
        delay = _ListDelay()
        self._next_list_delay = delay
        return delay

    def latest_runtime(self, id: str) -> TestSessionRuntime:
        runtimes = self.runtimes.get(id)
        if not runtimes:
            raise RuntimeError(f"No runtime for {id}")
        return runtimes[-1]

    def _acquire(self, id: str) -> TestSessionRuntime:
        stored = self.sessions.get(id)
        if stored is None:
            raise RuntimeError(f"Unknown session: {id}")
        self.locked.add(id)
        runtime = TestSessionRuntime(stored, lambda: self.locked.discard(id))
        self.runtimes.setdefault(id, []).append(runtime)
        return runtime
