"""Server configuration and the injectable session-driver boundary.

Python port of `packages/server/src/types.ts`. `PiSessionRuntime` and
`PiServerService` are the injectable interface a session-driver backend must
implement; see the `pi_server` package docstring for the boundary rationale.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from .errors import PiServerError
from .listener import PiServerListener

MaybeAwaitable: TypeAlias = Any
"""A value or an awaitable of it (TS `MaybePromise<T>`); callers `await` either."""


@dataclass
class PiServerOptions:
    listeners: list[PiServerListener]
    max_frame_length: int | None = None
    handshake_timeout_ms: int | None = None
    server_id: str | None = None
    on_error: Callable[[Exception], None] | None = None


@dataclass
class PromptInput:
    """`Omit<Extract<Command, {command: "prompt"}>, "command" | "sessionId">`."""

    text: str


@dataclass
class SteerInput:
    """`Omit<Extract<Command, {command: "steer"}>, "command" | "sessionId">`."""

    text: str


@dataclass
class CreateSessionOptions:
    """A collision-resistant ID assigned by PiServer. The service must persist this exact ID."""

    id: str
    cwd: str | None = None
    name: str | None = None
    model: dict[str, Any] | None = None
    thinking_level: str | None = None


@dataclass
class SnapshotRuntimeEvent:
    type: Literal["snapshot"] = "snapshot"


@dataclass
class ProgressRuntimeEvent:
    progress: dict[str, Any]
    type: Literal["progress"] = "progress"


@dataclass
class ErrorRuntimeEvent:
    error: PiServerError
    type: Literal["error"] = "error"


PiSessionRuntimeEvent = SnapshotRuntimeEvent | ProgressRuntimeEvent | ErrorRuntimeEvent


class PiSessionRuntime(Protocol):
    """One acquired durable session. Conflicting operations must reject rather than queue."""

    def snapshot(self) -> Awaitable[dict[str, Any]] | dict[str, Any]: ...
    def get_phase(self) -> str: ...
    async def prompt(self, input: PromptInput) -> None: ...
    async def steer(self, input: SteerInput) -> None: ...
    async def abort(self) -> None: ...
    async def set_model(self, model: dict[str, Any]) -> None: ...
    async def set_thinking(self, thinking_level: str) -> None: ...
    def subscribe(self, listener: Callable[[PiSessionRuntimeEvent], None]) -> Callable[[], None]: ...
    async def dispose(self) -> None: ...


class PiServerService(Protocol):
    """Service boundary for durable sessions and exclusively acquired runtimes."""

    async def list_sessions(self) -> list[dict[str, Any]]: ...
    async def list_models(self) -> list[dict[str, Any]]: ...
    async def create_session(self, options: CreateSessionOptions) -> PiSessionRuntime: ...
    async def open_session(self, session_id: str) -> PiSessionRuntime: ...


SessionRuntime = PiSessionRuntime
SessionRuntimeEvent = PiSessionRuntimeEvent
