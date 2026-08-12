"""Client-side error types.

Python port of `packages/client/src/errors.ts`.
"""

from __future__ import annotations

from typing import Any


class PiServerError(Exception):
    """Wraps a `ProtocolError` (dict with `code`/`message`/`details`) received from the server."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error["message"])
        self.code: str = error["code"]
        self.details: Any = error.get("details")


class PiDisconnectedError(Exception):
    def __init__(self, message: str = "Pi client is disconnected") -> None:
        super().__init__(message)


class PiClientDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Pi client is disposed")


class PiSessionOwnershipError(Exception):
    def __init__(self, session_id: str, message: str) -> None:
        super().__init__(message)
        self.session_id = session_id


class PiSessionDetachedError(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id} is not attached")
        self.session_id = session_id


def to_error(error: BaseException | Any) -> Exception:
    if isinstance(error, Exception):
        return error
    return Exception(str(error))


def to_disconnected_error(error: BaseException | Any) -> PiDisconnectedError:
    cause = to_error(error)
    if isinstance(cause, PiDisconnectedError):
        return cause
    return PiDisconnectedError(str(cause))
