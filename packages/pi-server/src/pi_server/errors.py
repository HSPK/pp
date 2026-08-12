"""Server-side error types.

Python port of `packages/server/src/errors.ts`.
"""

from __future__ import annotations

from typing import Any, Literal

JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]


PiServerOperationErrorCode = Literal["busy", "session_locked", "not_found", "invalid_request", "not_implemented"]

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"
NOT_IMPLEMENTED_MESSAGE = "Operation is not implemented"


class PiServerError(Exception):
    """A service/runtime error that can safely cross the protocol boundary."""

    def __init__(self, code: PiServerOperationErrorCode, message: str, details: JsonValue | None = None) -> None:
        super().__init__(message)
        self.name = "PiServerError"
        self.code = code
        self.message = message
        self.details = details


class SessionBusyError(PiServerError):
    def __init__(self, message: str = "Session is busy", details: JsonValue | None = None) -> None:
        super().__init__("busy", message, details)
        self.name = "SessionBusyError"


class SessionLockedError(PiServerError):
    def __init__(self, message: str = "Session is locked", details: JsonValue | None = None) -> None:
        super().__init__("session_locked", message, details)
        self.name = "SessionLockedError"


class SessionNotFoundError(PiServerError):
    def __init__(self, message: str = "Session was not found", details: JsonValue | None = None) -> None:
        super().__init__("not_found", message, details)
        self.name = "SessionNotFoundError"


class NotImplementedProtocolError(PiServerError):
    """`NotImplementedError` in TS; renamed to avoid shadowing Python's builtin."""

    def __init__(self) -> None:
        super().__init__("not_implemented", NOT_IMPLEMENTED_MESSAGE)
        self.name = "NotImplementedError"


class InternalServerError(Exception):
    """An unsafe failure whose cause is retained for reporting but never serialized."""

    def __init__(self, cause: BaseException | None) -> None:
        super().__init__(INTERNAL_SERVER_ERROR_MESSAGE)
        self.name = "InternalServerError"
        self.__cause__ = cause
