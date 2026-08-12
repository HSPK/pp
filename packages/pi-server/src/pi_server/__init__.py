"""Python port of `@earendil-works/pi-server`.

`PiServer` drives one or more `PiServerListener`s, performs the wire handshake,
and dispatches client commands to session runtimes. It is a thin RPC front end:
all durable session state and agent execution live behind the
`PiSessionRuntime` / `PiServerService` protocols declared in `types.py`. This
is an intentional, documented boundary between transport and execution.
`pi_server.testing` provides `TestServerService` / `TestSessionRuntime`, an
in-memory fake implementing that boundary, used both by this package's own
tests and by `pi_client`'s integration tests.

`pi_coding_agent.core.agent_session_runtime` supplies the real, `pi_agent`-
backed implementation of that boundary (`AgentSessionRuntime` /
`AgentSessionRuntimeService`), wrapping a live `AgentSession` so `pi_server`
can drive real coding-agent sessions over the wire. `pi_coding_agent` depends
on `pi_server` for the protocol types; `pi_server` itself has no dependency
back on `pi_coding_agent`.

`protocol.py` ports both the pure model/usage/metadata JSON-shaping helpers
from the TS `protocol.ts` and the transcript-message conversions
(`to_protocol_user_message` and friends), since a real adapter now exists to
exercise them.
"""

from __future__ import annotations

from .errors import (
    InternalServerError,
    NotImplementedProtocolError,
    PiServerError,
    SessionBusyError,
    SessionLockedError,
    SessionNotFoundError,
)
from .listener import PiServerListener
from .protocol import (
    UNDEFINED,
    AssistantTranscriptOptions,
    ToolTranscriptOptions,
    UserTranscriptOptions,
    sanitize_protocol_details,
    to_protocol_assistant_message,
    to_protocol_json_value,
    to_protocol_model_metadata,
    to_protocol_tool_result_message,
    to_protocol_usage,
    to_protocol_user_message,
)
from .server import PiServer
from .types import (
    CreateSessionOptions,
    PiServerOptions,
    PiServerService,
    PiSessionRuntime,
    PiSessionRuntimeEvent,
    PromptInput,
    SteerInput,
)

__all__ = [
    "UNDEFINED",
    "AssistantTranscriptOptions",
    "CreateSessionOptions",
    "InternalServerError",
    "NotImplementedProtocolError",
    "PiServer",
    "PiServerError",
    "PiServerListener",
    "PiServerOptions",
    "PiServerService",
    "PiSessionRuntime",
    "PiSessionRuntimeEvent",
    "PromptInput",
    "SessionBusyError",
    "SessionLockedError",
    "SessionNotFoundError",
    "SteerInput",
    "ToolTranscriptOptions",
    "UserTranscriptOptions",
    "sanitize_protocol_details",
    "to_protocol_assistant_message",
    "to_protocol_json_value",
    "to_protocol_model_metadata",
    "to_protocol_tool_result_message",
    "to_protocol_usage",
    "to_protocol_user_message",
]
