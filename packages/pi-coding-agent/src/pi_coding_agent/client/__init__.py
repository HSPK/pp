"""Client-side helpers for driving a remote coding-agent session.

Ported from `packages/coding-agent/src/client/` in the TypeScript "pi"
monorepo: `transcript.ts` (a pure snapshot+progress reducer) and
`remote-session.ts` (a reactive single-session wrapper over
`pi_client.PiClient`). Both are extension-free and portable as-is; see each
module's docstring for details.
"""

from pi_coding_agent.client.remote_session import (
    CreateRemoteSessionOptions,
    RemoteSession,
    RemoteSessionDisposedError,
    RemoteSessionLifecycle,
    RemoteSessionOperation,
    RemoteSessionOptions,
    RemoteSessionState,
)
from pi_coding_agent.client.transcript import (
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)

__all__ = [
    "CreateRemoteSessionOptions",
    "RemoteSession",
    "RemoteSessionDisposedError",
    "RemoteSessionLifecycle",
    "RemoteSessionOperation",
    "RemoteSessionOptions",
    "RemoteSessionState",
    "TranscriptState",
    "apply_transcript_progress",
    "apply_transcript_snapshot",
    "create_transcript_state",
    "select_transcript",
]
