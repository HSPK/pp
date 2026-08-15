"""Python port of `packages/coding-agent/test/rpc-prompt-response-semantics.test.ts`.

Not portable: every case calls `runRpcMode` from `src/modes/rpc/rpc-mode.ts`,
the legacy line-framed stdio RPC mode, which this port does not have (see
`tests/test_ca_rpc.py`, which explains the same omission for `rpc.test.ts`).
There is no `modes/rpc/` package under `src/pi_coding_agent/`, so there is no
loop to feed a request line into and no output line stream to assert on.

The three TypeScript cases pin the *response cardinality* of the `prompt`
request -- exactly one response per request, whatever the agent does:

1. "emits one failure response when prompt preflight rejects"
2. "emits one success response when prompt preflight succeeds"
3. "emits one success response when prompt is queued during streaming"

All three assert on lines written through `writeRawStdout` by `runRpcMode`. The
underlying `AgentSession.prompt` preflight and queueing behavior they lean on
*is* ported and is covered by the agent-session tests; only the RPC framing on
top of it is missing.
"""

from __future__ import annotations

import pytest

_NO_RPC_MODE = (
    "src/modes/rpc/rpc-mode.ts (legacy line-framed stdio RPC mode) is not ported; this "
    "port only has modes/rpc/jsonl.py, the framing helper. The AgentSession.prompt "
    "preflight and queueing behavior underneath is covered by the agent-session tests."
)


@pytest.mark.skip(reason=_NO_RPC_MODE)
def test_emits_one_failure_response_when_prompt_preflight_rejects() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_RPC_MODE)
def test_emits_one_success_response_when_prompt_preflight_succeeds() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_RPC_MODE)
def test_emits_one_success_response_when_prompt_is_queued_during_streaming() -> None:
    raise AssertionError("unreachable")
