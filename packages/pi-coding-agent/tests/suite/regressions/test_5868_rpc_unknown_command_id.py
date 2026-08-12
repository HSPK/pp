"""Python port of `packages/coding-agent/test/suite/regressions/5868-rpc-unknown-command-id.test.ts`.

Not portable: the test drives `runRpcMode`, the legacy stdio JSON-line RPC mode
driver. This port ships only that mode's framing (`modes/rpc/jsonl.py`) and
deliberately omits the driver -- it is superseded by the
`pi_server`/`pi_protocol`/`pi_client` socket stack and depends on the
interactive mode's `ExtensionUIContext` plus `output-guard.ts`'s raw-stdout
takeover, neither of which exists here (see `modes/rpc/__init__.py`'s module
docstring). There is no unknown-command dispatch path to pin.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="legacy stdio RPC mode driver is deliberately not ported (only its framing is)")


def test_preserves_the_request_id_on_unknown_command_errors() -> None:
    raise AssertionError("unreachable")
