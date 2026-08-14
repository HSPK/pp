"""Python port of `packages/coding-agent/test/suite/regressions/5868-rpc-unknown-command-id.test.ts`.

Regression for https://github.com/earendil-works/pi/issues/5868: an unknown
command answered without echoing the request's `id`, so a host correlating
responses to requests could not match the failure to what it sent, and would
keep waiting for an answer that had already arrived.

The TypeScript test mocks `output-guard` and `jsonl` to drive `runRpcMode`
without a real process. This port does the same through `run_rpc_mode`'s
`read_input` parameter, which exists for exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness import create_harness
from pi_coding_agent.modes.rpc import rpc_mode as rpc_mode_module
from pi_coding_agent.modes.rpc.rpc_mode import run_rpc_mode


class _CancellingRuntimeHost:
    """Every replacement is refused, matching the TypeScript stubs."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.disposed = False

    @property
    def session(self) -> Any:
        return self._session

    async def new_session(self, parent_session: str | None = None) -> dict[str, Any]:
        return {"cancelled": True}

    async def switch_session(self, session_path: str, cwd_override: str | None = None) -> dict[str, Any]:
        return {"cancelled": True}

    async def fork(self, entry_id: str, position: str = "before") -> dict[str, Any]:
        return {"cancelled": True, "selected_text": ""}

    async def dispose(self) -> None:
        self.disposed = True

    def set_rebind_session(self, _rebind: Any) -> None:
        pass


async def test_preserves_the_request_id_on_unknown_command_errors(tmp_path: Path, monkeypatch) -> None:
    lines: list[dict[str, Any]] = []

    async def noop() -> None:
        return None

    monkeypatch.setattr(rpc_mode_module, "take_over_stdout", lambda: None)
    monkeypatch.setattr(rpc_mode_module, "write_raw_stdout", lambda text: lines.append(json.loads(text)))
    monkeypatch.setattr(rpc_mode_module, "flush_raw_stdout", noop)
    monkeypatch.setattr(rpc_mode_module, "wait_for_raw_stdout_backpressure", noop)

    async def read_input(on_line) -> None:
        on_line(json.dumps({"id": "test", "type": "foobar"}))

    harness = await create_harness(tmp_path)
    try:
        await run_rpc_mode(_CancellingRuntimeHost(harness.session), read_input)
    finally:
        harness.cleanup()

    assert {
        "id": "test",
        "type": "response",
        "command": "foobar",
        "success": False,
        "error": "Unknown command: foobar",
    } in lines
