"""Python port of `packages/coding-agent/test/suite/regressions/5208-late-bash-output.test.ts`.

The TypeScript test injects a fake `BashOperations`. This port deliberately
drops the `BashOperations`/`BashSpawnHook` injection points (see `bash.py`'s
module docstring), so the fake process is installed by patching the module's
`_exec_local` -- the same seam the real tool uses to stream output.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

import pytest
from pi_coding_agent.tools import bash as bash_module
from pi_coding_agent.tools.bash import create_bash_tool


def _get_text_output(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")


async def test_ignores_output_callbacks_after_bash_operations_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    late_delivered = asyncio.Event()

    async def fake_exec(
        _command: str,
        _cwd: str,
        on_data: Callable[[bytes], None],
        _signal,
        _timeout,
        _env,
        _shell_path=None,
    ) -> int:
        on_data(b"before\n")
        loop = asyncio.get_running_loop()

        def deliver_late() -> None:
            on_data(b"late\n")
            late_delivered.set()

        loop.call_soon(deliver_late)
        return 0

    monkeypatch.setattr(bash_module, "_exec_local", fake_exec)
    bash = create_bash_tool(os.getcwd())

    callback_errors: list[BaseException] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context.get("exception")))

    updates: list[object] = []
    try:
        result = await bash.execute(
            "test-call-late-output", {"command": "late-output"}, None, lambda update: updates.append(update)
        )
        update_count_at_resolution = len(updates)
        # TypeScript waits a real 20ms for its `setTimeout(..., 0)`. Waiting on
        # the delivery itself is both faster and immune to a loaded host, and
        # unlike a fixed sleep it cannot pass vacuously by asserting before the
        # late callback ever ran.
        await asyncio.wait_for(late_delivered.wait(), timeout=5)
    finally:
        loop.set_exception_handler(previous_handler)

    assert _get_text_output(result).strip() == "before"
    # The late callback must be dropped, not appended to the finished accumulator.
    assert callback_errors == []
    assert len(updates) == update_count_at_resolution
