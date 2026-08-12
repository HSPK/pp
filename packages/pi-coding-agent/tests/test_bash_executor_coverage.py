"""Coverage tests for `core/bash_executor.py`.

Covers: large-output temp-file creation, on_chunk streaming, signal
cancellation, exception-path cleanup, and `create_local_bash_operations`.
"""

from __future__ import annotations

import pytest
from pi_ai.utils.abort import AbortSignal
from pi_coding_agent.core.bash_executor import (
    _LocalBashOperations,
    create_local_bash_operations,
    execute_bash_with_operations,
)
from pi_coding_agent.tools.truncate import DEFAULT_MAX_BYTES


class FixedOperations:
    """BashOperations that delivers fixed byte chunks."""

    def __init__(self, chunks: list[bytes], exit_code: int = 0) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.received_signal: AbortSignal | None = None

    async def exec(self, command, cwd, on_data, signal=None, timeout=None, env=None):
        self.received_signal = signal
        for chunk in self.chunks:
            on_data(chunk)
        return self.exit_code


class AbortingOperations:
    """BashOperations that marks the signal as aborted and raises."""

    def __init__(self, signal: AbortSignal) -> None:
        self._signal = signal

    async def exec(self, command, cwd, on_data, signal=None, timeout=None, env=None):
        on_data(b"partial output")
        self._signal.abort()
        raise RuntimeError("process killed")


class RaisingOperations:
    """BashOperations that raises without any abort signal."""

    async def exec(self, command, cwd, on_data, signal=None, timeout=None, env=None):
        on_data(b"data before raise")
        raise ValueError("unexpected failure")


# ---------------------------------------------------------------------------
# create_local_bash_operations
# ---------------------------------------------------------------------------


def test_create_local_bash_operations_returns_operations_object():
    ops = create_local_bash_operations()
    assert hasattr(ops, "exec")


def test_create_local_bash_operations_accepts_shell_path():
    ops = create_local_bash_operations(shell_path="/bin/sh")
    assert isinstance(ops, _LocalBashOperations)
    assert ops.shell_path == "/bin/sh"


# ---------------------------------------------------------------------------
# on_chunk streaming
# ---------------------------------------------------------------------------


async def test_on_chunk_receives_all_emitted_text():
    ops = FixedOperations([b"hello ", b"world"])
    received: list[str] = []

    result = await execute_bash_with_operations("echo", "/", ops, on_chunk=received.append)

    assert "".join(received) == "hello world"
    assert result.output == "hello world"


async def test_on_chunk_is_called_once_per_chunk():
    ops = FixedOperations([b"a", b"b", b"c"])
    received: list[str] = []

    await execute_bash_with_operations("echo", "/", ops, on_chunk=received.append)

    # Each non-empty decoded chunk triggers one on_chunk call.
    assert len(received) == 3
    assert "".join(received) == "abc"


# ---------------------------------------------------------------------------
# Large output: temp-file creation and rolling-buffer eviction
# ---------------------------------------------------------------------------


async def test_large_output_creates_temp_file(tmp_path):
    # Output large enough to exceed DEFAULT_MAX_BYTES triggers temp-file creation.
    big_chunk = b"x" * (DEFAULT_MAX_BYTES + 100)
    ops = FixedOperations([big_chunk])

    result = await execute_bash_with_operations("echo", "/", ops)

    assert result.full_output_path is not None
    import os

    assert os.path.exists(result.full_output_path)
    # Cleanup
    os.remove(result.full_output_path)


async def test_large_output_rolling_buffer_keeps_tail():
    # Two chunks each exceeding DEFAULT_MAX_BYTES so the rolling buffer must
    # evict the first to stay under 2*DEFAULT_MAX_BYTES.
    chunk = b"A" * (DEFAULT_MAX_BYTES + 10)
    ops = FixedOperations([chunk, b"END"])

    result = await execute_bash_with_operations("echo", "/", ops)

    # The output must end with "END" — the tail is preserved.
    assert result.output.endswith("END")
    import os

    if result.full_output_path:
        os.remove(result.full_output_path)


async def test_truncated_flag_is_set_when_output_exceeds_max():
    # A truly huge output (more than 2*DEFAULT_MAX_BYTES) will trigger truncation.
    big = b"B" * (DEFAULT_MAX_BYTES * 2 + 1000)
    ops = FixedOperations([big])

    result = await execute_bash_with_operations("echo", "/", ops)

    assert result.truncated is True
    import os

    if result.full_output_path:
        os.remove(result.full_output_path)


# ---------------------------------------------------------------------------
# Signal / cancellation
# ---------------------------------------------------------------------------


async def test_cancelled_result_when_signal_already_aborted():
    abort = AbortSignal()
    abort.abort()
    ops = FixedOperations([b"some output"])

    result = await execute_bash_with_operations("echo", "/", ops, signal=abort)

    assert result.cancelled is True
    assert result.exit_code is None


async def test_exception_with_aborted_signal_returns_cancelled_result():
    abort = AbortSignal()
    ops = AbortingOperations(abort)

    result = await execute_bash_with_operations("echo", "/", ops, signal=abort)

    assert result.cancelled is True
    assert result.exit_code is None
    assert "partial" in result.output


# ---------------------------------------------------------------------------
# Exception without abort → re-raise
# ---------------------------------------------------------------------------


async def test_exception_without_abort_signal_propagates():
    ops = RaisingOperations()

    with pytest.raises(ValueError, match="unexpected failure"):
        await execute_bash_with_operations("echo", "/", ops)


async def test_exception_without_signal_propagates_after_partial_data():
    ops = RaisingOperations()
    received: list[str] = []

    with pytest.raises(ValueError):
        await execute_bash_with_operations("echo", "/", ops, on_chunk=received.append)


# ---------------------------------------------------------------------------
# BashResult fields
# ---------------------------------------------------------------------------


async def test_result_not_cancelled_by_default():
    ops = FixedOperations([b"ok"])
    result = await execute_bash_with_operations("echo", "/", ops)
    assert result.cancelled is False
    assert result.truncated is False
    assert result.full_output_path is None
