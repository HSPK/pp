"""The RPC mode driver: stdin framing, event streaming, rebinding, shutdown.

`dispatcher` covers what each command does; this covers the parts that only
exist once the mode is actually running -- that events reach stdout, that the
loop ends when the host closes stdin, and that a session replacement re-subscribes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from harness import create_harness
from pi_ai.providers.faux import faux_assistant_message
from test_rpc_mode import FakeRuntimeHost

from pi_coding_agent.modes.rpc import rpc_mode as rpc_mode_module
from pi_coding_agent.modes.rpc.rpc_mode import run_rpc_mode


@pytest.fixture
def captured_stdout(monkeypatch) -> list[dict[str, Any]]:
    """Collect protocol lines instead of writing to the process's stdout."""
    lines: list[dict[str, Any]] = []

    async def noop() -> None:
        return None

    monkeypatch.setattr(rpc_mode_module, "take_over_stdout", lambda: None)
    monkeypatch.setattr(rpc_mode_module, "write_raw_stdout", lambda text: lines.append(json.loads(text)))
    monkeypatch.setattr(rpc_mode_module, "flush_raw_stdout", noop)
    monkeypatch.setattr(rpc_mode_module, "wait_for_raw_stdout_backpressure", noop)
    return lines


def feed(lines: list[str]):
    """Stand in for stdin: hand the driver a fixed script, then close."""

    async def read_input(on_line) -> None:
        for line in lines:
            on_line(line)

    return read_input


async def test_the_driver_answers_commands_and_stops_at_end_of_input(tmp_path: Path, captured_stdout) -> None:
    harness = await create_harness(tmp_path)
    host = FakeRuntimeHost(harness.session)
    try:
        exit_code = await run_rpc_mode(
            host,
            feed([json.dumps({"id": "1", "type": "get_state"})]),
        )
    finally:
        harness.cleanup()

    assert exit_code == 0
    responses = [line for line in captured_stdout if line.get("type") == "response"]
    assert responses[0]["id"] == "1"
    assert responses[0]["command"] == "get_state"
    assert host.disposed, "the runtime must be disposed when the host closes stdin"


async def test_session_events_are_streamed_as_they_happen(tmp_path: Path, captured_stdout) -> None:
    harness = await create_harness(tmp_path)
    harness.set_responses([faux_assistant_message("hello there")])
    host = FakeRuntimeHost(harness.session)

    async def read_input(on_line) -> None:
        on_line(json.dumps({"id": "1", "type": "prompt", "message": "hi"}))
        # `prompt` answers from its preflight and runs the turn in the
        # background, so the driver has to stay up long enough to stream it.
        await harness.session.wait_for_idle()

    try:
        await run_rpc_mode(host, read_input)
    finally:
        harness.cleanup()

    types = [line.get("type") for line in captured_stdout]
    assert "message_start" in types
    assert "message_end" in types
    assert any(line.get("command") == "prompt" and line.get("success") for line in captured_stdout)


async def test_a_bad_line_is_answered_without_ending_the_session(tmp_path: Path, captured_stdout) -> None:
    harness = await create_harness(tmp_path)
    try:
        await run_rpc_mode(
            FakeRuntimeHost(harness.session),
            feed(["{oops", json.dumps({"id": "2", "type": "get_state"})]),
        )
    finally:
        harness.cleanup()

    assert captured_stdout[0]["command"] == "parse"
    assert captured_stdout[0]["success"] is False
    assert captured_stdout[1]["id"] == "2"
    assert captured_stdout[1]["success"] is True


async def test_the_driver_binds_the_rpc_ui_context(tmp_path: Path, captured_stdout) -> None:
    """An extension calling `ctx.ui.notify()` in RPC mode must reach the host.

    Without this binding the runner keeps its `NullExtensionUIContext` and every
    extension dialog silently answers "cancelled".
    """
    harness = await create_harness(tmp_path)
    try:
        await run_rpc_mode(FakeRuntimeHost(harness.session), feed([]))
        assert harness.session.extension_runner.has_ui()
        harness.session.extension_runner.get_ui_context().notify("built", "info")
    finally:
        harness.cleanup()

    assert captured_stdout[-1]["type"] == "extension_ui_request"
    assert captured_stdout[-1]["method"] == "notify"
    assert captured_stdout[-1]["message"] == "built"


async def test_extension_errors_are_reported_on_the_wire(tmp_path: Path, captured_stdout) -> None:
    harness = await create_harness(tmp_path)
    try:
        await run_rpc_mode(FakeRuntimeHost(harness.session), feed([]))
        harness.session.extension_runner.emit_error(
            type("E", (), {"extension_path": "/x/ext.py", "event": "session_start", "error": "boom"})()
        )
    finally:
        harness.cleanup()

    errors = [line for line in captured_stdout if line.get("type") == "extension_error"]
    assert errors[0]["extensionPath"] == "/x/ext.py"
    assert errors[0]["error"] == "boom"


async def test_a_reader_that_raises_ends_the_run_instead_of_hanging(tmp_path: Path, captured_stdout) -> None:
    """A crash in either background half must stop the mode, not stall it.

    This hung the whole test suite once: `run_rpc_mode` was called without an
    injected reader, the real stdin reader raised against pytest's stdin stub,
    and nothing set the stop event -- so the mode waited forever on a loop that
    had already died. In production the same shape is a broken pipe.
    """
    harness = await create_harness(tmp_path)

    async def failing_reader(_on_line) -> None:
        raise OSError("stdin went away")

    try:
        with pytest.raises(OSError, match="stdin went away"):
            await asyncio.wait_for(run_rpc_mode(FakeRuntimeHost(harness.session), failing_reader), timeout=10)
    finally:
        harness.cleanup()


async def test_a_pump_crash_also_ends_the_run(tmp_path: Path, captured_stdout, monkeypatch) -> None:
    harness = await create_harness(tmp_path)

    async def exploding_backpressure() -> None:
        raise RuntimeError("stdout is gone")

    monkeypatch.setattr(rpc_mode_module, "wait_for_raw_stdout_backpressure", exploding_backpressure)

    async def read_input(on_line) -> None:
        on_line(json.dumps({"id": "1", "type": "get_state"}))
        await asyncio.Event().wait()

    try:
        with pytest.raises(RuntimeError, match="stdout is gone"):
            await asyncio.wait_for(run_rpc_mode(FakeRuntimeHost(harness.session), read_input), timeout=10)
    finally:
        harness.cleanup()


async def test_the_real_stdin_reader_tolerates_a_stream_without_read1(monkeypatch) -> None:
    """`read1` returns as soon as one syscall has data, but not every binary
    stream provides it -- pytest's own stdin stub does not.
    """
    chunks = [b'{"id": "1", "type": "get_state"}\n', b""]

    class OnlyRead:
        buffer = None

        def read(self, _size: int) -> bytes:
            return chunks.pop(0)

    stub = OnlyRead()
    stub.buffer = stub
    monkeypatch.setattr(rpc_mode_module.sys, "stdin", stub)

    seen: list[str] = []
    await rpc_mode_module._read_stdin_lines(seen.append)
    assert seen == ['{"id": "1", "type": "get_state"}']


async def test_an_extension_shutdown_request_ends_the_loop(tmp_path: Path, captured_stdout) -> None:
    """`ctx.shutdown()` must actually stop the mode, not just set a flag.

    The host has no other way to make the agent exit cleanly once it has
    handed over control. The input script never closes stdin and never returns,
    so `wait_for` below fails if anything other than the shutdown request ends
    the run.
    """
    harness = await create_harness(tmp_path)

    async def read_input(on_line) -> None:
        harness.session.extension_runner.create_context().shutdown()
        on_line(json.dumps({"id": "1", "type": "get_state"}))
        await asyncio.Event().wait()

    try:
        exit_code = await asyncio.wait_for(run_rpc_mode(FakeRuntimeHost(harness.session), read_input), timeout=10)
    finally:
        harness.cleanup()

    assert exit_code == 0
    assert any(line.get("command") == "get_state" for line in captured_stdout)
