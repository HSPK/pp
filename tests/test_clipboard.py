"""Python port of `packages/coding-agent/test/clipboard.test.ts`.

The TypeScript original mocks `utils/clipboard-native.ts`, the native Node
addon backed by the `clipboard-rs` crate. This port has no such addon (see the
module docstring of `pi_coding_agent.utils.clipboard`), so every assertion that
pins "the native addon succeeded" is expressed against the platform command
line tool the port uses in its place (`pbcopy`/`pbpaste` on macOS). The
fallback ordering, OSC 52 emission and error behaviour are asserted unchanged.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping

import pytest

from pi_coding_agent.utils import clipboard as clipboard_module
from pi_coding_agent.utils.clipboard import copy_to_clipboard, read_clipboard_text

LOCAL_ENV: Mapping[str, str] = {}
REMOTE_ENV: Mapping[str, str] = {"SSH_CONNECTION": "client server"}


def osc52_writes(captured: str) -> list[str]:
    return [chunk for chunk in captured.split("\x07") if chunk.startswith("\x1b]52;c;")]


@pytest.fixture
def read_calls(monkeypatch: pytest.MonkeyPatch) -> tuple[list[list[str]], dict[str, object]]:
    """Record `_read_command` invocations and serve scripted results."""
    calls: list[list[str]] = []
    results: dict[str, object] = {}

    def fake_read(argv: list[str]) -> str | None:
        calls.append(list(argv))
        outcome = results.get(argv[0])
        if isinstance(outcome, Exception):
            return None
        return outcome  # type: ignore[return-value]

    monkeypatch.setattr(clipboard_module, "_read_command", fake_read)
    return calls, results


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple[list[str], str]], dict[str, object]]:
    """Record `_run_with_input` invocations and serve scripted failures."""
    calls: list[tuple[list[str], str]] = []
    failures: dict[str, object] = {}

    def fake_run(argv: list[str], text: str) -> None:
        calls.append((list(argv), text))
        if argv[0] in failures:
            raise subprocess.SubprocessError(f"{argv[0]} failed")

    monkeypatch.setattr(clipboard_module, "_run_with_input", fake_run)
    return calls, failures


# ---------------------------------------------------------------------------
# read_clipboard_text
# ---------------------------------------------------------------------------


async def test_returns_clipboard_text(
    monkeypatch: pytest.MonkeyPatch, read_calls: tuple[list[list[str]], dict[str, object]]
) -> None:
    calls, results = read_calls
    monkeypatch.setattr("sys.platform", "darwin")
    results["pbpaste"] = "clipboard text"

    assert await read_clipboard_text(LOCAL_ENV) == "clipboard text"
    assert calls == [["pbpaste"]]


async def test_reads_the_wayland_clipboard_before_the_stale_x11_clipboard(
    monkeypatch: pytest.MonkeyPatch, read_calls: tuple[list[list[str]], dict[str, object]]
) -> None:
    # Regression test for #7248.
    calls, results = read_calls
    monkeypatch.setattr("sys.platform", "linux")
    results["wl-paste"] = "Wayland text"
    results["xclip"] = "stale X11 text"

    env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    assert await read_clipboard_text(env) == "Wayland text"
    assert calls == [["wl-paste", "--no-newline", "--type", "text"]]


async def test_does_not_fall_back_to_stale_x11_text_when_wayland_clipboard_is_empty(
    monkeypatch: pytest.MonkeyPatch, read_calls: tuple[list[list[str]], dict[str, object]]
) -> None:
    calls, results = read_calls
    monkeypatch.setattr("sys.platform", "linux")
    results["wl-paste"] = ""
    results["xclip"] = "stale X11 text"

    env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    assert await read_clipboard_text(env) is None
    assert calls == [["wl-paste", "--no-newline", "--type", "text"]]


async def test_falls_back_to_x11_when_wl_paste_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, read_calls: tuple[list[list[str]], dict[str, object]]
) -> None:
    calls, results = read_calls
    monkeypatch.setattr("sys.platform", "linux")
    results["wl-paste"] = None
    results["xclip"] = "X11 fallback text"

    env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    assert await read_clipboard_text(env) == "X11 fallback text"
    assert calls[0] == ["wl-paste", "--no-newline", "--type", "text"]
    assert calls[1] == ["xclip", "-selection", "clipboard", "-o"]


async def test_returns_none_for_empty_or_unavailable_clipboard_text(
    monkeypatch: pytest.MonkeyPatch, read_calls: tuple[list[list[str]], dict[str, object]]
) -> None:
    _calls, results = read_calls
    monkeypatch.setattr("sys.platform", "darwin")

    results["pbpaste"] = ""
    assert await read_clipboard_text(LOCAL_ENV) is None

    results["pbpaste"] = None
    assert await read_clipboard_text(LOCAL_ENV) is None


# ---------------------------------------------------------------------------
# copy_to_clipboard
# ---------------------------------------------------------------------------


async def test_local_platform_tool_success_skips_osc52(
    monkeypatch: pytest.MonkeyPatch,
    run_calls: tuple[list[tuple[list[str], str]], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, _failures = run_calls
    monkeypatch.setattr("sys.platform", "darwin")

    await copy_to_clipboard("hello", LOCAL_ENV)

    assert calls == [(["pbcopy"], "hello")]
    assert osc52_writes(capsys.readouterr().out) == []


async def test_remote_success_emits_osc52_after_the_native_write(
    monkeypatch: pytest.MonkeyPatch,
    run_calls: tuple[list[tuple[list[str], str]], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, _failures = run_calls
    monkeypatch.setattr("sys.platform", "darwin")
    seen_before_native: list[str] = []

    original_emit = clipboard_module.emit_osc52

    def tracking_emit(text: str, *, stream: object | None = None) -> bool:
        seen_before_native.append(calls[0][0][0] if calls else "")
        return original_emit(text, stream=stream)

    monkeypatch.setattr(clipboard_module, "emit_osc52", tracking_emit)

    await copy_to_clipboard("hello", REMOTE_ENV)

    assert calls == [(["pbcopy"], "hello")]
    assert seen_before_native == ["pbcopy"]
    assert len(osc52_writes(capsys.readouterr().out)) == 1


async def test_shell_fallback_success_skips_osc52(
    monkeypatch: pytest.MonkeyPatch,
    run_calls: tuple[list[tuple[list[str], str]], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # TypeScript reaches this case by making the native addon reject and the
    # shell tool succeed, and also asserts `execSync` was *not* called in the
    # native-success case above. With no native addon there is only one layer,
    # so the two TypeScript cases collapse into the same observable behaviour:
    # `pbcopy` runs, OSC 52 does not.
    calls, _failures = run_calls
    monkeypatch.setattr("sys.platform", "darwin")

    await copy_to_clipboard("hello", LOCAL_ENV)

    assert calls == [(["pbcopy"], "hello")]
    assert osc52_writes(capsys.readouterr().out) == []


async def test_uses_osc52_fallback_when_shell_tools_fail(
    monkeypatch: pytest.MonkeyPatch,
    run_calls: tuple[list[tuple[list[str], str]], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _calls, failures = run_calls
    monkeypatch.setattr("sys.platform", "darwin")
    failures["pbcopy"] = True

    await copy_to_clipboard("hello", LOCAL_ENV)

    assert len(osc52_writes(capsys.readouterr().out)) == 1


async def test_does_not_emit_oversized_osc52_payloads(
    monkeypatch: pytest.MonkeyPatch,
    run_calls: tuple[list[tuple[list[str], str]], dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _calls, failures = run_calls
    monkeypatch.setattr("sys.platform", "darwin")
    failures["pbcopy"] = True

    with pytest.raises(RuntimeError, match="Failed to copy to clipboard"):
        await copy_to_clipboard("x" * 80_000, LOCAL_ENV)

    assert osc52_writes(capsys.readouterr().out) == []


# ---------------------------------------------------------------------------
# The read/run command seams themselves
#
# The cases above patch `_read_command`/`_run_with_input`, so they cannot see
# the options TypeScript pins on `execFileSync`: `{ encoding: "utf8",
# maxBuffer: 50 * 1024 * 1024, timeout: 5000 }`. These drive the real
# functions with only `subprocess.run` replaced, so the timeout and the buffer
# cap are asserted rather than assumed.
# ---------------------------------------------------------------------------


async def test_read_command_applies_the_five_second_timeout_and_decodes_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.update(kwargs)
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, b"clipboard text", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert clipboard_module._read_command(["pbpaste"]) == "clipboard text"
    assert seen["argv"] == ["pbpaste"]
    assert seen["timeout"] == 5.0
    assert seen["capture_output"] is True
    assert seen["check"] is True


async def test_read_command_caps_output_at_fifty_megabytes(monkeypatch: pytest.MonkeyPatch) -> None:
    oversized = b"y" * (50 * 1024 * 1024 + 10)

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, oversized, b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = clipboard_module._read_command(["pbpaste"])
    assert result is not None
    assert len(result) == 50 * 1024 * 1024


async def test_read_command_returns_none_when_the_tool_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert clipboard_module._read_command(["wl-paste"]) is None


async def test_run_with_input_applies_the_five_second_timeout_and_feeds_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.update(kwargs)
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    clipboard_module._run_with_input(["pbcopy"], "hello")

    assert seen["argv"] == ["pbcopy"]
    assert seen["input"] == b"hello"
    assert seen["timeout"] == 5.0
    assert seen["check"] is True
    # TS pins `stdio: ["pipe", "ignore", "ignore"]` on the execSync call:
    # stdin piped (the `input` option above), stdout/stderr discarded.
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.DEVNULL
