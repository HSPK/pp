"""Python port of `packages/coding-agent/test/clipboard-image.test.ts`.

The TypeScript original mocks the native `clipboard-native.ts` addon. This port
has no addon, so `read_native_clipboard_image` is a hook that normally returns
`None`; the tests monkeypatch it exactly where TypeScript mocks
`clipboard.hasImage` / `clipboard.getImageBinary`, so each assertion keeps its
original intent.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pi_coding_agent.utils import clipboard_image as clipboard_image_module
from pi_coding_agent.utils.clipboard_image import ClipboardImage, CommandResult, read_clipboard_image


def spawn_ok(stdout: bytes) -> CommandResult:
    return CommandResult(ok=True, stdout=stdout)


def spawn_error() -> CommandResult:
    return CommandResult(ok=False, stdout=b"")


@pytest.fixture
def forbid_native(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail() -> ClipboardImage | None:
        raise AssertionError("the native clipboard must not be consulted here")

    monkeypatch.setattr(clipboard_image_module, "read_native_clipboard_image", fail)


def install_run_command(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str, list[str]], CommandResult],
) -> list[tuple[str, list[str], dict[str, object]]]:
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    def fake_run(command: str, args: list[str], **kwargs: object) -> CommandResult:
        calls.append((command, list(args), dict(kwargs)))
        return handler(command, list(args))

    monkeypatch.setattr(clipboard_image_module, "run_command", fake_run)
    return calls


async def test_wayland_uses_wl_paste_and_never_calls_the_native_clipboard(
    monkeypatch: pytest.MonkeyPatch, forbid_native: None
) -> None:
    def handler(command: str, args: list[str]) -> CommandResult:
        if command == "wl-paste" and args[0] == "--list-types":
            return spawn_ok(b"text/plain\nimage/png\n")
        if command == "wl-paste" and args[0] == "--type":
            return spawn_ok(bytes([1, 2, 3]))
        raise AssertionError(f"Unexpected run_command call: {command} {' '.join(args)}")

    install_run_command(monkeypatch, handler)

    result = await read_clipboard_image(platform="linux", env={"WAYLAND_DISPLAY": "1"})
    assert result is not None
    assert result.mime_type == "image/png"
    assert list(result.bytes) == [1, 2, 3]


async def test_wayland_falls_back_to_xclip_when_wl_paste_is_missing(
    monkeypatch: pytest.MonkeyPatch, forbid_native: None
) -> None:
    def handler(command: str, args: list[str]) -> CommandResult:
        if command == "wl-paste":
            return spawn_error()
        if command == "xclip" and "TARGETS" in args:
            return spawn_ok(b"image/png\n")
        if command == "xclip" and "image/png" in args:
            return spawn_ok(bytes([9, 8]))
        return spawn_ok(b"")

    install_run_command(monkeypatch, handler)

    result = await read_clipboard_image(platform="linux", env={"XDG_SESSION_TYPE": "wayland"})
    assert result is not None
    assert result.mime_type == "image/png"
    assert list(result.bytes) == [9, 8]


async def test_wsl_passes_the_powershell_path_directly(monkeypatch: pytest.MonkeyPatch, forbid_native: None) -> None:
    tmp_files: list[str] = []

    def handler(command: str, args: list[str]) -> CommandResult:
        if command in ("wl-paste", "xclip"):
            return spawn_ok(b"")
        if command == "wslpath":
            tmp_files.append(args[1])
            return spawn_ok(b"C:\\Users\\O'Hare\\clip.png\n")
        if command == "powershell.exe":
            assert args[2].find("$path = 'C:\\Users\\O''Hare\\clip.png'") != -1
            assert tmp_files, "wslpath must be called before powershell.exe"
            Path(tmp_files[0]).write_bytes(bytes([4, 5, 6]))
            return spawn_ok(b"ok\n")
        raise AssertionError(f"Unexpected run_command call: {command} {' '.join(args)}")

    calls = install_run_command(monkeypatch, handler)

    result = await read_clipboard_image(platform="linux", env={"WSL_DISTRO_NAME": "Ubuntu"})
    assert result is not None
    assert result.mime_type == "image/png"
    assert list(result.bytes) == [4, 5, 6]

    # The Windows path is passed inline; no custom environment variable is used.
    powershell_calls = [call for call in calls if call[0] == "powershell.exe"]
    assert len(powershell_calls) == 1
    assert powershell_calls[0][2].get("env") is None


async def test_non_wayland_uses_the_native_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(command: str, args: list[str]) -> CommandResult:
        raise AssertionError("no command may run when the native clipboard returns an image")

    install_run_command(monkeypatch, handler)

    async def native() -> ClipboardImage | None:
        return ClipboardImage(bytes=bytes([7]), mime_type="image/png")

    monkeypatch.setattr(clipboard_image_module, "read_native_clipboard_image", native)

    result = await read_clipboard_image(platform="linux", env={})
    assert result is not None
    assert result.mime_type == "image/png"
    assert list(result.bytes) == [7]


async def test_non_wayland_falls_back_to_xclip_when_the_native_clipboard_has_no_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(command: str, args: list[str]) -> CommandResult:
        if command == "xclip" and "TARGETS" in args:
            return spawn_ok(b"image/png\n")
        if command == "xclip" and "image/png" in args:
            return spawn_ok(bytes([8, 9]))
        raise AssertionError(f"Unexpected run_command call: {command} {' '.join(args)}")

    install_run_command(monkeypatch, handler)

    async def native() -> ClipboardImage | None:
        return None

    monkeypatch.setattr(clipboard_image_module, "read_native_clipboard_image", native)

    result = await read_clipboard_image(platform="linux", env={})
    assert result is not None
    assert result.mime_type == "image/png"
    assert list(result.bytes) == [8, 9]
