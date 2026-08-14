"""Terminal capability detection and raw-mode driver.

Python port of `packages/tui/src/terminal.ts`.

The TypeScript file reads/writes `process.stdin`/`process.stdout` directly.
To keep this module testable without a real TTY, all interaction with the
operating system (raw mode, stdin bytes, terminal size, resize signals) is
routed through a small `TerminalIo` bundle of callables. `real_terminal_io()`
builds the bundle backed by the real terminal; tests build their own bundle
backed by an in-memory fake (see `pi_tui.testing`) so no test
ever touches a real TTY or blocks on stdin.

Windows console support is out of scope for this port (the native
`win32-console-mode` helper that enables `ENABLE_VIRTUAL_TERMINAL_INPUT` is a
compiled Node addon with no Python equivalent here). On a platform without
`termios`/`tty` (Windows), `enter_raw_mode` falls back to cooked mode: input
still works, but Shift+Tab and similar modified keys may not be
distinguishable from their unmodified form.

Timers (the Kitty protocol negotiation fragment timeout, and the terminal
progress-indicator keepalive) use `asyncio` rather than `threading.Timer`
because their callbacks call back into the input/render pipeline, which is
not thread-safe. `ProcessTerminal` therefore requires a running asyncio event
loop for `start()`, `stop()`, and `set_progress()`.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import math
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pi_tui.keys import set_kitty_protocol_active
from pi_tui.stdin_buffer import StdinBuffer
from pi_tui.tasks import spawn

with contextlib.suppress(ImportError):
    import termios
    import tty

_HAS_TERMIOS = "termios" in globals() and "tty" in globals()

TERMINAL_PROGRESS_KEEPALIVE_S = 1.0
TERMINAL_PROGRESS_ACTIVE_SEQUENCE = "\x1b]9;4;3\x07"
TERMINAL_PROGRESS_CLEAR_SEQUENCE = "\x1b]9;4;0\x07"
NATIVE_SHIFT_ENTER_SEQUENCE = "\x1b[13;2u"
DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS = 7
KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_S = 0.15
KITTY_KEYBOARD_PROTOCOL_QUERY = f"\x1b[>{DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS}u\x1b[?u\x1b[c"

_KITTY_FLAGS_RE = re.compile(r"^\x1b\[\?(\d+)u$")
_DEVICE_ATTRIBUTES_RE = re.compile(r"^\x1b\[\?[\d;]*c$")
_NEGOTIATION_PREFIX_RE = re.compile(r"^\x1b\[\?[\d;]*$")


@dataclass
class KittyFlagsNegotiation:
    flags: int
    type: Literal["kitty-flags"] = "kitty-flags"


@dataclass
class DeviceAttributesNegotiation:
    type: Literal["device-attributes"] = "device-attributes"


KeyboardProtocolNegotiationSequence = KittyFlagsNegotiation | DeviceAttributesNegotiation


def parse_keyboard_protocol_negotiation_sequence(sequence: str) -> KeyboardProtocolNegotiationSequence | None:
    kitty_flags = _KITTY_FLAGS_RE.match(sequence)
    if kitty_flags:
        return KittyFlagsNegotiation(flags=int(kitty_flags.group(1)))
    if _DEVICE_ATTRIBUTES_RE.match(sequence):
        return DeviceAttributesNegotiation()
    return None


def _is_keyboard_protocol_negotiation_sequence_prefix(sequence: str) -> bool:
    return sequence == "\x1b[" or bool(_NEGOTIATION_PREFIX_RE.match(sequence))


def is_apple_terminal_session() -> bool:
    return sys.platform == "darwin" and os.environ.get("TERM_PROGRAM") == "Apple_Terminal"


def normalize_native_shift_enter_input(
    data: str, should_detect_native_shift_enter: bool, is_shift_pressed: bool
) -> str:
    if should_detect_native_shift_enter and data == "\r" and is_shift_pressed:
        return NATIVE_SHIFT_ENTER_SEQUENCE
    return data


def normalize_apple_terminal_input(data: str, is_apple_terminal: bool, is_shift_pressed: bool) -> str:
    return normalize_native_shift_enter_input(data, is_apple_terminal, is_shift_pressed)


ModifierKey = Literal["shift", "command", "control", "option"]


def is_native_modifier_pressed(_key: ModifierKey) -> bool:
    """Whether a modifier key is currently physically held.

    `native-modifiers.ts` loads a compiled Node addon (macOS/Windows only) to
    read live keyboard-modifier state, used only to distinguish a native
    Shift+Enter from a plain Enter on terminals that don't otherwise report
    it (Apple Terminal, Windows consoles without the Kitty/modifyOtherKeys
    protocols). There is no portable Python equivalent, so this always
    returns `False`: those terminals see Shift+Enter as a plain Enter, same
    as if the physical Shift key were never observed.
    """
    return False


@runtime_checkable
class StdinSource(Protocol):
    """Delivers raw stdin bytes to a callback until `stop()` is called."""

    def start(self, on_data: Callable[[bytes], None]) -> None: ...

    def stop(self) -> None: ...


@dataclass
class TerminalIo:
    """Everything `ProcessTerminal` needs from the operating system.

    Bundles the pieces the TypeScript `ProcessTerminal` reads directly from
    `process.stdin`/`process.stdout`: a write callable, a stdin byte source,
    a TTY check, a terminal-size getter, raw-mode enter/exit, and resize
    signal registration. `real_terminal_io()` builds the real one; tests
    build a fake with the same shape.
    """

    write: Callable[[str], None]
    stdin: StdinSource
    is_tty: bool
    get_size: Callable[[], tuple[int, int]]
    enter_raw_mode: Callable[[], object]
    exit_raw_mode: Callable[[object], None]
    register_resize_handler: Callable[[Callable[[], None]], Callable[[], None]]
    trigger_resize_refresh: Callable[[], None] = lambda: None
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    platform: str = field(default_factory=lambda: sys.platform)
    pid: int = field(default_factory=os.getpid)


class _RealStdinSource:
    """Reads stdin via `loop.add_reader`, requiring a running asyncio loop."""

    def __init__(self, fd: int = 0) -> None:
        self._fd = fd
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_data: Callable[[bytes], None] | None = None

    def start(self, on_data: Callable[[bytes], None]) -> None:
        self._on_data = on_data
        loop = asyncio.get_running_loop()
        self._loop = loop
        loop.add_reader(self._fd, self._read_ready)

    def _read_ready(self) -> None:
        try:
            data = os.read(self._fd, 65536)
        except OSError:
            data = b""
        if self._on_data is not None:
            self._on_data(data)

    def stop(self) -> None:
        if self._loop is not None:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self._fd)
            self._loop = None
        self._on_data = None


def _real_get_size() -> tuple[int, int]:
    try:
        size = os.get_terminal_size(sys.stdout.fileno())
    except OSError:
        return (0, 0)
    return (size.columns, size.lines)


def _real_enter_raw_mode() -> object:
    if not _HAS_TERMIOS or not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    tty.setraw(fd)
    return previous


def _real_exit_raw_mode(previous: object) -> None:
    if previous is None or not _HAS_TERMIOS:
        return
    with contextlib.suppress(Exception):
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, previous)


def _real_register_resize_handler(callback: Callable[[], None]) -> Callable[[], None]:
    if not hasattr(os, "name") or sys.platform == "win32":
        return lambda: None
    import signal

    if not hasattr(signal, "SIGWINCH"):
        return lambda: None

    previous_handler = signal.getsignal(signal.SIGWINCH)

    def _handler(_signum: int, _frame: object) -> None:
        callback()

    signal.signal(signal.SIGWINCH, _handler)

    def unregister() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGWINCH, previous_handler)

    return unregister


def _real_trigger_resize_refresh() -> None:
    if sys.platform == "win32":
        return
    import signal

    if hasattr(signal, "SIGWINCH"):
        with contextlib.suppress(Exception):
            os.kill(os.getpid(), signal.SIGWINCH)


def real_terminal_io() -> TerminalIo:
    """Build a `TerminalIo` backed by the real stdin/stdout."""

    def _write(data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()

    return TerminalIo(
        write=_write,
        stdin=_RealStdinSource(),
        is_tty=sys.stdin.isatty(),
        get_size=_real_get_size,
        enter_raw_mode=_real_enter_raw_mode,
        exit_raw_mode=_real_exit_raw_mode,
        register_resize_handler=_real_register_resize_handler,
        trigger_resize_refresh=_real_trigger_resize_refresh,
    )


class Terminal(Protocol):
    """Minimal terminal interface for TUI."""

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None: ...

    def stop(self) -> None: ...

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None: ...

    def write(self, data: str) -> None: ...

    @property
    def columns(self) -> int: ...

    @property
    def rows(self) -> int: ...

    @property
    def kitty_protocol_active(self) -> bool: ...

    def move_by(self, lines: int) -> None: ...

    def hide_cursor(self) -> None: ...

    def show_cursor(self) -> None: ...

    def clear_line(self) -> None: ...

    def clear_from_cursor(self) -> None: ...

    def clear_screen(self) -> None: ...

    def set_title(self, title: str) -> None: ...

    def set_progress(self, active: bool) -> None: ...


def _compute_write_log_path(environ: Mapping[str, str], pid: int) -> str:
    env = environ.get("PI_TUI_WRITE_LOG", "")
    if not env:
        return ""
    try:
        if os.path.isdir(env):
            now = datetime.now()
            ts = now.strftime("%Y-%m-%d_%H-%M-%S")
            return os.path.join(env, f"tui-{ts}-{pid}.log")
    except OSError:
        pass
    return env


DEFAULT_ESCAPE_TIMEOUT_MS = 10
DEFAULT_SSH_ESCAPE_TIMEOUT_MS = 100


def resolve_escape_timeout_ms(env: Mapping[str, str] | None = None) -> int:
    """How long to wait for the rest of an escape sequence before treating a
    lone ESC as the Escape key.

    Port of `resolveEscapeTimeoutMs` (`terminal.ts:112`). Legacy Alt+key input
    arrives as ESC followed by another byte, so a high-latency transport needs
    a longer reassembly window or every Alt+key is misread as a bare Escape.
    This port previously hard-coded 10 ms, which is exactly the case that
    breaks over SSH.
    """
    source = os.environ if env is None else env
    configured = source.get("PI_TUI_ESC_TIMEOUT")
    if configured is not None:
        try:
            value = float(configured)
        except ValueError:
            value = 0.0
        # `Number.isFinite` upstream: NaN and infinities fall through to the
        # defaults rather than becoming the timeout.
        if math.isfinite(value) and value > 0:
            return int(value)
    if source.get("SSH_CONNECTION") or source.get("SSH_TTY"):
        return DEFAULT_SSH_ESCAPE_TIMEOUT_MS
    return DEFAULT_ESCAPE_TIMEOUT_MS


class ProcessTerminal:
    """Real terminal, driven through an injected `TerminalIo`."""

    def __init__(self, terminal_io: TerminalIo | None = None) -> None:
        self._io = terminal_io if terminal_io is not None else real_terminal_io()
        self._was_raw_state: object = None
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._kitty_protocol_active = False
        self._modify_other_keys_active = False
        self._keyboard_protocol_pushed = False
        self._keyboard_protocol_negotiation_buffer = ""
        self._keyboard_protocol_buffer_flush_timer: asyncio.TimerHandle | None = None
        self._stdin_buffer: StdinBuffer | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._unregister_resize: Callable[[], None] | None = None
        self._extra_byte_listeners: list[Callable[[bytes], None]] = []
        self._progress_task: asyncio.Task[None] | None = None
        self._write_log_path = _compute_write_log_path(self._io.environ, self._io.pid)

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    @property
    def modify_other_keys_active(self) -> bool:
        return self._modify_other_keys_active

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize

        self._was_raw_state = self._io.enter_raw_mode()

        self._io.write("\x1b[?2004h")

        self._unregister_resize = self._io.register_resize_handler(lambda: self._on_resize())

        self._io.trigger_resize_refresh()

        self._query_and_enable_kitty_protocol()

    def _on_resize(self) -> None:
        if self._resize_handler is not None:
            self._resize_handler()

    def _setup_stdin_buffer(self) -> None:
        self._stdin_buffer = StdinBuffer(timeout=resolve_escape_timeout_ms() / 1000)
        self._stdin_buffer.on("data", self._on_stdin_buffer_data)
        self._stdin_buffer.on("paste", self._on_stdin_buffer_paste)

    def _on_stdin_buffer_data(self, sequence: str) -> None:
        negotiation = self._read_keyboard_protocol_negotiation_sequence(sequence)
        if negotiation == "pending":
            self._schedule_keyboard_protocol_negotiation_buffer_flush()
            return
        if self._handle_keyboard_protocol_negotiation_sequence(negotiation):
            return
        self._forward_input_sequence(sequence)

    def _on_stdin_buffer_paste(self, content: str) -> None:
        if self._input_handler is not None:
            self._input_handler(f"\x1b[200~{content}\x1b[201~")

    def _query_and_enable_kitty_protocol(self) -> None:
        self._setup_stdin_buffer()
        self._io.stdin.start(self._on_stdin_bytes)
        self._keyboard_protocol_pushed = True
        self._clear_keyboard_protocol_negotiation_buffer()
        self._io.write(KITTY_KEYBOARD_PROTOCOL_QUERY)

    def _on_stdin_bytes(self, raw: bytes) -> None:
        decoded = self._decoder.decode(raw)
        if self._stdin_buffer is not None and decoded:
            self._stdin_buffer.process(decoded)
        for listener in list(self._extra_byte_listeners):
            listener(raw)

    def _handle_keyboard_protocol_negotiation_sequence(
        self, negotiation: KeyboardProtocolNegotiationSequence | None
    ) -> bool:
        if negotiation is None:
            return False
        self._clear_keyboard_protocol_negotiation_buffer()
        if isinstance(negotiation, KittyFlagsNegotiation):
            if negotiation.flags != 0:
                self._disable_modify_other_keys()
                if not self._kitty_protocol_active:
                    self._kitty_protocol_active = True
                    set_kitty_protocol_active(True)
            else:
                self._enable_modify_other_keys()
            return True

        if not self._kitty_protocol_active:
            self._enable_modify_other_keys()
        return True

    def _read_keyboard_protocol_negotiation_sequence(
        self, sequence: str
    ) -> KeyboardProtocolNegotiationSequence | Literal["pending"] | None:
        if self._keyboard_protocol_negotiation_buffer:
            buffered = self._keyboard_protocol_negotiation_buffer + sequence
            negotiation = parse_keyboard_protocol_negotiation_sequence(buffered)
            if negotiation:
                self._clear_keyboard_protocol_negotiation_buffer()
                return negotiation
            if _is_keyboard_protocol_negotiation_sequence_prefix(buffered):
                self._set_keyboard_protocol_negotiation_buffer(buffered)
                return "pending"
            self._flush_keyboard_protocol_negotiation_buffer_as_input()

        negotiation = parse_keyboard_protocol_negotiation_sequence(sequence)
        if negotiation:
            return negotiation
        if _is_keyboard_protocol_negotiation_sequence_prefix(sequence):
            self._set_keyboard_protocol_negotiation_buffer(sequence)
            return "pending"
        return None

    def _set_keyboard_protocol_negotiation_buffer(self, sequence: str) -> None:
        self._clear_keyboard_protocol_negotiation_buffer_flush_timer()
        self._keyboard_protocol_negotiation_buffer = sequence

    def _clear_keyboard_protocol_negotiation_buffer(self) -> None:
        self._clear_keyboard_protocol_negotiation_buffer_flush_timer()
        self._keyboard_protocol_negotiation_buffer = ""

    def _flush_keyboard_protocol_negotiation_buffer_as_input(self) -> None:
        if not self._keyboard_protocol_negotiation_buffer:
            return
        sequence = self._keyboard_protocol_negotiation_buffer
        self._clear_keyboard_protocol_negotiation_buffer()
        self._forward_input_sequence(sequence)

    def _schedule_keyboard_protocol_negotiation_buffer_flush(self) -> None:
        if not self._keyboard_protocol_negotiation_buffer or self._keyboard_protocol_buffer_flush_timer is not None:
            return

        def _on_timeout() -> None:
            self._keyboard_protocol_buffer_flush_timer = None
            self._flush_keyboard_protocol_negotiation_buffer_as_input()

        loop = asyncio.get_running_loop()
        self._keyboard_protocol_buffer_flush_timer = loop.call_later(
            KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_S, _on_timeout
        )

    def _clear_keyboard_protocol_negotiation_buffer_flush_timer(self) -> None:
        if self._keyboard_protocol_buffer_flush_timer is None:
            return
        self._keyboard_protocol_buffer_flush_timer.cancel()
        self._keyboard_protocol_buffer_flush_timer = None

    def _forward_input_sequence(self, sequence: str) -> None:
        if self._input_handler is None:
            return
        should_detect_native_shift_enter = sequence == "\r" and (
            is_apple_terminal_session() or self._io.platform == "win32"
        )
        data = normalize_native_shift_enter_input(
            sequence,
            should_detect_native_shift_enter,
            should_detect_native_shift_enter and is_native_modifier_pressed("shift"),
        )
        self._input_handler(data)

    def _enable_modify_other_keys(self) -> None:
        if self._kitty_protocol_active or self._modify_other_keys_active:
            return
        self._io.write("\x1b[>4;2m")
        self._modify_other_keys_active = True

    def _disable_modify_other_keys(self) -> None:
        if not self._modify_other_keys_active:
            return
        self._io.write("\x1b[>4;0m")
        self._modify_other_keys_active = False

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        should_disable_kitty = self._keyboard_protocol_pushed or self._kitty_protocol_active
        self._clear_keyboard_protocol_negotiation_buffer()
        if should_disable_kitty:
            self._io.write("\x1b[<u")
            self._keyboard_protocol_pushed = False
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        self._disable_modify_other_keys()

        previous_handler = self._input_handler
        self._input_handler = None

        loop = asyncio.get_running_loop()
        last_data_time = loop.time()

        def on_data(_raw: bytes) -> None:
            nonlocal last_data_time
            last_data_time = loop.time()

        self._extra_byte_listeners.append(on_data)
        end_time = loop.time() + max_ms / 1000
        idle_s = idle_ms / 1000
        try:
            while True:
                now = loop.time()
                time_left = end_time - now
                if time_left <= 0:
                    break
                if now - last_data_time >= idle_s:
                    break
                await asyncio.sleep(min(idle_s, time_left))
        finally:
            with contextlib.suppress(ValueError):
                self._extra_byte_listeners.remove(on_data)
            self._input_handler = previous_handler

    def stop(self) -> None:
        if self._clear_progress_task():
            self._io.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

        self._io.write("\x1b[?2004l")

        should_disable_kitty = self._keyboard_protocol_pushed or self._kitty_protocol_active
        self._clear_keyboard_protocol_negotiation_buffer()

        if should_disable_kitty:
            self._io.write("\x1b[<u")
            self._keyboard_protocol_pushed = False
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        self._disable_modify_other_keys()

        if self._stdin_buffer is not None:
            self._stdin_buffer.destroy()
            self._stdin_buffer = None

        self._io.stdin.stop()
        self._input_handler = None
        if self._unregister_resize is not None:
            self._unregister_resize()
            self._unregister_resize = None
        self._resize_handler = None

        self._io.exit_raw_mode(self._was_raw_state)

    def write(self, data: str) -> None:
        self._io.write(data)
        if self._write_log_path:
            with contextlib.suppress(OSError), open(self._write_log_path, "a", encoding="utf-8") as f:
                f.write(data)

    @property
    def columns(self) -> int:
        columns, _ = self._io.get_size()
        if columns:
            return columns
        env_columns = self._io.environ.get("COLUMNS")
        if env_columns:
            with contextlib.suppress(ValueError):
                value = int(env_columns)
                if value:
                    return value
        return 80

    @property
    def rows(self) -> int:
        _, rows = self._io.get_size()
        if rows:
            return rows
        env_rows = self._io.environ.get("LINES")
        if env_rows:
            with contextlib.suppress(ValueError):
                value = int(env_rows)
                if value:
                    return value
        return 24

    def move_by(self, lines: int) -> None:
        if lines > 0:
            self.write(f"\x1b[{lines}B")
        elif lines < 0:
            self.write(f"\x1b[{-lines}A")

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        self.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        self.write("\x1b[J")

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self.write(f"\x1b]0;{title}\x07")

    def set_progress(self, active: bool) -> None:
        if active:
            self.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)
            if self._progress_task is None:
                self._progress_task = spawn(self._progress_keepalive())
        else:
            self._clear_progress_task()
            self.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

    async def _progress_keepalive(self) -> None:
        while True:
            await asyncio.sleep(TERMINAL_PROGRESS_KEEPALIVE_S)
            self.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)

    def _clear_progress_task(self) -> bool:
        if self._progress_task is None:
            return False
        self._progress_task.cancel()
        self._progress_task = None
        return True
