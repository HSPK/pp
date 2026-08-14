"""In-memory fakes for the terminal I/O boundary.

Shipped with the package rather than kept in `tests/`: `pp-coding-agent` builds
its interactive-mode tests on these doubles, and a helper that lives only in
another distribution's test directory is not importable once the two are
separate installs. It happened to work in the monorepo because `pytest` put
every package's `tests/` on `sys.path`.

Used by tests exercising `pi_tui.terminal`, `pi_tui.tui`, `pi_tui.tui_main_screen`,
and `pi_tui.tui_alt_screen` so none of them ever touch a real TTY or block on
stdin. Mirrors the role of `packages/tui/test/virtual-terminal.ts`, but at the
lower `TerminalIo` layer (see `pi_tui.terminal.TerminalIo`) rather than the
higher-level `Terminal` interface, so both `ProcessTerminal` itself and
anything built on top of it can be tested headlessly.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable

from pi_tui.terminal import TerminalIo


class FakeStdinSource:
    """Fake `StdinSource`: delivers bytes via `send()` instead of a real fd."""

    def __init__(self) -> None:
        self._on_data: Callable[[bytes], None] | None = None
        self.started = False
        self.stopped = False

    def start(self, on_data: Callable[[bytes], None]) -> None:
        self._on_data = on_data
        self.started = True
        self.stopped = False

    def stop(self) -> None:
        self._on_data = None
        self.stopped = True

    def send(self, data: bytes) -> None:
        if self._on_data is not None:
            self._on_data(data)


class FakeTerminal:
    """Fake `Terminal`: an in-memory stand-in for `ProcessTerminal`.

    Mirrors the TypeScript test suite's `TestTerminal` (see
    `packages/tui/test/terminal-colors.test.ts` and `tui-render.test.ts`):
    used to test `TuiBase`/`TuiMainScreen`/`TuiAltScreen` directly, without
    going through stdin-byte parsing or a real TTY.
    """

    def __init__(self, columns: int = 80, rows: int = 24, *, kitty_protocol_active: bool = False) -> None:
        self._columns = columns
        self._rows = rows
        self._kitty_protocol_active = kitty_protocol_active
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self.writes: list[str] = []
        self.progress_calls: list[bool] = []
        self.titles: list[str] = []

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize

    def stop(self) -> None:
        self._input_handler = None
        self._resize_handler = None

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        return None

    def write(self, data: str) -> None:
        self.writes.append(data)

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def kitty_protocol_active(self) -> bool:
        # Read-only property, like `ProcessTerminal.kitty_protocol_active` and
        # the TypeScript `TestTerminal`'s `get kittyProtocolActive()`. A plain
        # writable attribute would be a broader shape than production.
        return self._kitty_protocol_active

    def move_by(self, lines: int) -> None:
        if lines > 0:
            self.writes.append(f"\x1b[{lines}B")
        elif lines < 0:
            self.writes.append(f"\x1b[{-lines}A")

    def hide_cursor(self) -> None:
        self.writes.append("\x1b[?25l")

    def show_cursor(self) -> None:
        self.writes.append("\x1b[?25h")

    def clear_line(self) -> None:
        self.writes.append("\x1b[K")

    def clear_from_cursor(self) -> None:
        self.writes.append("\x1b[J")

    def clear_screen(self) -> None:
        self.writes.append("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self.titles.append(title)
        self.writes.append(f"\x1b]0;{title}\x07")

    def set_progress(self, active: bool) -> None:
        self.progress_calls.append(active)

    # -- test helpers -------------------------------------------------

    def send_input(self, data: str) -> None:
        if self._input_handler is not None:
            self._input_handler(data)

    def send_resize(self, columns: int | None = None, rows: int | None = None) -> None:
        if columns is not None:
            self._columns = columns
        if rows is not None:
            self._rows = rows
        if self._resize_handler is not None:
            self._resize_handler()


class FakeTerminalIo:
    """Fake `TerminalIo`: records writes, simulates resize/raw-mode/size.

    Build the `TerminalIo` bundle passed to `ProcessTerminal` with `.build()`;
    use the other methods/attributes to inspect or drive it from a test.
    """

    def __init__(
        self,
        columns: int = 80,
        rows: int = 24,
        is_tty: bool = True,
        environ: dict[str, str] | None = None,
        platform: str = "linux",
        pid: int = 4242,
    ) -> None:
        self.writes: list[str] = []
        self.stdin = FakeStdinSource()
        self._columns = columns
        self._rows = rows
        self.is_tty = is_tty
        self.environ = dict(environ or {})
        self.platform = platform
        self.pid = pid
        self.raw_mode_enter_count = 0
        self.raw_mode_exit_count = 0
        self.last_raw_mode_exit_arg: object = "not-called"
        self.resize_callback: Callable[[], None] | None = None
        self.resize_refresh_count = 0

    def _write(self, data: str) -> None:
        self.writes.append(data)

    def _get_size(self) -> tuple[int, int]:
        return (self._columns, self._rows)

    def _enter_raw_mode(self) -> object:
        self.raw_mode_enter_count += 1
        return "fake-previous-termios-state"

    def _exit_raw_mode(self, previous: object) -> None:
        self.raw_mode_exit_count += 1
        self.last_raw_mode_exit_arg = previous

    def _register_resize_handler(self, callback: Callable[[], None]) -> Callable[[], None]:
        self.resize_callback = callback

        def unregister() -> None:
            if self.resize_callback is callback:
                self.resize_callback = None

        return unregister

    def _trigger_resize_refresh(self) -> None:
        self.resize_refresh_count += 1

    def build(self) -> TerminalIo:
        return TerminalIo(
            write=self._write,
            stdin=self.stdin,
            is_tty=self.is_tty,
            get_size=self._get_size,
            enter_raw_mode=self._enter_raw_mode,
            exit_raw_mode=self._exit_raw_mode,
            register_resize_handler=self._register_resize_handler,
            trigger_resize_refresh=self._trigger_resize_refresh,
            environ=self.environ,
            platform=self.platform,
            pid=self.pid,
        )

    def send(self, data: str | bytes) -> None:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        self.stdin.send(raw)

    def resize(self, columns: int, rows: int) -> None:
        self._columns = columns
        self._rows = rows
        if self.resize_callback is not None:
            self.resize_callback()


class MiniTerminalModel:
    """Minimal terminal-content model for TuiMainScreen's differential output.

    Not a general escape-sequence interpreter: only understands the small
    subset of sequences `TuiMainScreen`/`TuiBase` actually emit (cursor
    up/down/absolute-column, clear-line, clear-screen+scrollback, CR, CRLF,
    SGR/OSC-8 resets, plain text). Lets tests assert on rendered *content*
    (mirroring the TypeScript suite's xterm.js-backed `VirtualTerminal`)
    without depending on a real terminal-emulator library.

    Also tracks the italic and underline SGR attributes per cell
    (`viewport_italics` / `cell_italic`, `cell_underline`), which is what
    `packages/tui/test/tui-overlay-style-leak.test.ts` and
    `packages/tui/test/markdown.test.ts` read off xterm.js cells to detect
    style leaking past a token or line boundary.
    """

    _MOVE_PATTERN = re.compile(r"\x1b\[(\d+)([ABG])")
    _SGR_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
    _OSC8_RESET_PATTERN = re.compile(r"\x1b\]8;;\x07")

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.lines: list[str] = [""]
        self.italics: list[list[bool]] = [[]]
        self.underlines: list[list[bool]] = [[]]
        self.row = 0
        self.col = 0
        self._italic = False
        self._underline = False

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            if data.startswith(("\x1b[?2026h", "\x1b[?2026l"), i):
                i += 8
                continue
            if data.startswith(("\x1b[?25l", "\x1b[?25h"), i):
                i += 6
                continue
            if data.startswith("\x1b[2J\x1b[H\x1b[3J", i):
                self.lines = [""]
                self.italics = [[]]
                self.underlines = [[]]
                self.row = 0
                self.col = 0
                i += len("\x1b[2J\x1b[H\x1b[3J")
                continue
            if data.startswith("\x1b[2K", i):
                self.lines[self.row] = ""
                self.italics[self.row] = []
                self.underlines[self.row] = []
                self.col = 0
                i += 4
                continue
            move = self._MOVE_PATTERN.match(data, i)
            if move:
                n = int(move.group(1))
                letter = move.group(2)
                if letter == "A":
                    self.row = max(0, self.row - n)
                elif letter == "B":
                    self.row += n
                    self._ensure_row()
                elif letter == "G":
                    self.col = n - 1
                i = move.end()
                continue
            if data.startswith("\r\n", i):
                self.row += 1
                self._ensure_row()
                self.col = 0
                i += 2
                continue
            if data[i] == "\r":
                self.col = 0
                i += 1
                continue
            if data[i] == "\n":
                self.row += 1
                self._ensure_row()
                i += 1
                continue
            sgr = self._SGR_PATTERN.match(data, i)
            if sgr:
                self._apply_sgr(sgr.group(0))
                i = sgr.end()
                continue
            osc8 = self._OSC8_RESET_PATTERN.match(data, i)
            if osc8:
                i = osc8.end()
                continue
            line = self.lines[self.row]
            if len(line) < self.col:
                line = line + " " * (self.col - len(line))
            self.lines[self.row] = line[: self.col] + data[i] + line[self.col + 1 :]
            attrs = self.italics[self.row]
            while len(attrs) <= self.col:
                attrs.append(False)
            attrs[self.col] = self._italic
            underline_attrs = self.underlines[self.row]
            while len(underline_attrs) <= self.col:
                underline_attrs.append(False)
            underline_attrs[self.col] = self._underline
            self.col += 1
            i += 1

    def _apply_sgr(self, sequence: str) -> None:
        body = sequence[2:-1]
        params = body.split(";") if body else [""]
        for param in params:
            code = int(param) if param else 0
            if code == 3:
                self._italic = True
            elif code == 4:
                self._underline = True
            elif code == 23:
                self._italic = False
            elif code == 24:
                self._underline = False
            elif code == 0:
                self._italic = False
                self._underline = False

    def _ensure_row(self) -> None:
        while self.row >= len(self.lines):
            self.lines.append("")
            self.italics.append([])
            self.underlines.append([])

    def viewport(self) -> list[str]:
        start = max(0, len(self.lines) - self.height)
        result = list(self.lines[start:])
        while len(result) < self.height:
            result.append("")
        return result

    def viewport_italics(self) -> list[list[bool]]:
        start = max(0, len(self.lines) - self.height)
        result = [list(row) for row in self.italics[start:]]
        while len(result) < self.height:
            result.append([])
        return result

    def viewport_underlines(self) -> list[list[bool]]:
        start = max(0, len(self.lines) - self.height)
        result = [list(row) for row in self.underlines[start:]]
        while len(result) < self.height:
            result.append([])
        return result

    def cell_italic(self, row: int, col: int) -> bool:
        attrs = self.viewport_italics()[row]
        return attrs[col] if col < len(attrs) else False

    def cell_underline(self, row: int, col: int) -> bool:
        attrs = self.viewport_underlines()[row]
        return attrs[col] if col < len(attrs) else False


class MiniAltScreenModel:
    """Minimal fixed-grid terminal model for `TuiAltScreen`'s escape output.

    Unlike `MiniTerminalModel` (which models a scrolling main-screen buffer),
    the alternate screen is a fixed-size grid addressed with absolute
    `\\x1b[row;colH` (CUP) cursor positioning and cleared wholesale with a
    bare `\\x1b[2J` (no scrollback to clear). Understands only the sequences
    `TuiAltScreen` actually emits: `\\x1b[?2026h`/`\\x1b[?2026l` (sync output,
    ignored), `\\x1b[?25l`/`\\x1b[?25h` (cursor show/hide, ignored), `\\x1b[2J`
    (clear grid), `\\x1b[row;colH` (absolute cursor position), `\\x1b[2K`
    (clear current line), SGR (`\\x1b[...m`) and OSC-8/OSC-52 sequences
    (stripped from the content grid; use `terminal.writes` directly to assert
    on the raw escape sequences for selection-highlight/clipboard tests).
    """

    _CUP_PATTERN = re.compile(r"\x1b\[(\d+);(\d+)H")
    _SGR_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
    _OSC8_RESET_PATTERN = re.compile(r"\x1b\]8;;\x07")
    _OSC8_LINK_START_PATTERN = re.compile(r"\x1b\]8;[^;]*;[^\x07\x1b]*(?:\x07|\x1b\\)")
    _OSC52_PATTERN = re.compile(r"\x1b\]52;c;([^\x07]*)\x07")
    #: Kitty graphics commands (APC ... ST). Real terminals consume these
    #: without advancing the cursor, so they must not land in the grid.
    _APC_PATTERN = re.compile(r"\x1b_[^\x1b]*\x1b\\")
    #: Any other OSC (iTerm2 `1337;File=`, shell-integration `133;`, ...).
    _OSC_PATTERN = re.compile(r"\x1b\][0-9]+;[^\x07\x1b]*(?:\x07|\x1b\\)")

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid: list[str] = ["" for _ in range(height)]
        self.row = 0
        self.col = 0
        self.last_osc52_payload: str | None = None

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            if data.startswith(("\x1b[?2026h", "\x1b[?2026l"), i):
                i += 8
                continue
            if data.startswith(("\x1b[?25l", "\x1b[?25h"), i):
                i += 6
                continue
            if data.startswith("\x1b[2J", i):
                self.grid = ["" for _ in range(self.height)]
                i += 4
                continue
            if data.startswith("\x1b[2K", i):
                self._ensure_row()
                self.grid[self.row] = ""
                self.col = 0
                i += 4
                continue
            cup = self._CUP_PATTERN.match(data, i)
            if cup:
                self.row = int(cup.group(1)) - 1
                self.col = int(cup.group(2)) - 1
                i = cup.end()
                continue
            osc52 = self._OSC52_PATTERN.match(data, i)
            if osc52:
                self.last_osc52_payload = osc52.group(1)
                i = osc52.end()
                continue
            osc8 = self._OSC8_RESET_PATTERN.match(data, i)
            if osc8:
                i = osc8.end()
                continue
            osc8_link = self._OSC8_LINK_START_PATTERN.match(data, i)
            if osc8_link:
                i = osc8_link.end()
                continue
            sgr = self._SGR_PATTERN.match(data, i)
            if sgr:
                i = sgr.end()
                continue
            apc = self._APC_PATTERN.match(data, i)
            if apc:
                i = apc.end()
                continue
            osc = self._OSC_PATTERN.match(data, i)
            if osc:
                i = osc.end()
                continue
            self._ensure_row()
            line = self.grid[self.row]
            if len(line) < self.col:
                line = line + " " * (self.col - len(line))
            self.grid[self.row] = line[: self.col] + data[i] + line[self.col + 1 :]
            self.col += 1
            i += 1

    def _ensure_row(self) -> None:
        while self.row >= len(self.grid):
            self.grid.append("")

    def screen(self) -> list[str]:
        result = list(self.grid[: self.height])
        while len(result) < self.height:
            result.append("")
        return result


class _ManualTimerHandle:
    """Cancellable stand-in for `asyncio.TimerHandle`."""

    def __init__(self, when: float, callback: Callable[..., None], args: tuple[object, ...]) -> None:
        self.when_ = when
        self.callback = callback
        self.args = args
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def when(self) -> float:
        return self.when_


class ManualTimers:
    """Fake timers for `loop.call_later`, in the role of node's `mock.timers`.

    `packages/tui/test/terminal.test.ts` drives the Kitty negotiation timeouts
    with `mock.timers.enable({apis: ["setTimeout"]})` and `mock.timers.tick(ms)`.
    Porting `tick(150)` as a real 150 ms `asyncio.sleep` makes the test depend on
    wall-clock time, which is unreliable when the suite runs in parallel. This
    captures `call_later` scheduling instead so `tick()` advances a virtual
    clock exactly like the TypeScript test does.

    Timers scheduled by callbacks that `tick()` fires are themselves eligible in
    the same `tick()` if their deadline falls inside the advanced window, which
    matches node's behavior.
    """

    def __init__(self, loop: object) -> None:
        self._loop = loop
        self._now = 0.0
        self._pending: list[_ManualTimerHandle] = []
        self._real_call_later = loop.call_later  # type: ignore[attr-defined]

    def install(self) -> None:
        self._loop.call_later = self._call_later  # type: ignore[attr-defined]

    def uninstall(self) -> None:
        self._loop.call_later = self._real_call_later  # type: ignore[attr-defined]

    def _call_later(self, delay: float, callback: Callable[..., None], *args: object) -> _ManualTimerHandle:
        handle = _ManualTimerHandle(self._now + delay, callback, args)
        self._pending.append(handle)
        return handle

    @property
    def pending(self) -> list[_ManualTimerHandle]:
        return [handle for handle in self._pending if not handle.cancelled]

    def tick(self, milliseconds: float) -> None:
        """Advance the virtual clock and run every callback that comes due."""
        target = self._now + milliseconds / 1000
        while True:
            due = sorted(
                (h for h in self._pending if not h.cancelled and h.when_ <= target),
                key=lambda h: h.when_,
            )
            if not due:
                break
            for handle in due:
                self._pending.remove(handle)
                self._now = max(self._now, handle.when_)
                handle.callback(*handle.args)
        self._now = target


async def wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0, message: str = "") -> None:
    """Poll `predicate` until it holds, or fail after `timeout` seconds.

    Preferred over sleeping for a fixed duration when a test waits for a
    production timer to fire: the wait ends as soon as the outcome is visible,
    and a loaded machine (the suite runs with `-n auto`) gets the slack it needs
    instead of failing on a wall-clock margin.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError(message or f"condition not met within {timeout}s")
        await asyncio.sleep(0.001)
