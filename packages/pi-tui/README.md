# pi-tui

Minimal terminal UI framework with differential rendering and synchronized output for flicker-free interactive CLI applications.

## Features

- **Interchangeable renderers**: shared `TuiBase` API with main-screen and alternate-screen implementations.
- **Differential rendering**: updates only changed main-screen lines or fullscreen viewport rows.
- **Application-owned scrolling**: `TuiAltScreen` owns a fixed-height viewport with mouse, trackpad, keyboard navigation, scrollbars, OSC 8 hyperlinks, and text selection.
- **Synchronized output**: uses CSI 2026 for atomic screen updates.
- **Bracketed paste mode**: `Input` and `Editor` handle terminal paste markers; large editor pastes are collapsed into `[paste #N ...]` markers.
- **Component-based**: components implement `render(width)` and optional input/invalidation hooks.
- **Theme support**: components accept callable theme objects for styling.
- **Built-in components**: `Text`, `TruncatedText`, `Input`, `Editor`, `Markdown`, `Loader`, `CancellableLoader`, `SelectList`, `SettingsList`, `Spacer`, `Image`, `Box`, `Container`, `VStack`, `HStack`, and `ScrollView`.
- **Inline images**: renders Kitty or iTerm2 inline images when the terminal supports them.
- **Autocomplete support**: slash commands, file paths, and `@` attachment paths through `CombinedAutocompleteProvider`.

Not ported from the TypeScript package: the `VirtualTerminal` test implementation, the fullscreen search UI and `searchMatchStyle` options, and Windows-only right-click paste handling. `alt_screen_search.py` contains the search matching helper, but `TuiAltScreen` does not expose the interactive search UI.

## Quick Start

```python
from pi_tui import (
    Editor,
    EditorOptions,
    EditorTheme,
    ProcessTerminal,
    SelectListTheme,
    Text,
    TuiMainScreen,
    matches_key,
)


def plain(text: str) -> str:
    return text


terminal = ProcessTerminal()
tui = TuiMainScreen(terminal)

tui.add_child(Text("Welcome to my app!"))

editor_theme = EditorTheme(
    border_color=plain,
    select_list=SelectListTheme(
        selected_prefix=plain,
        selected_text=plain,
        description=plain,
        scroll_info=plain,
        no_match=plain,
    ),
)
editor = Editor(tui, editor_theme, EditorOptions(padding_x=1))


def on_submit(text: str) -> None:
    tui.add_child(Text(f"You said: {text}"))


editor.on_submit = on_submit
tui.add_child(editor)
tui.set_focus(editor)


def exit_on_ctrl_c(data: str):
    if matches_key(data, "ctrl+c"):
        tui.stop()
        raise SystemExit(0)
    return None


tui.add_input_listener(exit_on_ctrl_c)
tui.start()
```

## Core API

### TUI interface and renderers

The Python port exposes `TuiBase` as the common renderer base. Construct one concrete renderer at the application boundary:

- `TuiMainScreen` renders into the main terminal buffer and preserves terminal scrollback. Its `mode` is `"regular"`.
- `TuiAltScreen` renders a fixed-height viewport in the alternate terminal buffer with application-owned scrolling. Its `mode` is `"fullscreen"`. When stopped without `TuiStopOptions(preserve_screen=True)`, it restores the main buffer and prints the complete final document.

```python
from pi_tui import ProcessTerminal, Text, TuiAltScreen, TuiMainScreen

terminal = ProcessTerminal()
tui = TuiMainScreen(terminal)
# tui = TuiAltScreen(terminal)

component = Text("hello")
tui.add_child(component)
tui.remove_child(component)
tui.add_child(component)
tui.request_render()
tui.start()
tui.stop()


def debug() -> None:
    print("Debug triggered")


tui.on_debug = debug
```

### Alternate-screen viewport layouts

`TuiAltScreen` can render an explicit terminal-height layout. `VStack` and `HStack` allocate constrained regions, while `ScrollView` owns scrolling for one region. These semantics are unavailable on `TuiMainScreen`, where the terminal owns scrollback.

```python
from pi_tui import Container, ScrollView, ScrollViewOptions, StackEntry, Text, TuiAltScreen, VStack

transcript = Container()
transcript.add_child(Text("History"))

editor_and_footer = VStack([Text("> "), Text("status")])
tui = TuiAltScreen(terminal)

tui.set_layout_root(
    VStack(
        [
            StackEntry(
                component=ScrollView(
                    transcript,
                    ScrollViewOptions(follow="end", primary=True, overscroll="chain"),
                ),
                basis=0,
                grow=1,
                min_size=1,
            ),
            StackEntry(
                component=editor_and_footer,
                basis="auto",
                shrink=1,
                min_size=1,
            ),
        ]
    )
)
```

Stack entries support `basis`, `grow`, `shrink`, `min_size`, `max_size`, and responsive `visible` callbacks. Mouse-wheel input targets the scroll view under the pointer, and unused delta chains to outer scroll views by default. The primary scroll view receives fullscreen keyboard navigation actions and wheel input over non-scrollable regions. Prompt-marker jumps use OSC 133 markers and the `tui.altScreen.previousPrompt` / `tui.altScreen.nextPrompt` keybindings.

Layout geometry is rebuilt for each requested frame. Stateful components are retained, and their rendered-line caches remain effective. Calling `render(width)` directly on layout components produces an unbounded document, which is also used when alt mode restores the main screen.

### Overlays

Overlays render components on top of existing content. They are useful for dialogs, menus, and modal UI.

```python
from pi_tui import OverlayOptions, Text

handle = tui.show_overlay(Text("Dialog"))

menu = Text("Menu")
handle = tui.show_overlay(
    menu,
    OverlayOptions(
        width="80%",
        min_width=40,
        max_height="50%",
        anchor="bottom-right",
        offset_x=2,
        offset_y=-1,
        margin=2,
        visible=lambda term_width, term_height: term_width >= 100,
        non_capturing=True,
    ),
)

handle.set_hidden(True)
handle.set_hidden(False)
handle.focus()
handle.unfocus(None)
handle.hide()

tui.hide_overlay()
has_overlay = tui.has_overlay()
```

**Anchor values**: `"center"`, `"top-left"`, `"top-right"`, `"bottom-left"`, `"bottom-right"`, `"top-center"`, `"bottom-center"`, `"left-center"`, `"right-center"`.

**Resolution order**:

1. `min_width` is applied as a floor after width calculation.
2. For position: absolute `row`/`col` > percentage `row`/`col` > `anchor`.
3. `margin` clamps final position to stay within terminal bounds.
4. `visible` controls whether the overlay renders on each frame.

### Component Interface

All components implement this structural interface:

```python
from typing import Protocol


class ComponentLike(Protocol):
    def render(self, width: int) -> list[str]: ...

    def handle_input(self, data: str) -> None: ...

    def invalidate(self) -> None: ...
```

| Method | Description |
|--------|-------------|
| `render(width)` | Returns one string per line. Each line must not exceed `width`, or the TUI will error. Use `truncate_to_width()` or wrapping utilities to enforce this. |
| `handle_input(data)` | Called when the component has focus and receives keyboard input. The string contains raw terminal input, including ANSI escape sequences. |
| `invalidate()` | Clears cached render state. Components should re-render from scratch on the next `render()` call. |

The TUI appends a full SGR reset and OSC 8 reset at the end of each rendered line. Styles do not carry across lines. If you emit multi-line styled text, reapply styles per line or use `wrap_text_with_ansi()`.

### Focusable Interface (IME Support)

Components that display a text cursor and need IME (Input Method Editor) support should expose a `focused` attribute and emit `CURSOR_MARKER` immediately before the fake cursor.

```python
from pi_tui import CURSOR_MARKER, Component, Focusable


class MyInput(Component, Focusable):
    def __init__(self) -> None:
        self.focused = False
        self.before_cursor = ""
        self.at_cursor = " "
        self.after_cursor = ""

    def render(self, width: int) -> list[str]:
        marker = CURSOR_MARKER if self.focused else ""
        return [f"> {self.before_cursor}{marker}\x1b[7m{self.at_cursor}\x1b[27m{self.after_cursor}"]
```

When a focusable component has focus, TUI sets `focused = True`, scans for `CURSOR_MARKER`, positions the hardware cursor there, and shows it only when enabled. Enable the hardware cursor with the renderer constructor's `show_hardware_cursor` argument, `set_show_hardware_cursor(True)`, or `PI_HARDWARE_CURSOR=1`. `Editor` and `Input` already implement this interface.

**Container components with embedded inputs:** propagate focus to the child input.

```python
from pi_tui import Container, Focusable, Input


class SearchDialog(Container, Focusable):
    def __init__(self) -> None:
        super().__init__()
        self.search_input = Input()
        self.add_child(self.search_input)
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.search_input.focused = value
```

Without this propagation, IME candidate windows can appear in the wrong position.

## Built-in Components

### Container

Groups child components.

```python
from pi_tui import Container, Text

container = Container()
component = Text("child")
container.add_child(component)
container.remove_child(component)
```

### Box

Container that applies padding and background color to all children.

```python
from pi_tui import Box, Text

box = Box(1, 1, lambda text: f"\x1b[100m{text}\x1b[49m")
box.add_child(Text("Content"))
box.set_bg_fn(lambda text: f"\x1b[44m{text}\x1b[49m")
```

### Text

Displays multi-line text with word wrapping and padding.

```python
from pi_tui import Text

text = Text("Hello World", 1, 1, lambda value: value)
text.set_text("Updated text")
text.set_custom_bg_fn(lambda value: value)
```

### TruncatedText

Single-line text that truncates to fit the viewport width. Useful for status lines and headers.

```python
from pi_tui import TruncatedText

truncated = TruncatedText("This is a very long line that will be truncated...", 0, 0)
```

### Input

Single-line text input with horizontal scrolling.

```python
from pi_tui import Input

input_component = Input()
input_component.on_submit = lambda value: print(value)
input_component.set_value("initial")
value = input_component.get_value()
```

**Key bindings:**

- `Enter` - Submit
- `Ctrl+A` / `Ctrl+E` - Line start/end
- `Ctrl+W` or `Alt+Backspace` - Delete word backwards
- `Ctrl+U` - Delete to start of line
- `Ctrl+K` - Delete to end of line
- `Ctrl+Left` / `Ctrl+Right` and `Alt+Left` / `Alt+Right` - Word navigation
- Arrow keys, Backspace, Delete work as expected

### Editor

Multi-line text editor with autocomplete, file completion, paste handling, history, undo, kill ring, and vertical scrolling when content exceeds terminal height.

```python
from pi_tui import Editor, EditorOptions, EditorTheme, SelectListTheme


def plain(text: str) -> str:
    return text


theme = EditorTheme(
    border_color=plain,
    select_list=SelectListTheme(plain, plain, plain, plain, plain),
)
editor = Editor(tui, theme, EditorOptions(padding_x=1, autocomplete_max_visible=8))
editor.on_submit = lambda text: print(text)
editor.on_change = lambda text: print("Changed:", text)
editor.disable_submit = True
editor.set_autocomplete_provider(provider)
editor.border_color = plain
editor.set_padding_x(1)
padding = editor.get_padding_x()
```

**Features:**

- Multi-line editing with word wrap.
- Slash command autocomplete and file path autocomplete.
- Large paste handling.
- Horizontal border lines above and below the editor.
- Fake cursor rendering with optional hardware cursor positioning for IME.
- Grapheme-aware movement through `iter_graphemes`; the port does not use `Intl.Segmenter`.

**Key bindings:** editor bindings are defined in `TUI_KEYBINDINGS`, including submit, newline, tab/autocomplete, line start/end, word navigation, delete commands, undo, yank/yank-pop, and character jump.

### Markdown

Renders Markdown with theming support. The port uses a small built-in tokenizer instead of the TypeScript package's `marked` dependency. It supports headings, emphasis, code blocks, lists, links, blockquotes, tables, task list markers, bare URL/email autolinks, and LaTeX rendering through `render_latex`. It does not include a built-in `highlight.js` equivalent; pass `highlight_code` in the theme if you need syntax highlighting.

```python
from pi_tui import Markdown, MarkdownOptions, MarkdownTheme


def plain(text: str) -> str:
    return text


theme = MarkdownTheme(
    heading=plain,
    link=plain,
    link_url=plain,
    code=plain,
    code_block=plain,
    code_block_border=plain,
    quote=plain,
    quote_border=plain,
    hr=plain,
    list_bullet=plain,
    bold=plain,
    italic=plain,
    strikethrough=plain,
    underline=plain,
    highlight_code=None,
)
md = Markdown("# Hello\n\nSome **bold** text", 1, 1, theme, options=MarkdownOptions())
md.set_text("Updated markdown")
```

### Loader

Animated loading spinner.

```python
from pi_tui import Loader, LoaderIndicatorOptions

loader = Loader(tui, lambda value: value, lambda value: value, "Loading...")
loader.start()
loader.set_message("Still loading...")
loader.set_indicator(LoaderIndicatorOptions(frames=[".", "..", "..."], interval_ms=120))
loader.stop()
```

### CancellableLoader

Extends `Loader` with Escape key handling and an abort controller.

```python
from pi_tui import CancellableLoader

loader = CancellableLoader(tui, lambda value: value, lambda value: value, "Working...")
loader.on_abort = lambda: print("cancelled")
if loader.signal.aborted:
    print("already cancelled")
loader.dispose()
```

### SelectList

Interactive selection list with keyboard navigation.

```python
from pi_tui import SelectItem, SelectList, SelectListTheme


def plain(text: str) -> str:
    return text


items = [
    SelectItem("opt1", "Option 1", "First option"),
    SelectItem("opt2", "Option 2", "Second option"),
]
select_list = SelectList(items, 5, SelectListTheme(plain, plain, plain, plain, plain))
select_list.on_select = lambda item: print("Selected:", item.value)
select_list.on_cancel = lambda: print("Cancelled")
select_list.on_selection_change = lambda item: print("Highlighted:", item.value)
select_list.set_filter("opt")
```

**Controls:** Arrow keys navigate, Enter selects, Escape cancels.

### SettingsList

Settings panel with value cycling, optional fuzzy search, and submenus.

```python
from pi_tui import SettingItem, SettingsList, SettingsListOptions, SettingsListTheme


def style(text: str, selected: bool = False) -> str:
    return text


settings = SettingsList(
    [SettingItem("theme", "Theme", "dark", values=["dark", "light"])],
    10,
    SettingsListTheme(
        label=style,
        value=style,
        description=lambda text: text,
        cursor="> ",
        hint=lambda text: text,
    ),
    lambda name, new_value: print(name, new_value),
    lambda: print("Cancelled"),
    SettingsListOptions(enable_search=True),
)
settings.update_value("theme", "light")
```

**Controls:** Arrow keys navigate, Enter/Space activates, Escape cancels.

### Spacer

Empty lines for vertical spacing.

```python
from pi_tui import Spacer

spacer = Spacer(2)
spacer.set_lines(1)
```

### Image

Renders images inline for terminals that support the Kitty graphics protocol or iTerm2 inline images. Falls back to a text placeholder on unsupported terminals.

```python
from pi_tui import Image, ImageOptions, ImageTheme

image = Image(
    "iVBORw0KGgo=",
    "image/png",
    ImageTheme(fallback_color=lambda text: text),
    ImageOptions(max_width_cells=40, max_height_cells=20, filename="demo.png"),
)
tui.add_child(image)
```

Supported formats for dimension detection are PNG, JPEG, GIF, and WebP.

#### Alternate-screen image compatibility

`TuiAltScreen` supports inline images and partial viewport cropping in terminals that implement the Kitty graphics protocol. iTerm2's inline-image protocol cannot delete or crop existing placements during viewport repainting, so `TuiAltScreen` renders iTerm2 image components as text placeholders. `TuiMainScreen` continues to render iTerm2 inline images normally.

## Autocomplete

### CombinedAutocompleteProvider

Supports slash commands, command argument completions, file paths, and `@` file attachment paths. `get_suggestions()` is async and accepts an `asyncio.Event` cancellation signal.

```python
import asyncio

from pi_tui import AutocompleteItem, CombinedAutocompleteProvider, SlashCommand


async def model_completions(prefix: str):
    return [AutocompleteItem("gpt", "gpt", "Example model")]


provider = CombinedAutocompleteProvider(
    [
        SlashCommand("help", "Show help"),
        SlashCommand("model", "Choose model", "<name>", model_completions),
    ],
    ".",
)


async def load_suggestions() -> None:
    suggestions = await provider.get_suggestions(["/mo"], 0, 3, signal=asyncio.Event())
    if suggestions:
        print([item.value for item in suggestions.items])
```

**Features:**

- Type `/` to see slash commands.
- Press Tab for file path completion.
- Supports `~/`, `./`, `../`, quoted paths, and `@` prefixes.
- Uses `fd` when available, otherwise falls back to Python directory walking.

## Key Detection

Use `matches_key()` with the `Key` helper for keyboard input. Kitty keyboard protocol and legacy terminal sequences are both supported.

```python
from pi_tui import Key, matches_key

if matches_key(data, Key.ctrl("c")):
    raise SystemExit(0)

if matches_key(data, Key.enter):
    submit()
elif matches_key(data, Key.escape):
    cancel()
elif matches_key(data, Key.up):
    move_up()
```

**Key identifiers** include basic keys (`Key.enter`, `Key.escape`, `Key.tab`, `Key.space`, `Key.backspace`, `Key.delete`, `Key.home`, `Key.end`), arrows (`Key.up`, `Key.down`, `Key.left`, `Key.right`), and modifiers (`Key.ctrl("c")`, `Key.shift("tab")`, `Key.alt("left")`, `Key.ctrl_shift("p")`). String IDs such as `"enter"`, `"ctrl+c"`, `"shift+tab"`, and `"ctrl+shift+p"` also work.

## Rendering modes

`TuiMainScreen` uses three rendering strategies:

1. First render: output all lines without clearing scrollback.
2. Width changed or change above viewport: clear screen and fully re-render.
3. Normal update: move to the first changed line, clear to the end, and render changed lines.

`TuiAltScreen` owns a terminal-height viewport. Without a layout root, it preserves the legacy single-document scrolling behavior. With `set_layout_root()`, `VStack`, `HStack`, and nested `ScrollView` components reserve fixed regions and independently scroll constrained regions. It updates changed viewport rows in place, follows streaming output while at the bottom, and preserves a manual scroll position while content grows. Mouse-wheel and configurable keyboard navigation scroll without modifying terminal scrollback. Clicking an OSC 8 hyperlink calls `TuiAltScreenOptions.open_url`. Dragging with the primary mouse button selects text and copies it to the clipboard with OSC 52.

Both renderers wrap updates in synchronized output (`CSI ?2026h` ... `CSI ?2026l`) for atomic rendering.

## Terminal Interface

The TUI works with any object implementing the `Terminal` protocol.

```python
from typing import Protocol


class TerminalLike(Protocol):
    def start(self, on_input, on_resize) -> None: ...
    def stop(self) -> None: ...
    async def drain_input(self, max_ms: int, idle_ms: int) -> None: ...
    def write(self, data: str) -> None: ...
    @property
    def columns(self) -> int: ...
    @property
    def rows(self) -> int: ...
    def move_by(self, lines: int) -> None: ...
    def hide_cursor(self) -> None: ...
    def show_cursor(self) -> None: ...
    def clear_line(self) -> None: ...
    def clear_from_cursor(self) -> None: ...
    def clear_screen(self) -> None: ...
```

Built-in implementation: `ProcessTerminal`, backed by stdin/stdout through `TerminalIo`. The TypeScript `VirtualTerminal` based on `@xterm/headless` is not ported.

## Utilities

```python
from pi_tui import iter_graphemes, truncate_to_width, visible_width, wrap_text_with_ansi

width = visible_width("\x1b[31mHello\x1b[0m")
truncated = truncate_to_width("Hello World", 8)
truncated_no_ellipsis = truncate_to_width("Hello World", 8, "")
lines = wrap_text_with_ansi("This is a long line that needs wrapping", 20)
graphemes = list(iter_graphemes("a\u0301b"))
```

`visible_width()` ignores ANSI and terminal control sequences. `truncate_to_width()` preserves ANSI state and closes styles when truncating. `wrap_text_with_ansi()` preserves styles across line breaks. Grapheme handling goes through `iter_graphemes` in `src/pi_tui/utils.py`; there is no `Intl.Segmenter` in the Python runtime.

## Creating Custom Components

Each returned line from `render()` must not exceed the `width` parameter.

### Handling Input

Use `matches_key()` and `Key` for keyboard input.

```python
from pi_tui import Component, Key, matches_key, truncate_to_width


class MyInteractiveComponent(Component):
    def __init__(self) -> None:
        self.selected_index = 0
        self.items = ["Option 1", "Option 2", "Option 3"]
        self.on_select = None
        self.on_cancel = None

    def handle_input(self, data: str) -> None:
        if matches_key(data, Key.up):
            self.selected_index = max(0, self.selected_index - 1)
        elif matches_key(data, Key.down):
            self.selected_index = min(len(self.items) - 1, self.selected_index + 1)
        elif matches_key(data, Key.enter) and self.on_select:
            self.on_select(self.selected_index)
        elif (matches_key(data, Key.escape) or matches_key(data, Key.ctrl("c"))) and self.on_cancel:
            self.on_cancel()

    def render(self, width: int) -> list[str]:
        result: list[str] = []
        for index, item in enumerate(self.items):
            prefix = "> " if index == self.selected_index else "  "
            result.append(truncate_to_width(prefix + item, width))
        return result
```

### Handling Line Width

Use the provided utilities to ensure lines fit.

```python
from pi_tui import Component, truncate_to_width, visible_width


class MyComponent(Component):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        line = self.text
        visible = visible_width(line)
        if visible > width:
            return [truncate_to_width(line, width)]
        return [line + " " * (width - visible)]
```

### ANSI Code Considerations

Both `visible_width()` and `truncate_to_width()` correctly handle ANSI escape codes.

```python
from pi_tui import truncate_to_width, visible_width

styled = "\x1b[31mHello\x1b[0m " + "\x1b[34mWorld\x1b[0m"
width = visible_width(styled)
truncated = truncate_to_width(styled, 8)
```

### Caching

For performance, components should cache rendered output and clear the cache in `invalidate()`.

```python
from pi_tui import Component, truncate_to_width


class CachedComponent(Component):
    def __init__(self, text: str) -> None:
        self.text = text
        self.cached_width: int | None = None
        self.cached_lines: list[str] | None = None

    def render(self, width: int) -> list[str]:
        if self.cached_lines is not None and self.cached_width == width:
            return self.cached_lines
        lines = [truncate_to_width(self.text, width)]
        self.cached_width = width
        self.cached_lines = lines
        return lines

    def invalidate(self) -> None:
        self.cached_width = None
        self.cached_lines = None
```

## Example

See `packages/pi-coding-agent/docs/tui.md` for the higher-level coding-agent TUI integration. Package tests under `packages/pi-tui/tests/` show focused examples for components, keyboard parsing, rendering utilities, images, and alternate-screen behavior.

## Development

```bash
uv sync --all-packages
uv run pytest packages/pi-tui
uv run ruff check packages/pi-tui
```

### Debug logging

Set `PI_TUI_WRITE_LOG` to capture the raw ANSI stream written to stdout. If the value is a directory, `ProcessTerminal` writes a timestamped log file inside it; otherwise it writes to the exact path.

```bash
PI_TUI_WRITE_LOG=.scratch/tui-ansi.log uv run pp
```
