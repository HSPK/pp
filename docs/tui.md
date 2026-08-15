> pp can create TUI components for Python code that runs inside the ported interactive UI.

# TUI Components

Python code can build terminal UI components with `pi_tui`. The interactive mode in `pp` uses these components directly. Extension custom TUI mounting (`ctx.ui.custom()`), editor widgets, custom footers, custom headers, extension-driven dialogs, and terminal input listeners are not available in the Python extension API; extension UI is limited to `select`, `confirm`, `input`, `notify`, `set_status`, `set_title`, and tool expansion controls.

**Source:** `packages/pi-tui/src/pi_tui/`

## Component Interface

All components subclass or implement `pi_tui.Component`:

```python
from pi_tui import Component


class MyComponent(Component):
    wants_key_release = False

    def render(self, width: int) -> list[str]:
        return ["content"[:width]]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        pass
```

| Method / attribute | Description |
|--------|-------------|
| `render(width)` | Return a list of strings, one per line. Each rendered line must fit `width` visible cells. |
| `handle_input(data)` | Optional method. Receive keyboard input when the component has focus. |
| `wants_key_release` | Optional boolean attribute. If true, the component receives Kitty key release events. Default: absent/false. |
| `invalidate()` | Clear cached render state. Called on theme changes and explicit invalidation. |

The TUI appends line resets when it writes component output. Styles do not safely carry across lines. Reapply styles per line or use `wrap_text_with_ansi()`.

## Focusable Interface (IME Support)

Components that display a text cursor and need IME (Input Method Editor) support should implement the `Focusable` protocol by exposing a `focused: bool` attribute and emitting `CURSOR_MARKER` immediately before the fake cursor.

```python
from pi_tui import CURSOR_MARKER, Component, truncate_to_width


class SearchBox(Component):
    def __init__(self) -> None:
        self.focused = False
        self.text = ""
        self.cursor = 0

    def render(self, width: int) -> list[str]:
        before = self.text[: self.cursor]
        at = self.text[self.cursor : self.cursor + 1] or " "
        after = self.text[self.cursor + 1 :]
        marker = CURSOR_MARKER if self.focused else ""
        line = f"> {before}{marker}\x1b[7m{at}\x1b[27m{after}"
        return [truncate_to_width(line, width)]

    def invalidate(self) -> None:
        pass
```

When a focusable component has focus, the TUI:

1. Sets `focused = True` on the component.
2. Scans rendered output for `CURSOR_MARKER`.
3. Positions the hardware terminal cursor at that marker.
4. Shows the hardware cursor only when `showHardwareCursor` is enabled.

The cursor remains hidden by default. Enable a visible hardware cursor with the `showHardwareCursor` setting or `PI_HARDWARE_CURSOR=1`. The built-in `Editor` and `Input` components already implement this protocol.

## TUI Modes

The interactive TUI supports `regular` and `fullscreen` modes. Select them with the `tuiMode` setting or `--tui-mode`. Runtime switching between the two modes is not ported; changing the setting applies on the next start.

### Container Components with Embedded Inputs

When a container component contains an `Input` or `Editor` child, propagate the focus state to the child. Otherwise IME candidate windows can appear at the wrong screen position.

```python
from pi_tui import Container, Input


class SearchDialog(Container):
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

## Using Components

Use components directly from Python code that owns a `TuiMainScreen`, `TuiAltScreen`, or other `TuiBase` instance. The TypeScript extension API examples that call `ctx.ui.custom()` do not apply to the Python port.

```python
from pi_tui import Container, Text, TuiMainScreen, ProcessTerminal


ui = TuiMainScreen(ProcessTerminal())
root = Container()
root.add_child(Text("Hello from pi_tui", padding_x=1, padding_y=0))
ui.add_child(root)
ui.set_focus(root)
ui.request_render()
```

`TuiBase.add_child()`, `remove_child()`, `clear()`, `set_focus()`, `request_render()`, and `show_overlay()` are the low-level APIs used by the interactive mode.

## Overlays

Overlays are available on `TuiBase` through `show_overlay(component, options)`. The Python API uses dataclasses instead of JavaScript option objects.

```python
from pi_tui import OverlayOptions, Text
from pi_tui.tui import OverlayMargin


def show_panel(tui) -> None:
    panel = Text("Panel", padding_x=1, padding_y=1)
    handle = tui.show_overlay(
        panel,
        OverlayOptions(
            width="50%",
            min_width=40,
            max_height="80%",
            anchor="right-center",
            offset_x=-2,
            margin=OverlayMargin(top=1, right=2, bottom=1, left=2),
            visible=lambda term_width, term_height: term_width >= 80,
        ),
    )
    handle.focus()
```

### Overlay Focus

A focused visible overlay owns input until it is hidden, removed, or unfocused. `OverlayHandle.unfocus()` can release focus to fallback input handling or to a specific target component.

### Overlay Lifecycle

Overlay components are disposed when closed. Do not keep and reuse component references after `handle.hide()`; create a fresh component when showing the overlay again.

## Built-in Components

Import public components from `pi_tui`:

```python
from pi_tui import Text, Box, Container, Spacer, Markdown, Image
```

### Text

Multi-line text with word wrapping.

```python
from pi_tui import Text


text = Text("Hello World", padding_x=1, padding_y=1)
text.set_text("Updated")
```

### Box

Container with padding and an optional background function.

```python
from pi_tui import Box, Text


def bg_gray(text: str) -> str:
    return f"\x1b[48;5;240m{text}\x1b[49m"


box = Box(padding_x=1, padding_y=1, bg_fn=bg_gray)
box.add_child(Text("Content", padding_x=0, padding_y=0))
box.set_bg_fn(lambda text: f"\x1b[44m{text}\x1b[49m")
```

### Container

Groups child components vertically.

```python
from pi_tui import Container, Text


container = Container()
child = Text("Content", padding_x=0, padding_y=0)
container.add_child(child)
container.remove_child(child)
container.clear()
```

### Spacer

Empty vertical space.

```python
from pi_tui import Spacer


spacer = Spacer(2)
```

### Markdown

Renders Markdown with the ported Markdown renderer and LaTeX support.

```python
from pi_tui import Markdown
from pi_coding_agent.modes.interactive.theme.theme import get_markdown_theme, init_theme


init_theme("dark")
markdown = Markdown("# Title\n\nSome **bold** text", 1, 1, get_markdown_theme())
markdown.set_text("Updated markdown")
```

Mermaid diagram rendering is not available in the Python port.

### Image

Renders base64 images in supported terminals (Kitty, iTerm2, Ghostty, WezTerm, Warp) and falls back to text elsewhere.

```python
from pi_tui import Image, ImageOptions, ImageTheme


def muted(text: str) -> str:
    return f"\x1b[2m{text}\x1b[22m"


image = Image(
    base64_data="iVBORw0KGgo=",
    mime_type="image/png",
    theme=ImageTheme(fallback_color=muted),
    options=ImageOptions(max_width_cells=80, max_height_cells=24),
)
```

## Keyboard Input

Use `matches_key()` for key detection:

```python
from pi_tui import Key, matches_key


class MyKeys:
    def __init__(self) -> None:
        self.selected_index = 0
        self.cancelled = False

    def handle_input(self, data: str) -> None:
        if matches_key(data, Key.up):
            self.selected_index -= 1
        elif matches_key(data, Key.enter):
            self.selected_index = 0
        elif matches_key(data, Key.escape):
            self.cancelled = True
        elif matches_key(data, Key.ctrl("c")):
            self.cancelled = True
```

**Key identifiers** (use `Key.*` for autocomplete, or string literals):

- Basic keys: `Key.enter`, `Key.escape`, `Key.tab`, `Key.space`, `Key.backspace`, `Key.delete`, `Key.home`, `Key.end`
- Arrow keys: `Key.up`, `Key.down`, `Key.left`, `Key.right`
- With modifiers: `Key.ctrl("c")`, `Key.shift("tab")`, `Key.alt("left")`, `Key.ctrl_shift("p")`
- String format also works: `"enter"`, `"ctrl+c"`, `"shift+tab"`, `"ctrl+shift+p"`

## Line Width

Each line from `render()` must fit the `width` parameter.

```python
from pi_tui import Component, truncate_to_width, visible_width, wrap_text_with_ansi


class SingleLine(Component):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        return [truncate_to_width(self.text, width)]

    def invalidate(self) -> None:
        pass
```

Utilities:

- `visible_width(text)` - display width excluding terminal escape sequences.
- `truncate_to_width(text, max_width, ellipsis="...", pad=False)` - ANSI-aware truncation.
- `wrap_text_with_ansi(text, width)` - word wrap preserving ANSI SGR and OSC 8 state.

## Creating Custom Components

Example: interactive selector.

```python
from collections.abc import Callable

from pi_tui import Component, Key, matches_key, truncate_to_width


class MySelector(Component):
    def __init__(self, items: list[str]) -> None:
        self.items = items
        self.selected = 0
        self.cached_width: int | None = None
        self.cached_lines: list[str] | None = None
        self.on_select: Callable[[str], None] | None = None
        self.on_cancel: Callable[[], None] | None = None

    def handle_input(self, data: str) -> None:
        if matches_key(data, Key.up) and self.selected > 0:
            self.selected -= 1
            self.invalidate()
        elif matches_key(data, Key.down) and self.selected < len(self.items) - 1:
            self.selected += 1
            self.invalidate()
        elif matches_key(data, Key.enter) and self.on_select is not None:
            self.on_select(self.items[self.selected])
        elif matches_key(data, Key.escape) and self.on_cancel is not None:
            self.on_cancel()

    def render(self, width: int) -> list[str]:
        if self.cached_lines is not None and self.cached_width == width:
            return self.cached_lines

        self.cached_lines = [
            truncate_to_width(("> " if i == self.selected else "  ") + item, width) for i, item in enumerate(self.items)
        ]
        self.cached_width = width
        return self.cached_lines

    def invalidate(self) -> None:
        self.cached_width = None
        self.cached_lines = None
```

Use `tui.request_render()` after state changes when a component is live in a TUI.

## Theming

The interactive theme helpers live in `pi_coding_agent.modes.interactive.theme.theme`.

```python
from pi_coding_agent.modes.interactive.theme.theme import init_theme, theme


init_theme("dark")
styled = theme.fg("success", "Done")
background = theme.bg("toolPendingBg", theme.fg("accent", "text"))
```

**Foreground colors** (`theme.fg(color, text)`):

| Category | Colors |
|----------|--------|
| General | `text`, `accent`, `muted`, `dim` |
| Status | `success`, `error`, `warning` |
| Borders | `border`, `borderAccent`, `borderMuted` |
| Messages | `userMessageText`, `customMessageText`, `customMessageLabel` |
| Tools | `toolTitle`, `toolOutput` |
| Diffs | `toolDiffAdded`, `toolDiffRemoved`, `toolDiffContext` |
| Markdown | `mdHeading`, `mdLink`, `mdLinkUrl`, `mdCode`, `mdCodeBlock`, `mdCodeBlockBorder`, `mdQuote`, `mdQuoteBorder`, `mdHr`, `mdListBullet` |
| Syntax | `syntaxComment`, `syntaxKeyword`, `syntaxFunction`, `syntaxVariable`, `syntaxString`, `syntaxNumber`, `syntaxType`, `syntaxOperator`, `syntaxPunctuation` |
| Thinking | `thinkingOff`, `thinkingMinimal`, `thinkingLow`, `thinkingMedium`, `thinkingHigh`, `thinkingXhigh`, `thinkingMax` |
| Modes | `bashMode` |

**Background colors** (`theme.bg(color, text)`):

`selectedBg`, `scrollbarThumb`, `userMessageBg`, `customMessageBg`, `toolPendingBg`, `toolSuccessBg`, `toolErrorBg`

For Markdown, use `get_markdown_theme()`:

```python
from pi_tui import Markdown
from pi_coding_agent.modes.interactive.theme.theme import get_markdown_theme, init_theme


init_theme("dark")
markdown = Markdown("**details**", 0, 0, get_markdown_theme())
```

## Debug logging

Set `PI_TUI_WRITE_LOG` to capture the raw ANSI stream written by the terminal driver.

```bash
PI_TUI_WRITE_LOG=./tui-ansi.log uv run pp -i
```

## Performance

Cache rendered output when possible:

```python
from pi_tui import Component


class CachedComponent(Component):
    def __init__(self) -> None:
        self.cached_width: int | None = None
        self.cached_lines: list[str] | None = None

    def render(self, width: int) -> list[str]:
        if self.cached_lines is not None and self.cached_width == width:
            return self.cached_lines
        lines = ["computed"]
        self.cached_width = width
        self.cached_lines = lines
        return lines

    def invalidate(self) -> None:
        self.cached_width = None
        self.cached_lines = None
```

Call `invalidate()` when state changes, then call `tui.request_render()`.

## Invalidation and Theme Changes

When the theme changes, the TUI invalidates components to clear cached render state. Components must rebuild any strings that bake in theme colors.

### The Problem

If a component stores strings already styled with `theme.fg()` or `theme.bg()`, clearing the render cache is not enough: the stored string still contains ANSI escapes from the old theme.

**Wrong approach** (theme colors will not update):

```python
from pi_tui import Container, Text
from pi_coding_agent.modes.interactive.theme.theme import theme


class BadComponent(Container):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.content = Text(theme.fg("accent", message), 1, 0)
        self.add_child(self.content)
```

### The Solution

Rebuild themed content in `invalidate()`:

```python
from pi_tui import Container, Text
from pi_coding_agent.modes.interactive.theme.theme import theme


class GoodComponent(Container):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message
        self.content = Text("", 1, 0)
        self.add_child(self.content)
        self.update_display()

    def update_display(self) -> None:
        self.content.set_text(theme.fg("accent", self.message))

    def invalidate(self) -> None:
        super().invalidate()
        self.update_display()
```

### Pattern: Rebuild on Invalidate

For complex layouts, clear and rebuild the child tree:

```python
from dataclasses import dataclass

from pi_tui import Container, Spacer, Text
from pi_coding_agent.modes.interactive.theme.theme import theme


@dataclass
class Item:
    label: str
    active: bool


class ComplexComponent(Container):
    def __init__(self, items: list[Item]) -> None:
        super().__init__()
        self.items = items
        self.rebuild()

    def rebuild(self) -> None:
        self.clear()
        self.add_child(Text(theme.fg("accent", theme.bold("Title")), 1, 0))
        self.add_child(Spacer(1))
        for item in self.items:
            color = "success" if item.active else "muted"
            self.add_child(Text(theme.fg(color, item.label), 1, 0))

    def invalidate(self) -> None:
        super().invalidate()
        self.rebuild()
```

### When This Matters

This pattern is needed when:

1. Pre-baking theme colors with `theme.fg()` or `theme.bg()`.
2. Syntax highlighting or other code that stores themed strings.
3. Building child component trees that embed theme colors.

It is not needed when:

1. Computing themed output fresh in every `render()` call.
2. Passing callbacks that apply colors during render.
3. Grouping components without adding themed content.

## Common Patterns

These patterns cover common UI needs in the Python port. Extension mounting APIs from the TypeScript docs are unavailable unless noted.

### Pattern 1: Selection Dialog (SelectList)

Use `SelectList` from `pi_tui` with `DynamicBorder` for framing.

```python
from pi_tui import Container, SelectItem, SelectList, Text
from pi_coding_agent.modes.interactive.components.dynamic_border import DynamicBorder
from pi_coding_agent.modes.interactive.theme.theme import get_select_list_theme, theme


class PickDialog(Container):
    def __init__(self) -> None:
        super().__init__()
        items = [
            SelectItem(value="opt1", label="Option 1", description="First option"),
            SelectItem(value="opt2", label="Option 2", description="Second option"),
            SelectItem(value="opt3", label="Option 3"),
        ]
        self.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))
        self.add_child(Text(theme.fg("accent", theme.bold("Pick an Option")), 1, 0))
        self.select_list = SelectList(items, min(len(items), 10), get_select_list_theme())
        self.select_list.on_select = lambda item: None
        self.select_list.on_cancel = lambda: None
        self.add_child(self.select_list)
        self.add_child(Text(theme.fg("dim", "↑↓ navigate • enter select • esc cancel"), 1, 0))
        self.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))

    def handle_input(self, data: str) -> None:
        self.select_list.handle_input(data)
```

### Pattern 2: Async Operation with Cancel (BorderedLoader)

`BorderedLoader` exists in the Python port for internal interactive UI. It requires a live `TuiBase` instance and the current theme object.

```python
from pi_coding_agent.modes.interactive.components.bordered_loader import BorderedLoader
from pi_coding_agent.modes.interactive.theme.theme import theme


def make_loader(tui) -> BorderedLoader:
    loader = BorderedLoader(tui, theme, "Fetching data...")
    loader.on_abort = lambda: None
    return loader
```

### Pattern 3: Settings/Toggles (SettingsList)

Use `SettingsList` with `get_settings_list_theme()`.

```python
from pi_tui import Container, SettingItem, SettingsList, SettingsListOptions, Text
from pi_coding_agent.modes.interactive.theme.theme import get_settings_list_theme, theme


class SettingsDialog(Container):
    def __init__(self) -> None:
        super().__init__()
        items = [
            SettingItem(id="verbose", label="Verbose mode", current_value="off", values=["on", "off"]),
            SettingItem(id="color", label="Color output", current_value="on", values=["on", "off"]),
        ]
        self.add_child(Text(theme.fg("accent", theme.bold("Settings")), 1, 1))
        self.settings_list = SettingsList(
            items,
            min(len(items) + 2, 15),
            get_settings_list_theme(),
            lambda setting_id, new_value: None,
            lambda: None,
            SettingsListOptions(enable_search=True),
        )
        self.add_child(self.settings_list)

    def handle_input(self, data: str) -> None:
        self.settings_list.handle_input(data)
```

### Pattern 4: Persistent Status Indicator

The Python extension UI supports status text in the footer:

```python
from pi_coding_agent.core.extensions.types import ExtensionContext


def set_mode_status(ctx: ExtensionContext) -> None:
    ctx.ui.set_status("my-ext", "active")
    ctx.ui.set_status("my-ext", None)
```

### Pattern 4b: Working Indicator Customization

`ctx.ui.set_working_indicator()` is not available in the Python extension API.

### Pattern 5: Widgets Above/Below Editor

`ctx.ui.set_widget()` is not available in the Python extension API.

### Pattern 6: Custom Footer

`ctx.ui.set_footer()` is not available in the Python extension API.

### Pattern 7: Custom Editor (vim mode, etc.)

`CustomEditor` is ported and used by the interactive mode, but `ctx.ui.set_editor_component()` is not available to Python extensions.

```python
from pi_coding_agent.modes.interactive.components.custom_editor import CustomEditor
from pi_tui import matches_key, truncate_to_width


class VimEditor(CustomEditor):
    def __init__(self, tui, theme, keybindings) -> None:
        super().__init__(tui, theme, keybindings)
        self.mode = "insert"

    def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") and self.mode == "insert":
            self.mode = "normal"
            return
        if self.mode == "insert":
            super().handle_input(data)
            return
        if data == "i":
            self.mode = "insert"
            return
        if data == "h":
            super().handle_input("\x1b[D")
            return
        super().handle_input(data)

    def render(self, width: int) -> list[str]:
        lines = super().render(width)
        if lines:
            label = " NORMAL " if self.mode == "normal" else " INSERT "
            lines[-1] = truncate_to_width(lines[-1], width - len(label), "") + label
        return lines
```

## Key Rules

1. Use Python names: `handle_input`, `on_select`, `padding_x`, `max_width_cells`, `get_markdown_theme()`.
2. Keep every rendered line within `width` visible cells.
3. Call `invalidate()` after state changes and `tui.request_render()` for live UI.
4. Prefer `SelectList`, `SettingsList`, `BorderedLoader`, `Text`, `Box`, and `Markdown` over rebuilding them.
5. Do not use TypeScript-only extension APIs such as `ctx.ui.custom()`, widgets, custom footers, or custom editor factories in the Python port.

## Examples

Python examples in this package currently focus on extension lifecycle and commands, not custom TUI components:

- [examples/extensions/trigger_compact.py](../examples/extensions/trigger_compact.py)
- [examples/extensions/input_transform_streaming.py](../examples/extensions/input_transform_streaming.py)
- [examples/extensions/git_merge_and_resolve.py](../examples/extensions/git_merge_and_resolve.py)

For component usage, read the implementation under `packages/pi-coding-agent/src/pi_coding_agent/modes/interactive/components/` and `packages/pi-tui/src/pi_tui/components/`.
