"""Python port of `packages/coding-agent/test/suite/regressions/7731-tui-method-wrapping.test.ts`.

Regression #7731: `InteractiveMode` hands `this.ui` (and methods pulled off it,
such as `requestRender`) to components that outlive a renderer swap. The fix was
`createInteractiveTuiReference` -- a stable handle that re-resolves attributes
against whatever renderer is current.

Ported as `pi_coding_agent.modes.interactive.interactive_mode.create_interactive_tui_reference`
(a `Proxy` in TypeScript, `__getattr__`/`__setattr__` here).
"""

from __future__ import annotations

from typing import Any

from pi_coding_agent.modes.interactive.interactive_mode import create_interactive_tui_reference


class _RenderOnly:
    def render(self, width: int) -> list[str]:
        return [f"width: {width}"]


class _RequestRenderOnly:
    def __init__(self) -> None:
        self.calls = 0

    def request_render(self) -> None:
        self.calls += 1


def test_calls_the_method_captured_before_a_replacement() -> None:
    """The monkey-patch idiom must not recurse forever.

    `const originalRender = tui.render; tui.render = (w) => originalRender(w)`
    writes through to the renderer, so a naive re-resolving wrapper would find
    the *patched* function again and loop. `originalRender` stays bound to the
    function that was live when it was captured.
    """
    renderer = _RenderOnly()
    tui = create_interactive_tui_reference(lambda: renderer)
    original_render = tui.render
    tui.render = lambda width: original_render(width)

    assert tui.render(80) == ["width: 80"]


def test_routes_a_captured_method_to_a_replacement_renderer() -> None:
    """A captured method follows the renderer when the renderer identity changes."""
    regular = _RequestRenderOnly()
    fullscreen = _RequestRenderOnly()
    current: list[Any] = [regular]
    tui = create_interactive_tui_reference(lambda: current[0])
    request_render = tui.request_render

    request_render()
    current[0] = fullscreen
    request_render()

    assert regular.calls == 1
    assert fullscreen.calls == 1


def test_non_callable_attributes_read_through_to_the_current_renderer() -> None:
    """`Reflect.get` on a non-function returns the value unwrapped."""
    first = _RenderOnly()
    first.label = "first"  # type: ignore[attr-defined]
    second = _RenderOnly()
    second.label = "second"  # type: ignore[attr-defined]
    current: list[Any] = [first]
    tui = create_interactive_tui_reference(lambda: current[0])

    assert tui.label == "first"
    current[0] = second
    assert tui.label == "second"


def test_writes_go_through_to_the_current_renderer() -> None:
    """The TypeScript `set` trap is `Reflect.set(tui, property, value, tui)`."""
    renderer = _RenderOnly()
    tui = create_interactive_tui_reference(lambda: renderer)

    tui.label = "written"

    assert renderer.label == "written"  # type: ignore[attr-defined]
