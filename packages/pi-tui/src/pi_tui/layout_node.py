"""Layout node types for `packages/tui/src/layout-node.ts`.

TypeScript uses a `Symbol.for(...)` key so layout-aware components can expose a
custom layout node without hardcoding concrete component checks. In Python this
port uses the string attribute name stored in `LAYOUT_NODE`; components that
participate in custom layout define a method literally named
`__pi_tui_layout_node__`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from pi_tui.component import Component

LAYOUT_NODE = "__pi_tui_layout_node__"


@dataclass
class LayoutViewport:
    width: int
    height: int


@dataclass
class StackLayoutEntry:
    component: Component
    basis: int | Literal["auto"] | None = None
    grow: int | None = None
    shrink: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass
class StackLayoutNode:
    type: Literal["vstack", "hstack"]
    entries: list[StackLayoutEntry]
    gap: int
    align: Literal["stretch", "start", "center", "end"]


class ScrollLayoutState(Protocol):
    @property
    def scroll_top(self) -> int: ...

    @property
    def primary(self) -> bool: ...

    @property
    def overscroll(self) -> Literal["chain", "contain"]: ...

    @property
    def viewport_height(self) -> int: ...

    def get_content_width(self, width: int) -> int: ...

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None: ...


@dataclass
class ScrollLayoutNode:
    component: Component
    state: ScrollLayoutState
    type: Literal["scroll"] = "scroll"


LayoutNode: TypeAlias = StackLayoutNode | ScrollLayoutNode


def get_layout_node(component: Component) -> LayoutNode | None:
    method = getattr(component, LAYOUT_NODE, None)
    return method() if callable(method) else None
