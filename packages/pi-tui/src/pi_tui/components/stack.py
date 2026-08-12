"""Stack layout primitives for `packages/tui/src/components/stack.ts`."""

from __future__ import annotations

import math
import sys
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pi_tui.component import Component, Container
from pi_tui.layout_node import LayoutViewport, StackLayoutEntry, StackLayoutNode


@dataclass
class StackEntryOptions:
    basis: int | Literal["auto"] | None = None
    grow: int | None = None
    shrink: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass
class StackEntry(StackEntryOptions):
    component: Component | None = None


StackChild: TypeAlias = Component | StackEntry


@dataclass
class StackOptions:
    gap: int | None = None
    align: Literal["stretch", "start", "center", "end"] | None = None


def _normalize_size(value: object, fallback: int) -> int:
    if not isinstance(value, int | float) or not math.isfinite(value):
        return fallback
    return max(0, math.floor(value))


class Stack(Container, ABC):
    layout_type: Literal["vstack", "hstack"]

    def __init__(self, children: list[StackChild] | None = None, options: StackOptions | None = None) -> None:
        super().__init__()
        options = options or StackOptions()
        self.entries: list[StackLayoutEntry] = []
        self.gap = _normalize_size(options.gap, 0)
        self.align = options.align or "stretch"
        for child in children or []:
            if isinstance(child, StackEntry):
                if child.component is None:
                    raise ValueError("StackEntry.component is required")
                self.add_child(child.component, child)
            else:
                self.add_child(child)

    def add_child(self, component: Component, options: StackEntryOptions | None = None) -> None:
        super().add_child(component)
        options = options or StackEntryOptions()
        self.entries.append(
            StackLayoutEntry(
                component=component,
                basis=options.basis,
                grow=None if options.grow is None else _normalize_size(options.grow, 0),
                shrink=None if options.shrink is None else _normalize_size(options.shrink, 1),
                min_size=None if options.min_size is None else _normalize_size(options.min_size, 0),
                max_size=None if options.max_size is None else _normalize_size(options.max_size, sys.maxsize),
                visible=options.visible,
            )
        )

    def remove_child(self, component: Component) -> None:
        super().remove_child(component)
        index = next((i for i, entry in enumerate(self.entries) if entry.component is component), -1)
        if index != -1:
            self.entries.pop(index)

    def clear(self) -> None:
        super().clear()
        self.entries.clear()

    def __pi_tui_layout_node__(self) -> StackLayoutNode:
        return StackLayoutNode(
            type=self.layout_type,
            entries=self.entries,
            gap=self.gap,
            align=self.align,
        )


def visible_stack_entries(
    entries: list[StackLayoutEntry],
    viewport: LayoutViewport,
) -> list[StackLayoutEntry]:
    return [entry for entry in entries if (entry.visible(viewport) if entry.visible is not None else True)]


def clamp_size(size: int, entry: StackLayoutEntry) -> int:
    minimum = max(0, math.floor(entry.min_size or 0))
    maximum = max(minimum, math.floor(entry.max_size if entry.max_size is not None else sys.maxsize))
    return max(minimum, min(maximum, max(0, math.floor(size))))


def distribute(
    sizes: list[int],
    entries: list[StackLayoutEntry],
    amount: int,
    mode: Literal["grow", "shrink"],
) -> None:
    remaining = amount
    while remaining > 0:
        candidates: list[tuple[StackLayoutEntry, int]] = []
        for index, entry in enumerate(entries):
            if mode == "grow":
                if (entry.grow or 0) > 0 and sizes[index] < (entry.max_size or sys.maxsize):
                    candidates.append((entry, index))
            elif (entry.shrink if entry.shrink is not None else 1) > 0 and sizes[index] > (entry.min_size or 0):
                candidates.append((entry, index))
        if not candidates:
            return

        total_weight = 0
        for entry, index in candidates:
            if mode == "grow":
                total_weight += entry.grow or 0
            else:
                total_weight += (entry.shrink if entry.shrink is not None else 1) * max(1, sizes[index])

        distributed = 0
        for entry, index in candidates:
            if remaining <= 0:
                break
            if mode == "grow":
                weight = entry.grow or 0
                capacity = (entry.max_size or sys.maxsize) - sizes[index]
            else:
                weight = (entry.shrink if entry.shrink is not None else 1) * max(1, sizes[index])
                capacity = sizes[index] - (entry.min_size or 0)
            proposed = max(1, math.floor((remaining * weight) / total_weight))
            delta = min(remaining, proposed, capacity)
            if delta <= 0:
                continue
            sizes[index] += delta if mode == "grow" else -delta
            remaining -= delta
            distributed += delta
        if distributed == 0:
            return


def allocate_stack_sizes(
    entries: list[StackLayoutEntry],
    intrinsic_sizes: list[int],
    available_size: int | None,
    gap: int,
) -> list[int]:
    sizes = [
        clamp_size(
            intrinsic_sizes[index] if entry.basis in (None, "auto") else int(entry.basis),
            entry,
        )
        for index, entry in enumerate(entries)
    ]
    if available_size is None:
        return sizes

    content_size = max(0, math.floor(available_size) - max(0, len(entries) - 1) * gap)
    total = sum(sizes)
    if total < content_size:
        distribute(sizes, entries, content_size - total, "grow")
    elif total > content_size:
        distribute(sizes, entries, total - content_size, "shrink")
    return sizes
