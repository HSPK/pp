"""Startup profiling instrumentation, enabled with the `PI_TIMING=1` env var.

Ported from ``packages/coding-agent/src/core/timings.ts``.

Records elapsed time between successive `time(label)` calls within a named
namespace (e.g. "main" for CLI startup, "extensions" for extension loading),
then prints a summary table to stderr. Disabled entirely (all functions are
no-ops) unless `PI_TIMING=1` is set, so normal runs pay no cost.
"""

from __future__ import annotations

import os
import sys
import time as _time
from dataclasses import dataclass, field

_ENABLED = os.environ.get("PI_TIMING") == "1"

TimingLabel = str
"""`"main"` | `"extensions"`, or any caller-chosen namespace name."""


@dataclass
class _Timing:
    label: str
    ms: int


@dataclass
class _TimingNamespace:
    timings: list[_Timing] = field(default_factory=list)
    last_time: int = 0


_timing_namespaces: dict[TimingLabel, _TimingNamespace] = {}


def _now_ms() -> int:
    """Port of `Date.now()`: whole milliseconds since the epoch, wall clock."""
    return int(_time.time() * 1000)


def reset_timings(namespace: TimingLabel = "main") -> None:
    if not _ENABLED:
        return
    _timing_namespaces[namespace] = _TimingNamespace(timings=[], last_time=_now_ms())


def time(label: str, namespace: TimingLabel = "main") -> None:
    if not _ENABLED:
        return
    now = _now_ms()

    if namespace not in _timing_namespaces:
        reset_timings(namespace)

    timing_namespace = _timing_namespaces[namespace]
    timing_namespace.timings.append(_Timing(label=label, ms=now - timing_namespace.last_time))
    timing_namespace.last_time = now


def _print_timing_group(title: str, timings: list[_Timing]) -> None:
    printable_timings = [timing for timing in timings if timing.ms >= 0]
    if not printable_timings:
        return
    print(f"\n--- {title} ---", file=sys.stderr)
    for timing in printable_timings:
        print(f"  {timing.label}: {timing.ms}ms", file=sys.stderr)
    total = sum(timing.ms for timing in printable_timings)
    print(f"  TOTAL: {total}ms", file=sys.stderr)
    print(f"{'-' * (len(title) + 8)}\n", file=sys.stderr)


def print_timings() -> None:
    if not _ENABLED:
        return
    for namespace, timing_namespace in _timing_namespaces.items():
        _print_timing_group(f"Startup Timings: {namespace}", timing_namespace.timings)
