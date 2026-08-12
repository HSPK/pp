"""Python port of `packages/coding-agent/test/status-indicator.test.ts`.

The TypeScript test uses `vi.useFakeTimers()`. This port installs a fake
`schedule_interval` into both modules that schedule timers for the retry
indicator (the `pi_tui` loader spinner and the countdown timer), which is the
same observation: after `dispose()` nothing may drive another render.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pi_coding_agent.modes.interactive.components import countdown_timer as countdown_timer_module
from pi_coding_agent.modes.interactive.components.status_indicator import IdleStatus, RetryStatusIndicator
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_tui.components import loader as loader_module


class _FakeInterval:
    def __init__(self, clock: _FakeClock, callback: Callable[[], None], interval_s: float) -> None:
        self._clock = clock
        self.callback = callback
        self.interval_s = interval_s
        self.cancelled = False
        self.next_at = clock.now + interval_s

    def cancel(self) -> None:
        self.cancelled = True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.intervals: list[_FakeInterval] = []

    def schedule(self, callback: Callable[[], None], interval_s: float) -> _FakeInterval:
        handle = _FakeInterval(self, callback, interval_s)
        self.intervals.append(handle)
        return handle

    def advance(self, seconds: float) -> None:
        target = self.now + seconds
        while True:
            due = [i for i in self.intervals if not i.cancelled and i.next_at <= target]
            if not due:
                break
            due.sort(key=lambda i: i.next_at)
            handle = due[0]
            self.now = handle.next_at
            handle.next_at += handle.interval_s
            handle.callback()
        self.now = target


@pytest.fixture()
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(loader_module, "schedule_interval", clock.schedule)
    monkeypatch.setattr(countdown_timer_module, "schedule_interval", clock.schedule)
    return clock


class _FakeTui:
    def __init__(self) -> None:
        self.render_calls = 0

    def request_render(self) -> None:
        self.render_calls += 1


def test_keeps_idle_status_at_the_same_height_as_status_indicators():
    idle_status = IdleStatus()

    lines = idle_status.render(20)

    assert len(lines) == 2
    assert lines == [" " * 20, " " * 20]


def test_disposes_retry_countdown_updates(fake_clock: _FakeClock):
    init_theme("dark")
    tui = _FakeTui()
    indicator = RetryStatusIndicator(tui, 1, 3, 1000)
    calls_before_dispose = tui.render_calls

    indicator.dispose()
    fake_clock.advance(2.0)

    assert tui.render_calls == calls_before_dispose
