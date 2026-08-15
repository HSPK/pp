"""Python port of `packages/coding-agent/test/interactive-mode-compaction.test.ts`.

The TypeScript test invokes the private methods through
`Reflect.get(InteractiveMode.prototype, ...)` bound to a hand-built `this`.
The Python analogue is calling the unbound function with a stand-in object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi_coding_agent.modes.interactive.interactive_mode import (
    CompactionQueuedMessage,
    InteractiveMode,
)


class _Recorder:
    def __init__(self, result: Any = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._result


class _AsyncRecorder(_Recorder):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.calls.append((args, kwargs))
        return self._result


async def _noop() -> None:
    return None


class _SpawnRecorder(_Recorder):
    """Records at call time (like `vi.fn()`) and returns an awaitable for `spawn`."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return _noop()


@dataclass
class _CompactionResult:
    tokens_before: int
    summary: str


class _CompactionEndEvent:
    type = "compaction_end"

    def __init__(
        self,
        *,
        reason: str,
        result: _CompactionResult | None,
        aborted: bool,
        will_retry: bool,
        error_message: str | None = None,
    ) -> None:
        self.reason = reason
        self.result = result
        self.aborted = aborted
        self.will_retry = will_retry
        self.error_message = error_message


class _FakeTerminal:
    def __init__(self) -> None:
        self.set_progress = _Recorder()


class _FakeUi:
    def __init__(self) -> None:
        self.request_render = _Recorder()
        self.terminal = _FakeTerminal()


class _FakeSettingsManager:
    def get_show_terminal_progress(self) -> bool:
        return False


class _FakeContainer:
    def __init__(self) -> None:
        self.clear = _Recorder()
        self.add_child = _Recorder()


class _CompactionEndThis:
    """Stand-in `self` for `handle_event`'s `compaction_end` branch."""

    _handle_event = InteractiveMode._handle_event

    def __init__(self) -> None:
        self.is_initialized = True
        self.footer = type("_Footer", (), {"invalidate": _Recorder()})()
        self.auto_compaction_escape_handler: Any = None
        self.default_editor = type("_Editor", (), {"on_escape": None})()
        self.status_container = _FakeContainer()
        self.chat_container = _FakeContainer()
        self._rebuild_chat_from_messages = _Recorder()
        self._add_message_to_chat = _Recorder()
        self.show_error = _Recorder()
        self.show_status = _Recorder()
        self._clear_status_indicator = _Recorder()
        self._flush_compaction_queue = _SpawnRecorder()
        self.settings_manager = _FakeSettingsManager()
        self.ui = _FakeUi()


class _FakeSession:
    def __init__(self) -> None:
        self.clear_queue = _Recorder()
        self.prompt = _AsyncRecorder()
        self.steer = _AsyncRecorder()
        self.follow_up = _AsyncRecorder()


class _FlushThis:
    """Stand-in `self` for `_flush_compaction_queue`."""

    _flush_compaction_queue = InteractiveMode._flush_compaction_queue

    def __init__(self) -> None:
        self.compaction_queued_messages = [CompactionQueuedMessage(text="change direction", mode="steer")]
        self.session = _FakeSession()
        self._is_extension_command = _Recorder(False)
        self._update_pending_messages_display = _Recorder()
        self.show_error = _Recorder()


async def test_rebuilds_chat_and_appends_synthetic_compaction_summary() -> None:
    fake = _CompactionEndThis()

    await fake._handle_event(
        _CompactionEndEvent(
            reason="manual",
            result=_CompactionResult(tokens_before=123, summary="summary"),
            aborted=False,
            will_retry=False,
        )
    )

    assert len(fake.chat_container.clear.calls) == 1
    assert len(fake._rebuild_chat_from_messages.calls) == 1
    assert len(fake._add_message_to_chat.calls) == 1
    message = fake._add_message_to_chat.calls[0][0][0]
    assert message.role == "compactionSummary"
    assert message.tokens_before == 123
    assert message.summary == "summary"
    assert fake._flush_compaction_queue.calls == [((), {"will_retry": False})]


async def test_preserves_steering_behavior_when_flushing_into_active_run() -> None:
    fake = _FlushThis()

    await fake._flush_compaction_queue(will_retry=False)

    assert fake.session.prompt.calls == [(("change direction",), {"streaming_behavior": "steer"})]
    assert fake.compaction_queued_messages == []
    assert fake.show_error.calls == []
