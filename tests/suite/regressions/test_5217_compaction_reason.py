"""Python port of `packages/coding-agent/test/suite/regressions/5217-compaction-reason.test.ts`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from harness import Harness, create_harness
from pi_ai.providers.faux import faux_assistant_message

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    CompactionResult,
    SessionBeforeCompactResult,
)


@dataclass
class RecordedCompactionEvent:
    type: Literal["session_before_compact", "session_compact"]
    reason: Literal["manual", "threshold", "overflow"]
    will_retry: bool


def _recording_extension(recorded: list[RecordedCompactionEvent]):
    def factory(pi: ExtensionAPI) -> None:
        async def on_before_compact(event, _ctx) -> SessionBeforeCompactResult:
            recorded.append(RecordedCompactionEvent(event.type, event.reason, event.will_retry))
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary="summary from extension",
                    first_kept_entry_id=event.preparation.first_kept_entry_id,
                    tokens_before=event.preparation.tokens_before,
                    details={},
                )
            )

        async def on_compact(event, _ctx) -> None:
            recorded.append(RecordedCompactionEvent(event.type, event.reason, event.will_retry))

        pi.on("session_before_compact", on_before_compact)
        pi.on("session_compact", on_compact)

    return factory


async def _create_compaction_harness(tmp_path: Path, recorded: list[RecordedCompactionEvent]) -> Harness:
    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[_recording_extension(recorded)],
    )
    harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
    await harness.session.prompt("first")
    await harness.session.prompt("second")
    return harness


async def test_reports_manual_reason_for_compact(tmp_path: Path) -> None:
    recorded: list[RecordedCompactionEvent] = []
    harness = await _create_compaction_harness(tmp_path, recorded)
    try:
        await harness.session.compact()

        assert recorded == [
            RecordedCompactionEvent("session_before_compact", "manual", False),
            RecordedCompactionEvent("session_compact", "manual", False),
        ]
    finally:
        harness.cleanup()


async def test_reports_threshold_reason_for_auto_compaction(tmp_path: Path) -> None:
    recorded: list[RecordedCompactionEvent] = []
    harness = await _create_compaction_harness(tmp_path, recorded)
    try:
        await harness.session._run_auto_compaction("threshold", False)

        assert recorded == [
            RecordedCompactionEvent("session_before_compact", "threshold", False),
            RecordedCompactionEvent("session_compact", "threshold", False),
        ]
    finally:
        harness.cleanup()


async def test_reports_overflow_reason_and_will_retry_for_overflow_recovery(tmp_path: Path) -> None:
    recorded: list[RecordedCompactionEvent] = []
    harness = await _create_compaction_harness(tmp_path, recorded)
    try:
        await harness.session._run_auto_compaction("overflow", True)

        assert recorded == [
            RecordedCompactionEvent("session_before_compact", "overflow", True),
            RecordedCompactionEvent("session_compact", "overflow", True),
        ]
    finally:
        harness.cleanup()
