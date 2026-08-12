"""Python port of `packages/coding-agent/test/cache-stats.test.ts`.

`collect_cache_misses` keys its result by `id(message)` (Python dataclasses are
not hashable by identity the way a JS object reference is), so the lookup in
the `collectCacheMisses` case uses `id(...)`.
"""

from __future__ import annotations

import pytest
from pi_ai.types import AssistantMessage, Cost, Model, ModelCost, Usage
from pi_coding_agent.core.cache_stats import (
    collect_cache_misses,
    compute_cache_waste,
    detect_cache_miss,
)
from pi_coding_agent.core.session_manager import (
    CompactionEntry,
    SessionEntry,
    SessionMessageEntry,
)


class _Models:
    """$/million tokens; used as cache-read price fallback on full-miss turns."""

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return Model(id=model_id, provider=provider, cost=ModelCost(cache_read=0.3))


MODELS = _Models()


def assistant(
    *,
    input_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cost: Cost | None = None,
    model: str = "test-model",
    timestamp: int = 0,
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api="anthropic-messages",
        provider="test",
        model=model,
        usage=Usage(
            input=input_tokens,
            output=10,
            cache_read=cache_read,
            cache_write=cache_write,
            total_tokens=0,
            cost=cost if cost is not None else Cost(),
        ),
        stop_reason="stop",
        timestamp=timestamp,
    )


def entry(message: AssistantMessage) -> SessionEntry:
    return SessionMessageEntry(id="x", parent_id=None, timestamp="", message=message)


# Turn 1: fresh 100k cache write at $3.75/M
TURN1 = assistant(cache_write=100_000, cost=Cost(cache_write=0.375), timestamp=0)
# Turn 2: healthy, everything read back at $0.30/M
TURN2 = assistant(
    cache_read=100_000,
    cache_write=5_000,
    cost=Cost(cache_read=0.03, cache_write=0.019),
    timestamp=60_000,
)


# ---------------------------------------------------------------------------
# compute_cache_waste
# ---------------------------------------------------------------------------


def test_accumulates_missed_tokens_and_cost_across_turns() -> None:
    # Turn 3: full miss, previous 105k prompt re-billed at $3.75/M write
    turn3 = assistant(cache_write=110_000, cost=Cost(cache_write=0.4125), timestamp=120_000)

    totals = compute_cache_waste([entry(TURN1), entry(TURN2), entry(turn3)], MODELS)

    assert totals.missed_tokens == 105_000
    # 105k at ($3.75 - $0.30)/M
    assert totals.missed_cost == pytest.approx(0.36225, abs=1e-5)


def test_counts_nothing_for_healthy_sessions() -> None:
    totals = compute_cache_waste([entry(TURN1), entry(TURN2)], MODELS)

    assert totals.missed_tokens == 0
    assert totals.missed_cost == 0


def test_skips_the_turn_after_a_compaction_reset() -> None:
    reset = CompactionEntry(
        id="c",
        parent_id=None,
        timestamp="",
        summary="s",
        first_kept_entry_id="x",
        tokens_before=0,
    )
    after_reset = assistant(cache_write=20_000, cost=Cost(cache_write=0.075))

    totals = compute_cache_waste([entry(TURN1), reset, entry(after_reset)], MODELS)

    assert totals.missed_tokens == 0


def test_counts_misses_caused_by_model_switches() -> None:
    other_model = assistant(cache_write=100_000, cost=Cost(cache_write=0.375), model="other-model")

    totals = compute_cache_waste([entry(TURN1), entry(other_model)], MODELS)

    assert totals.missed_tokens == 100_000
    assert totals.missed_count == 1


def test_skips_providers_that_report_no_cache_activity() -> None:
    a = assistant(input_tokens=100_000)
    b = assistant(input_tokens=110_000)

    totals = compute_cache_waste([entry(a), entry(b)], MODELS)

    assert totals.missed_tokens == 0


# ---------------------------------------------------------------------------
# collect_cache_misses
# ---------------------------------------------------------------------------


def test_maps_counted_misses_to_their_assistant_messages() -> None:
    miss_turn = assistant(cache_write=110_000, cost=Cost(cache_write=0.4125), timestamp=120_000)

    misses = collect_cache_misses([entry(TURN1), entry(TURN2), entry(miss_turn)], MODELS)

    assert len(misses) == 1
    assert misses[id(miss_turn)].missed_tokens == 105_000


# ---------------------------------------------------------------------------
# detect_cache_miss
# ---------------------------------------------------------------------------


def test_detects_a_miss_on_a_just_completed_message_with_idle_time() -> None:
    miss_message = assistant(cache_write=110_000, cost=Cost(cache_write=0.4125), timestamp=600_000)

    miss = detect_cache_miss([entry(TURN1), entry(TURN2)], miss_message, MODELS)

    assert miss is not None
    assert miss.missed_tokens == 105_000
    assert miss.missed_cost == pytest.approx(0.36225, abs=1e-5)
    # 600s - 60s since the previous request
    assert miss.idle_ms == 540_000
    assert miss.model_changed is False


def test_flags_model_switches_on_detected_misses() -> None:
    other_model = assistant(
        cache_write=110_000,
        cost=Cost(cache_write=0.4125),
        model="other-model",
        timestamp=120_000,
    )

    miss = detect_cache_miss([entry(TURN1), entry(TURN2)], other_model, MODELS)

    assert miss is not None
    assert miss.missed_tokens == 105_000
    assert miss.model_changed is True


def test_returns_none_for_healthy_turns() -> None:
    healthy = assistant(
        cache_read=105_000,
        cache_write=2_000,
        cost=Cost(cache_read=0.0315, cache_write=0.0075),
        timestamp=120_000,
    )

    assert detect_cache_miss([entry(TURN1), entry(TURN2)], healthy, MODELS) is None


def test_returns_none_for_the_first_turn_of_a_session() -> None:
    assert detect_cache_miss([], TURN1, MODELS) is None
