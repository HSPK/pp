"""Detect wasted prompt-cache spend across a session's assistant turns.

Ported from ``packages/coding-agent/src/core/cache-stats.ts``.

Providers like Anthropic bill a much lower rate for prompt tokens served from
cache than for tokens re-read from scratch. A cache miss happens when the
previous turn's prompt should still be cached (it was read/written recently
enough) but the next turn re-pays full price for those tokens anyway -- most
often because the idle gap between requests exceeded the provider's cache
TTL. This module replays a session's entries to find those misses and total
up the extra dollars/tokens they cost, so the CLI can show a "cache miss"
notice instead of leaving the user to wonder why a turn was more expensive
than expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pi_ai.types import AssistantMessage, Model

from pi_coding_agent.core.session_manager import SessionEntry

CACHE_TTL_MS = 5 * 60 * 1000
"""Prompt-cache TTL: idle gaps longer than this are worth mentioning as the
likely cause of a miss. Anthropic's default cache TTL is 5 minutes."""

_NOISE_FLOOR_TOKENS = 1024
"""Per-turn misses at or below this are cache breakpoint granularity noise."""


@dataclass
class CacheMiss:
    """A counted cache miss on a single assistant message."""

    missed_tokens: int
    """Prompt tokens that were in the previous turn's prompt but not read from cache."""
    missed_cost: float
    """Extra dollars paid vs. a full cache hit; 0 when pricing is unknown."""
    idle_ms: int
    """Milliseconds since the previous request (which last refreshed the cache)."""
    model_changed: bool
    """True when the model changed relative to the previous request."""


@dataclass
class CacheWasteTotals:
    missed_tokens: int
    missed_cost: float
    missed_count: int
    """Number of counted misses (turns above the noise floor)."""


class ModelPriceSource(Protocol):
    """Minimal pricing lookup, satisfied by `ModelRuntime`. Cost is $/million tokens."""

    def get_model(self, provider: str, model_id: str) -> Model | None: ...


@dataclass
class _PreviousRequest:
    """The last request seen by the scan; everything in its prompt should be cached."""

    prompt_tokens: int
    model_key: str
    timestamp: int
    reported_cache: bool
    """Sticky: some earlier request in this scan segment reported cache activity.
    Distinguishes a total miss on a cache-read-only provider (OpenAI-style,
    writes unreported) from a provider that never reports caching at all."""


def _detect_miss(
    prev: _PreviousRequest | None,
    message: AssistantMessage,
    models: ModelPriceSource,
) -> CacheMiss | None:
    """Compute the cache miss for one assistant message relative to the previous
    request. Returns `None` when nothing is counted: first turn, after a
    reset, no cache activity ever reported (provider without cache support), or
    miss below the noise floor.
    """
    usage = message.usage
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    # A zero-cache turn only counts when cache activity was reported before:
    # on cache-read-only providers that is a total miss, while on providers
    # that never report caching it means nothing.
    if prev is None or prompt_tokens <= 0 or (usage.cache_read + usage.cache_write == 0 and not prev.reported_cache):
        return None

    missed_tokens = min(prev.prompt_tokens, prompt_tokens) - usage.cache_read
    if missed_tokens <= _NOISE_FLOOR_TOKENS:
        return None

    # Extra cost = missed tokens billed at the actual paid rate (input/cacheWrite,
    # incl. write premium) instead of the cache-read rate. Missed tokens can only
    # land in the input or cacheWrite buckets, so the paid rate comes straight
    # from this message's own cost breakdown.
    paid_tokens = usage.input + usage.cache_write
    paid_per_token = (usage.cost.input + usage.cost.cache_write) / paid_tokens if paid_tokens > 0 else 0.0
    if usage.cache_read > 0:
        read_per_token = usage.cost.cache_read / usage.cache_read
    else:
        model = models.get_model(message.provider, message.model)
        read_per_token = (model.cost.cache_read if model is not None else 0.0) / 1_000_000

    return CacheMiss(
        missed_tokens=missed_tokens,
        missed_cost=missed_tokens * max(0.0, paid_per_token - read_per_token),
        idle_ms=max(0, message.timestamp - prev.timestamp),
        model_changed=f"{message.provider}/{message.model}" != prev.model_key,
    )


def _as_previous_request(message: AssistantMessage, reported_cache: bool) -> _PreviousRequest | None:
    usage = message.usage
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    if prompt_tokens <= 0:
        return None
    return _PreviousRequest(
        prompt_tokens=prompt_tokens,
        model_key=f"{message.provider}/{message.model}",
        timestamp=message.timestamp,
        reported_cache=reported_cache or usage.cache_read + usage.cache_write > 0,
    )


@dataclass
class _ScanResult:
    prev: _PreviousRequest | None
    totals: CacheWasteTotals
    misses: dict[int, CacheMiss]
    """Keyed by `id(message)` since `AssistantMessage` is not hashable-by-identity
    in Python the way a JS object reference is; callers that need the message
    back should zip against the entry list they passed in."""


def _scan(entries: list[SessionEntry], models: ModelPriceSource) -> _ScanResult:
    prev: _PreviousRequest | None = None
    totals = CacheWasteTotals(missed_tokens=0, missed_cost=0.0, missed_count=0)
    misses: dict[int, CacheMiss] = {}

    for entry in entries:
        if entry.type in ("compaction", "branch_summary"):
            # The context legitimately changed; the next turn's prompt is new content,
            # not re-billed content. Model switches are NOT exempt: they re-bill the
            # full prompt and should be counted.
            prev = None
            continue
        if entry.type == "message" and entry.message.role == "assistant":
            message = entry.message
            miss = _detect_miss(prev, message, models)
            if miss is not None:
                totals.missed_tokens += miss.missed_tokens
                totals.missed_cost += miss.missed_cost
                totals.missed_count += 1
                misses[id(message)] = miss
            prev = _as_previous_request(message, prev.reported_cache if prev is not None else False) or prev

    return _ScanResult(prev=prev, totals=totals, misses=misses)


def compute_cache_waste(entries: list[SessionEntry], models: ModelPriceSource) -> CacheWasteTotals:
    """Cumulative cache waste across a session: prompt tokens that should have been
    cache reads (they were in the previous turn's prompt) but were re-billed.
    """
    return _scan(entries, models).totals


def collect_cache_misses(entries: list[SessionEntry], models: ModelPriceSource) -> dict[int, CacheMiss]:
    """All counted cache misses across a session, keyed by `id()` of the assistant
    message that paid for them. Used to re-derive transcript notices when
    rebuilding the chat from entries (resume, post-compaction rebuild).
    """
    return _scan(entries, models).misses


def detect_cache_miss(
    entries: list[SessionEntry],
    message: AssistantMessage,
    models: ModelPriceSource,
) -> CacheMiss | None:
    """Detect a cache miss on a just-completed assistant message.
    `entries` must not yet contain `message` (message_end fires before persistence).
    """
    return _detect_miss(_scan(entries, models).prev, message, models)
