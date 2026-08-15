"""Aggregate token/cost usage across a session's entries.

Port of `packages/coding-agent/src/core/usage-totals.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_ai.types import Usage

from pi_coding_agent.core.session_manager import (
    BranchSummaryEntry,
    CompactionEntry,
    SessionEntry,
    SessionMessageEntry,
)


@dataclass
class UsageTotals:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0


def create_usage_totals() -> UsageTotals:
    return UsageTotals()


def add_usage_to_totals(totals: UsageTotals, usage: Usage) -> None:
    totals.input += usage.input
    totals.output += usage.output
    totals.cache_read += usage.cache_read
    totals.cache_write += usage.cache_write
    totals.cost += usage.cost.total


@dataclass
class UsageCostBreakdownEntry:
    key: str
    cost: float
    tokens: int


def get_usage_cost_breakdown(entries: list[SessionEntry]) -> list[UsageCostBreakdownEntry]:
    """Group attributable assistant usage by model; all other usage goes into `Tools/summaries`."""
    totals_by_key: dict[str, UsageTotals] = {}

    for entry in entries:
        key: str | None = None
        usage: Usage | None = None
        if isinstance(entry, SessionMessageEntry) and entry.message.role == "assistant":
            key = f"{entry.message.provider}/{entry.message.response_model or entry.message.model}"
            usage = entry.message.usage
        elif isinstance(entry, SessionMessageEntry) and entry.message.role == "toolResult" and entry.message.usage:
            key = "Tools/summaries"
            usage = entry.message.usage
        elif isinstance(entry, (BranchSummaryEntry, CompactionEntry)) and entry.usage:
            key = "Tools/summaries"
            usage = entry.usage
        if not key or usage is None:
            continue

        totals = totals_by_key.get(key)
        if totals is None:
            totals = create_usage_totals()
            totals_by_key[key] = totals
        add_usage_to_totals(totals, usage)

    breakdown = [
        UsageCostBreakdownEntry(
            key=key,
            cost=totals.cost,
            tokens=totals.input + totals.output + totals.cache_read + totals.cache_write,
        )
        for key, totals in totals_by_key.items()
    ]
    breakdown = [entry for entry in breakdown if entry.cost > 0 or entry.tokens > 0]
    breakdown.sort(key=lambda entry: entry.cost, reverse=True)
    return breakdown
