"""Python port of `packages/coding-agent/test/branch-summary-extensions.test.ts`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pi_ai.types import Cost, Usage

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import SessionBeforeTreeResult

sys.path.insert(0, str(Path(__file__).parent / "suite"))

from harness import Harness, assistant_msg, create_harness, user_msg


@pytest.fixture
def harnesses() -> list[Harness]:
    created: list[Harness] = []
    yield created
    while created:
        created.pop().cleanup()


async def test_persists_extension_provided_summary_usage_in_session_totals(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    usage = Usage(
        input=10,
        output=20,
        cache_read=30,
        cache_write=40,
        total_tokens=100,
        cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )

    def register(pi: ExtensionAPI) -> None:
        pi.on(
            "session_before_tree",
            lambda _event, _ctx: SessionBeforeTreeResult(
                summary={"summary": "Summary provided by extension", "usage": usage}
            ),
        )

    harness = await create_harness(tmp_path, extension_factories=[register])
    harnesses.append(harness)

    target_id = harness.session_manager.append_message(user_msg("first branch"))
    harness.session_manager.append_message(assistant_msg("first reply"))
    harness.session_manager.append_message(user_msg("abandoned branch work"))
    harness.session_manager.append_message(assistant_msg("abandoned reply"))

    result = await harness.session.navigate_tree(target_id, summarize=True)
    summary_entry = result.summary_entry

    assert summary_entry is not None
    assert summary_entry.type == "branch_summary"
    assert summary_entry.from_hook is True
    assert summary_entry.summary == "Summary provided by extension"
    assert summary_entry.usage == usage

    stats = harness.session.get_session_stats()
    assert (stats.tokens.input, stats.tokens.output) == (12, 22)
    assert (stats.tokens.cache_read, stats.tokens.cache_write) == (30, 40)
    assert stats.tokens.total == 104
    assert stats.cost == 1
