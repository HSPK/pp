"""Python port of `packages/coding-agent/test/suite/regressions/tree-during-streaming.test.ts`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import create_harness, user_msg
from pi_ai.providers.faux import faux_assistant_message


async def test_rejects_navigation_without_changing_the_active_leaf(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    target_id = harness.session_manager.append_message(user_msg("first"))
    navigation_result: Any = None
    leaf_unchanged = False

    try:
        # Navigate from inside the response factory, while the run is active.
        async def respond(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal navigation_result, leaf_unchanged
            active_leaf_id = harness.session_manager.get_leaf_id()
            try:
                navigation_result = await harness.session.navigate_tree(target_id, summarize=False)
            except Exception as error:
                navigation_result = error
            leaf_unchanged = active_leaf_id != target_id and harness.session_manager.get_leaf_id() == active_leaf_id
            return faux_assistant_message("response")

        harness.set_responses([respond])
        await harness.session.prompt("second")

        assert isinstance(navigation_result, RuntimeError)
        assert str(navigation_result) == "Wait for the current response to finish before navigating the session tree."
        assert leaf_unchanged is True
    finally:
        harness.cleanup()
