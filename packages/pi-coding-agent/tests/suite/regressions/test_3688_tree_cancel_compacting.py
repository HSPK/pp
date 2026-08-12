"""Python port of `packages/coding-agent/test/suite/regressions/3688-tree-cancel-compacting.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import assistant_msg, create_harness, user_msg
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import SessionBeforeTreeResult


async def test_clears_branch_summary_state_when_session_before_tree_cancels(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        def on_before_tree(event, ctx) -> SessionBeforeTreeResult:
            return SessionBeforeTreeResult(cancel=True)

        pi.on("session_before_tree", on_before_tree)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        target_id = harness.session_manager.append_message(user_msg("first"))
        harness.session_manager.append_message(assistant_msg("reply"))
        current_leaf_id = harness.session_manager.append_message(user_msg("second"))

        assert harness.session_manager.get_leaf_id() == current_leaf_id

        result = await harness.session.navigate_tree(target_id, summarize=False)

        assert result.cancelled is True
        assert result.editor_text is None
        assert result.summary_entry is None
        assert harness.session.is_compacting is False
        assert harness.session_manager.get_leaf_id() == current_leaf_id
    finally:
        harness.cleanup()
