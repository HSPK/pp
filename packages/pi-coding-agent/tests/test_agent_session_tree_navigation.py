"""Python port of `packages/coding-agent/test/agent-session-tree-navigation.test.ts`.

End-to-end tests for `AgentSession` tree navigation with branch summarization:
- Navigation to user messages (root and non-root)
- Navigation to non-user messages
- Branch summarization during navigation
- Summary attachment at the correct position in the tree
- Abort handling during summarization

The TypeScript suite is gated on a real `ANTHROPIC_API_KEY`. This port drives
the same code paths with `test_agent_session`'s scripted stream function, so
no network call is made.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_ai.types import ErrorEvent, StartEvent
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.session_manager import (
    BranchSummaryEntry,
    SessionManager,
    SessionMessageEntry,
)
from test_agent_session import (
    build_session,
    make_assistant_message,
    text_response,
)

SETTINGS = {"compaction": {"keepRecentTokens": 1}}


async def create_session(tmp_path: Path, responses: list[Any]) -> tuple[AgentSession, SessionManager]:
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, responses=list(responses), settings=SETTINGS
    )
    return session, session_manager


def _user_entries(session_manager: SessionManager) -> list[SessionMessageEntry]:
    return [
        e
        for e in session_manager.get_entries()
        if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
    ]


def _assistant_entries(session_manager: SessionManager) -> list[SessionMessageEntry]:
    return [
        e
        for e in session_manager.get_entries()
        if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "assistant"
    ]


class TestAgentSessionTreeNavigationE2E:
    async def test_should_navigate_to_user_message_and_put_text_in_editor(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(
            tmp_path, [text_response("first answer"), text_response("second answer")]
        )
        try:
            await asyncio.wait_for(session.prompt("First message"), timeout=5)
            await asyncio.wait_for(session.prompt("Second message"), timeout=5)

            tree = session_manager.get_tree()
            assert len(tree) == 1

            root_node = tree[0]
            assert root_node.entry.type == "message"

            result = await asyncio.wait_for(session.navigate_tree(root_node.entry.id, summarize=False), timeout=5)

            assert result.cancelled is False
            assert result.editor_text == "First message"

            # After navigating to the root user message the branch is empty again.
            assert session_manager.get_leaf_id() is None
        finally:
            session.dispose()

    async def test_should_navigate_to_non_user_message_without_editor_text(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(tmp_path, [text_response("hi there")])
        try:
            await asyncio.wait_for(session.prompt("Hello"), timeout=5)

            assistant_entry = _assistant_entries(session_manager)[0]

            result = await asyncio.wait_for(session.navigate_tree(assistant_entry.id, summarize=False), timeout=5)

            assert result.cancelled is False
            assert result.editor_text is None

            assert session_manager.get_leaf_id() == assistant_entry.id
        finally:
            session.dispose()

    async def test_should_create_branch_summary_when_navigating_with_summarize_true(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(
            tmp_path,
            [text_response("4"), text_response("6"), text_response("ABANDONED BRANCH SUMMARY")],
        )
        try:
            await asyncio.wait_for(session.prompt("What is 2+2?"), timeout=5)
            await asyncio.wait_for(session.prompt("What is 3+3?"), timeout=5)

            tree = session_manager.get_tree()
            root_node = tree[0]

            result = await asyncio.wait_for(session.navigate_tree(root_node.entry.id, summarize=True), timeout=10)

            assert result.cancelled is False
            assert result.editor_text == "What is 2+2?"
            assert result.summary_entry is not None
            assert result.summary_entry.type == "branch_summary"
            assert result.summary_entry.summary
            assert len(result.summary_entry.summary) > 0

            # Navigating to the root user message makes the summary a root entry.
            assert result.summary_entry.parent_id is None

            assert session_manager.get_leaf_id() == result.summary_entry.id
        finally:
            session.dispose()

    async def test_should_attach_summary_to_correct_parent_when_navigating_to_nested_user_message(
        self, tmp_path: Path
    ) -> None:
        session, session_manager = await create_session(
            tmp_path,
            [
                text_response("one"),
                text_response("two"),
                text_response("three"),
                text_response("ABANDONED BRANCH SUMMARY"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("Message one"), timeout=5)
            await asyncio.wait_for(session.prompt("Message two"), timeout=5)
            await asyncio.wait_for(session.prompt("Message three"), timeout=5)

            user_entries = _user_entries(session_manager)
            assert len(user_entries) == 3

            u2 = user_entries[1]
            a1 = session_manager.get_entry(u2.parent_id)
            assert a1 is not None

            result = await asyncio.wait_for(session.navigate_tree(u2.id, summarize=True), timeout=10)

            assert result.cancelled is False
            assert result.editor_text == "Message two"
            assert result.summary_entry is not None

            # The summary attaches to a1 (u2's parent), so a1 now has two children.
            assert result.summary_entry.parent_id == a1.id

            children = session_manager.get_children(a1.id)
            assert len(children) == 2
            child_types = sorted(c.type for c in children)
            assert "branch_summary" in child_types
            assert "message" in child_types
        finally:
            session.dispose()

    async def test_should_attach_summary_to_selected_node_when_navigating_to_assistant_message(
        self, tmp_path: Path
    ) -> None:
        session, session_manager = await create_session(
            tmp_path,
            [text_response("hi"), text_response("bye"), text_response("ABANDONED BRANCH SUMMARY")],
        )
        try:
            await asyncio.wait_for(session.prompt("Hello"), timeout=5)
            await asyncio.wait_for(session.prompt("Goodbye"), timeout=5)

            a1 = _assistant_entries(session_manager)[0]

            result = await asyncio.wait_for(session.navigate_tree(a1.id, summarize=True), timeout=10)

            assert result.cancelled is False
            assert result.editor_text is None  # No editor text for assistant messages
            assert result.summary_entry is not None

            assert result.summary_entry.parent_id == a1.id
            assert session_manager.get_leaf_id() == result.summary_entry.id
        finally:
            session.dispose()

    async def test_should_handle_abort_during_summarization(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(
            tmp_path, [text_response("something"), text_response("continued")]
        )
        try:
            await asyncio.wait_for(session.prompt("Tell me about something"), timeout=5)
            await asyncio.wait_for(session.prompt("Continue"), timeout=5)

            entries_before = session_manager.get_entries()
            leaf_before = session_manager.get_leaf_id()

            # Swap in a stream that never completes until the branch-summary
            # signal aborts. TS relies on a real (slow) LLM call plus a 100 ms
            # `setTimeout` for the same effect; that wall-clock wait is what
            # makes the assertion below load-dependent, so the stream signals
            # when it has actually been entered and waits on the abort signal
            # rather than polling.
            summarization_started = asyncio.Event()

            def hanging_stream_fn(model, context, options=None):
                stream = AssistantMessageEventStream()
                partial = make_assistant_message([])
                stream.push(StartEvent(partial=partial))
                signal = getattr(options, "signal", None)
                summarization_started.set()

                async def finish() -> None:
                    if signal is not None:
                        await signal.wait()
                    aborted = make_assistant_message([], stop_reason="aborted")
                    stream.push(ErrorEvent(reason="aborted", error=aborted))
                    stream.end()

                asyncio.ensure_future(finish())  # noqa: RUF006
                return stream

            session.agent.stream_function = hanging_stream_fn

            tree = session_manager.get_tree()
            root_node = tree[0]

            navigation = asyncio.ensure_future(session.navigate_tree(root_node.entry.id, summarize=True))
            await asyncio.wait_for(summarization_started.wait(), timeout=10)

            assert session.is_compacting is True

            session.abort_branch_summary()

            result = await asyncio.wait_for(navigation, timeout=10)

            assert result.cancelled is True
            assert result.aborted is True
            assert result.summary_entry is None

            entries_after = session_manager.get_entries()
            assert len(entries_after) == len(entries_before)
            assert session_manager.get_leaf_id() == leaf_before
        finally:
            session.dispose()

    async def test_should_not_create_summary_when_navigating_without_summarize_option(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(tmp_path, [text_response("one"), text_response("two")])
        try:
            await asyncio.wait_for(session.prompt("First"), timeout=5)
            await asyncio.wait_for(session.prompt("Second"), timeout=5)

            entries_before = len(session_manager.get_entries())

            tree = session_manager.get_tree()
            await asyncio.wait_for(session.navigate_tree(tree[0].entry.id, summarize=False), timeout=5)

            assert len(session_manager.get_entries()) == entries_before
            assert [e for e in session_manager.get_entries() if isinstance(e, BranchSummaryEntry)] == []
        finally:
            session.dispose()

    async def test_should_handle_navigation_to_same_position_no_op(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(tmp_path, [text_response("hi")])
        try:
            await asyncio.wait_for(session.prompt("Hello"), timeout=5)

            leaf_before = session_manager.get_leaf_id()
            assert leaf_before
            entries_before = len(session_manager.get_entries())

            result = await asyncio.wait_for(session.navigate_tree(leaf_before, summarize=False), timeout=5)

            assert result.cancelled is False
            assert session_manager.get_leaf_id() == leaf_before
            assert len(session_manager.get_entries()) == entries_before
        finally:
            session.dispose()

    async def test_should_support_custom_summarization_instructions(self, tmp_path: Path) -> None:
        seen_prompts: list[str] = []

        def summary_response(context) -> Any:
            messages = getattr(context, "messages", None) or []
            for message in messages:
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    seen_prompts.append(content)
                else:
                    seen_prompts.extend(part.text for part in content if getattr(part, "type", None) == "text")
            return text_response("A summary. MONKEY MONKEY MONKEY")

        session, session_manager = await create_session(
            tmp_path, [text_response("A typed language."), summary_response]
        )
        try:
            await asyncio.wait_for(session.prompt("What is TypeScript?"), timeout=5)

            tree = session_manager.get_tree()
            result = await asyncio.wait_for(
                session.navigate_tree(
                    tree[0].entry.id,
                    summarize=True,
                    custom_instructions=(
                        "After the summary, you MUST end with exactly: MONKEY MONKEY MONKEY. "
                        "This is of utmost importance."
                    ),
                ),
                timeout=10,
            )

            assert result.summary_entry is not None
            assert result.summary_entry.summary
            # TS asserts the live model obeyed the instruction. With a scripted
            # provider, assert instead that the instruction actually reached the
            # summarization request (`Additional focus: ...`).
            assert any("MONKEY MONKEY MONKEY" in prompt for prompt in seen_prompts)
            assert any("Additional focus:" in prompt for prompt in seen_prompts)
            assert "MONKEY MONKEY MONKEY" in result.summary_entry.summary
        finally:
            session.dispose()


class TestAgentSessionTreeNavigationBranchScenarios:
    async def test_should_navigate_between_branches_correctly(self, tmp_path: Path) -> None:
        session, session_manager = await create_session(
            tmp_path,
            [
                text_response("main one"),
                text_response("main two"),
                text_response("branch one"),
                text_response("ABANDONED BRANCH SUMMARY"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("Main branch start"), timeout=5)
            await asyncio.wait_for(session.prompt("Main branch continue"), timeout=5)

            entries = session_manager.get_entries()
            a1 = _assistant_entries(session_manager)[0]

            # Create a branch from a1: a1 -> u3 -> a3
            session_manager.branch(a1.id)
            session.agent.state.messages = session_manager.build_session_context().messages
            await asyncio.wait_for(session.prompt("Branch path"), timeout=5)

            # Now navigate back to u2 (on the main branch) with summarization.
            user_entries = [
                e for e in entries if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
            ]
            u2 = user_entries[1]  # "Main branch continue"

            result = await asyncio.wait_for(session.navigate_tree(u2.id, summarize=True), timeout=10)

            assert result.cancelled is False
            assert result.editor_text == "Main branch continue"
            assert result.summary_entry is not None
            assert len(result.summary_entry.summary) > 0
        finally:
            session.dispose()
