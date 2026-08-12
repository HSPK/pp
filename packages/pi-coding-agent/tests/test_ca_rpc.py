"""Python port of `packages/coding-agent/test/rpc.test.ts`.

Nothing in this file runs, for two independent reasons, and both are worth
stating because either one alone would already disqualify it:

1. **The TypeScript suite is credential-gated e2e.** The whole `describe` is
   `describe.skipIf(!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_OAUTH_TOKEN)`,
   and every case makes real Anthropic calls through a spawned `dist/cli.js`.
   It pins no verified behavior on the TypeScript side either unless someone
   runs it with a paid key.
2. **The legacy stdio RPC mode is not ported.** `RpcClient` speaks
   `src/modes/rpc/rpc-client.ts`'s line-framed stdio protocol to
   `rpc-mode.ts`. This port keeps only that protocol's framing
   (`modes/rpc/jsonl.py`, cross-checked byte for byte against the TypeScript
   and covered by its own tests); the mode driver is superseded by the
   `pi_server`/`pi_protocol`/`pi_client` socket stack, which *is* ported and
   is exercised end to end over a real Unix socket in
   `tests/test_agent_session_runtime.py`. See the README's "Not ported, by
   decision" list and `modes/rpc/__init__.py`'s module docstring.

Each TypeScript case is recorded below with what it asserts, so the gap stays
visible if the mode is ever ported.
"""

from __future__ import annotations

import pytest

_REASON = (
    "test/rpc.test.ts is credential-gated e2e (ANTHROPIC_API_KEY/ANTHROPIC_OAUTH_TOKEN) "
    "and drives the legacy stdio RPC mode, which this port replaces with the "
    "pi_server/pi_client socket stack (see modes/rpc/__init__.py)."
)

pytestmark = pytest.mark.skip(reason=_REASON)


def test_should_get_state() -> None:
    """`state.model.provider == "anthropic"`, `state.model.id == "claude-sonnet-4-5"`,
    `state.isStreaming is False`, `state.messageCount == 0`."""


def test_should_save_messages_to_session_file() -> None:
    """After one prompt: at least two `message_end` events; exactly one `.jsonl`
    under `<sessionDir>/sessions/<cwd-dir>/`; its first entry has
    `type == "session"`; the `message` entries include roles `user` and
    `assistant`."""


def test_should_handle_manual_compaction() -> None:
    """`compact()` returns a defined `summary` and `tokensBefore > 0`, and the
    session file gains exactly one `type == "compaction"` entry carrying a
    defined `summary`."""


def test_should_execute_bash_command() -> None:
    """`bash("echo hello")` -> `output.strip() == "hello"`, `exitCode == 0`,
    `cancelled is False`."""


def test_should_add_bash_output_to_context() -> None:
    """The session file gains exactly one entry with
    `message.role == "bashExecution"` whose `message.output` contains the
    echoed unique value."""


def test_should_include_bash_output_in_llm_context() -> None:
    """After a bash command, the model's own reply text contains the echoed
    unique value -- i.e. the bash entry really reached the LLM context."""


def test_should_set_and_get_thinking_level() -> None:
    """`setThinkingLevel("high")` -> `getState().thinkingLevel == "high"`."""


def test_should_cycle_thinking_level() -> None:
    """`cycleThinkingLevel()` returns a level different from the current one,
    and `getState().thinkingLevel` then equals the returned level."""


def test_should_get_available_thinking_levels() -> None:
    """`getAvailableThinkingLevels()` is non-empty, contains the level reported
    by `getState()`, and `cycleThinkingLevel()` only ever lands on a level from
    that list (and on a *different* level when more than one exists)."""


def test_should_get_available_models() -> None:
    """`getAvailableModels()` is non-empty and every entry has a defined
    `provider` and `id`, `contextWindow > 0`, and a boolean `reasoning`."""


def test_should_get_session_stats() -> None:
    """`sessionFile` and `sessionId` defined, `userMessages >= 1`,
    `assistantMessages >= 1`."""


def test_should_create_new_session() -> None:
    """`messageCount > 0` after a prompt, then `newSession()` resets it to 0."""


def test_should_export_to_html() -> None:
    """`exportHtml().path` is defined, ends with `.html`, and exists on disk."""


def test_should_get_last_assistant_text() -> None:
    """Undefined before the first prompt; contains the requested token after."""


def test_should_get_session_entries_with_since_cursor() -> None:
    """`getEntries()` returns at least two entries, every entry has an `id`, and
    `leafId` is the last entry's id. `getEntries(entries[0].id)` returns exactly
    the entries strictly after that id with the same `leafId`. An unknown
    `since` id rejects with `"Entry not found"`."""


def test_should_get_session_tree() -> None:
    """`getTree()` reports the same `leafId` as `getEntries()`, has a single
    root, and walking the single-child chain yields exactly the entry ids from
    `getEntries()` (the chain ends at a node with no children)."""


def test_should_retain_pre_compaction_entries_in_get_entries() -> None:
    """After `compact()`, the first N entries are byte-for-byte the same ids as
    before (append-only), and a `type == "compaction"` entry was appended."""


def test_should_set_and_get_session_name() -> None:
    """`sessionName` undefined initially; after `setSessionName("my-test-session")`
    `getState().sessionName` matches, and the session file gains exactly one
    `type == "session_info"` entry with that `name`."""
