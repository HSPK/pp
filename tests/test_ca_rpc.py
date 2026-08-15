"""Python port of `packages/coding-agent/test/rpc.test.ts`.

Nothing in this file runs, but the reason is narrower than it used to be.

The RPC mode itself **is** ported (`modes/rpc/`), and the behaviour these cases
describe is pinned without credentials in `tests/suite/test_rpc_mode.py`,
driving a real `AgentSession` from the faux provider. What is not reproducible
here is the *shape* of the TypeScript suite: the whole `describe` is
`describe.skipIf(!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_OAUTH_TOKEN)`
and every case makes real Anthropic calls through a spawned `dist/cli.js`, so
it pins nothing on the TypeScript side either unless someone runs it with a
paid key. Reproducing it faithfully would mean spending real tokens per case.

Each TypeScript case is still recorded below with what it asserts, and with
where the same behaviour is covered here, so the mapping stays checkable.
"""

from __future__ import annotations

import pytest

_REASON = (
    "test/rpc.test.ts is credential-gated e2e (ANTHROPIC_API_KEY/ANTHROPIC_OAUTH_TOKEN): "
    "every case spawns a real CLI and makes paid Anthropic calls. The RPC mode is "
    "ported; the same behaviour is pinned credential-free in "
    "tests/suite/test_rpc_mode.py (see COVERED_BY below)."
)

COVERED_BY = {
    "test_should_get_state": "test_get_state_reports_the_live_session",
    "test_should_execute_bash_command": "test_bash_runs_the_command_and_records_it",
    "test_should_set_and_get_thinking_level": "test_toggle_commands_reach_the_session",
    "test_should_cycle_thinking_level": "test_cycle_with_nothing_to_cycle_to_answers_null_data",
    "test_should_get_available_thinking_levels": "test_get_available_thinking_levels",
    "test_should_get_available_models": "test_get_available_models_lists_the_snapshot",
    "test_should_create_new_session": "test_replacement_commands_rebind_when_they_succeed",
    "test_should_get_last_assistant_text": "test_get_last_assistant_text",
    "test_should_get_session_entries_with_since_cursor": "test_get_entries_returns_only_what_follows_since",
    "test_should_get_session_tree": "test_get_tree_reports_the_leaf",
    "test_should_set_and_get_session_name": "test_set_session_name_trims_and_applies",
}
"""Which credential-free case pins each credential-gated one.

The rest (`save_messages_to_session_file`, `manual_compaction`,
`bash_output_in_llm_context`, `session_stats`, `export_to_html`,
`retain_pre_compaction_entries`) assert on what a *real model* produced, so
they have no faux-provider equivalent.
"""


def test_every_covered_case_names_a_real_test() -> None:
    """A stale mapping is worse than none: it claims coverage that moved away."""
    import ast
    from pathlib import Path

    suite = Path(__file__).resolve().parent / "suite" / "test_rpc_mode.py"
    tree = ast.parse(suite.read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name.startswith("test_")
    }

    missing = sorted(name for name in COVERED_BY.values() if name not in defined)
    assert missing == [], f"COVERED_BY points at tests that no longer exist: {missing}"

    documented = {name for name in globals() if name.startswith("test_should_")}
    unknown = sorted(set(COVERED_BY) - documented)
    assert unknown == [], f"COVERED_BY names cases that are not in this file: {unknown}"


_skip_credential_gated = pytest.mark.skip(reason=_REASON)
"""Applied to the `test_should_*` stubs at the end of this module.

Not a module-level `pytestmark`: that would also skip the guard below, and a
mapping nobody checks is exactly how it goes stale.
"""


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


for _name, _case in list(globals().items()):
    if _name.startswith("test_should_"):
        _case.pytestmark = [_skip_credential_gated]
