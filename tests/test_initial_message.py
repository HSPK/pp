"""Python port of `packages/coding-agent/test/initial-message.test.ts`."""

from __future__ import annotations

from pi_coding_agent.cli.args import Args
from pi_coding_agent.cli.initial_message import build_initial_message


def create_args(messages: list[str] | None = None) -> Args:
    return Args(messages=list(messages or []), file_args=[], unknown_flags={}, diagnostics=[])


def test_merges_piped_stdin_with_the_first_cli_message() -> None:
    parsed = create_args(["Summarize the text given"])

    result = build_initial_message(parsed, stdin_content="README contents\n")

    assert result.initial_message == "README contents\nSummarize the text given"
    assert parsed.messages == []


def test_uses_stdin_as_the_prompt_when_no_cli_message_is_present() -> None:
    parsed = create_args()

    result = build_initial_message(parsed, stdin_content="README contents")

    assert result.initial_message == "README contents"
    assert parsed.messages == []


def test_combines_stdin_file_text_and_first_cli_message() -> None:
    parsed = create_args(["Explain it", "Second message"])

    result = build_initial_message(parsed, stdin_content="stdin\n", file_text="file\n")

    assert result.initial_message == "stdin\nfile\nExplain it"
    assert parsed.messages == ["Second message"]
