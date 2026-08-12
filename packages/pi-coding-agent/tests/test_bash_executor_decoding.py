"""Tests for bash output decoding.

The executor receives raw pipe reads, so a multi-byte character can land across
two chunks. Decoding each chunk independently corrupts it into replacement
characters, and that corruption reaches both the streamed deltas the UI shows
and the text recorded in the session transcript.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.core.bash_executor import execute_bash_with_operations


class ChunkedOperations:
    """A BashOperations stand-in that replays a fixed list of byte chunks."""

    def __init__(self, chunks: list[bytes], exit_code: int = 0) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.commands: list[str] = []

    async def exec(self, command, cwd, on_data, signal=None, *_args):
        self.commands.append(command)
        for chunk in self.chunks:
            on_data(chunk)
        return self.exit_code


async def test_multibyte_character_split_across_chunks_is_not_corrupted():
    payload = "héllo→世界".encode()
    # Split inside the two-byte "é".
    operations = ChunkedOperations([payload[:2], payload[2:]])

    result = await execute_bash_with_operations("echo", "/tmp", operations)

    assert result.output == "héllo→世界"
    assert "\ufffd" not in result.output


async def test_every_byte_delivered_separately_still_decodes():
    payload = "日本語テキスト".encode()
    operations = ChunkedOperations([payload[i : i + 1] for i in range(len(payload))])

    result = await execute_bash_with_operations("echo", "/tmp", operations)

    assert result.output == "日本語テキスト"


async def test_streamed_chunks_are_also_clean():
    payload = "aé→b".encode()
    operations = ChunkedOperations([payload[:2], payload[2:]])
    streamed: list[str] = []

    await execute_bash_with_operations("echo", "/tmp", operations, on_chunk=streamed.append)

    assert "\ufffd" not in "".join(streamed)
    assert "".join(streamed) == "aé→b"


async def test_output_ending_mid_character_is_flushed_not_dropped():
    # A truncated trailing sequence must still surface (as a replacement
    # character) rather than vanishing from the output.
    operations = ChunkedOperations([b"ok", "é".encode()[:1]])

    result = await execute_bash_with_operations("echo", "/tmp", operations)

    assert result.output.startswith("ok")
    assert len(result.output) > 2


async def test_plain_ascii_is_unaffected():
    operations = ChunkedOperations([b"hello ", b"world"])
    result = await execute_bash_with_operations("echo", "/tmp", operations)
    assert result.output == "hello world"


async def test_exit_code_and_command_are_passed_through():
    operations = ChunkedOperations([b"out"], exit_code=3)
    result = await execute_bash_with_operations("ls -la", "/tmp", operations)
    assert result.exit_code == 3
    assert result.cancelled is False
    assert operations.commands == ["ls -la"]


async def test_carriage_returns_are_stripped():
    operations = ChunkedOperations([b"a\r\nb\r\n"])
    result = await execute_bash_with_operations("echo", "/tmp", operations)
    assert result.output == "a\nb\n"


@pytest.mark.parametrize(
    "text",
    ["émoji 🎉 mixed", "中文测试", "ελληνικά", "a" * 100 + "é"],
)
async def test_round_trip_for_various_scripts(text):
    payload = text.encode()
    midpoint = len(payload) // 2
    operations = ChunkedOperations([payload[:midpoint], payload[midpoint:]])

    result = await execute_bash_with_operations("echo", "/tmp", operations)

    assert result.output == text
