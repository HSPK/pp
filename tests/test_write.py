"""Tests for the `write` tool. Ported from `tools.test.ts` (describe("write tool"))."""

from __future__ import annotations

import pytest
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.write import create_write_tool


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


async def test_writes_file_contents(tmp_path):
    tool = create_write_tool(str(tmp_path))
    test_file = tmp_path / "write-test.txt"
    content = "Test content"

    result = await tool.execute("call-1", {"path": str(test_file), "content": content})

    assert "Successfully wrote" in get_text(result)
    assert str(test_file) in get_text(result)
    assert result.details is None
    assert test_file.read_text() == content


async def test_creates_parent_directories(tmp_path):
    tool = create_write_tool(str(tmp_path))
    test_file = tmp_path / "nested" / "dir" / "test.txt"
    content = "Nested content"

    result = await tool.execute("call-2", {"path": str(test_file), "content": content})

    assert "Successfully wrote" in get_text(result)
    assert test_file.read_text() == content


async def test_overwrites_existing_file(tmp_path):
    tool = create_write_tool(str(tmp_path))
    test_file = tmp_path / "overwrite.txt"
    test_file.write_text("old content")

    await tool.execute("call-3", {"path": str(test_file), "content": "new content"})

    assert test_file.read_text() == "new content"


async def test_relative_path_resolved_against_cwd(tmp_path):
    tool = create_write_tool(str(tmp_path))

    await tool.execute("call-4", {"path": "rel.txt", "content": "hi"})

    assert (tmp_path / "rel.txt").read_text() == "hi"


async def test_aborted_signal_raises(tmp_path):
    tool = create_write_tool(str(tmp_path))
    signal = AbortSignal()
    signal.abort()

    with pytest.raises(RuntimeError, match="Operation aborted"):
        await tool.execute("call-5", {"path": str(tmp_path / "aborted.txt"), "content": "x"}, signal)
