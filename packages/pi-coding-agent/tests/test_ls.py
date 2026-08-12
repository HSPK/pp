"""Tests for the `ls` tool. Ported from `tools.test.ts` (describe("ls tool"))."""

from __future__ import annotations

import pytest
from pi_coding_agent.tools.ls import create_ls_tool


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


async def test_lists_dotfiles_and_directories(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    (tmp_path / ".hidden-file").write_text("secret")
    (tmp_path / ".hidden-dir").mkdir()

    result = await tool.execute("call-1", {"path": str(tmp_path)})
    output = get_text(result)

    assert ".hidden-file" in output
    assert ".hidden-dir/" in output


async def test_sorts_alphabetically_case_insensitive(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    for name in ["banana.txt", "Apple.txt", "cherry.txt"]:
        (tmp_path / name).write_text("x")

    result = await tool.execute("call-2", {"path": str(tmp_path)})
    output = get_text(result).split("\n")

    assert output == ["Apple.txt", "banana.txt", "cherry.txt"]


async def test_empty_directory(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = await tool.execute("call-3", {"path": str(empty_dir)})

    assert get_text(result) == "(empty directory)"


async def test_nonexistent_path_raises(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    missing = tmp_path / "does-not-exist"

    with pytest.raises(RuntimeError, match="Path not found"):
        await tool.execute("call-4", {"path": str(missing)})


async def test_not_a_directory_raises(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")

    with pytest.raises(RuntimeError, match="Not a directory"):
        await tool.execute("call-5", {"path": str(file_path)})


async def test_default_path_is_cwd(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    (tmp_path / "a.txt").write_text("x")

    result = await tool.execute("call-6", {})

    assert "a.txt" in get_text(result)


async def test_entry_limit_reached(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    for i in range(10):
        (tmp_path / f"file{i:02d}.txt").write_text("x")

    result = await tool.execute("call-7", {"path": str(tmp_path), "limit": 3})
    output = get_text(result)

    assert "3 entries limit reached" in output
    assert result.details is not None
    assert result.details.entry_limit_reached == 3


async def test_marks_directories_with_trailing_slash(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file.txt").write_text("x")

    result = await tool.execute("call-8", {"path": str(tmp_path)})
    output = get_text(result).split("\n")

    assert "subdir/" in output
    assert "file.txt" in output
