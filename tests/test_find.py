"""Tests for the `find` tool. Ported from `tools.test.ts` (describe("find tool")).

The TypeScript version shells out to `fd`; this port always uses a
pure-Python glob+gitignore walk (see `find.py` module docstring). The one TS
case that depends on `fd` is skipped individually below.
"""

from __future__ import annotations

import pytest

from pi_coding_agent.tools.find import create_find_tool


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


def lines_of(result) -> list[str]:
    return [line.strip() for line in get_text(result).split("\n") if line.strip()]


async def test_includes_hidden_files_not_gitignored(tmp_path):
    tool = create_find_tool(str(tmp_path))
    hidden_dir = tmp_path / ".secret"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.txt").write_text("hidden")
    (tmp_path / "visible.txt").write_text("visible")

    result = await tool.execute("call-1", {"pattern": "**/*.txt", "path": str(tmp_path)})
    output_lines = lines_of(result)

    assert "visible.txt" in output_lines
    assert ".secret/hidden.txt" in output_lines


async def test_respects_gitignore(tmp_path):
    tool = create_find_tool(str(tmp_path))
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("ignored")
    (tmp_path / "kept.txt").write_text("kept")

    result = await tool.execute("call-2", {"pattern": "**/*.txt", "path": str(tmp_path)})
    output = get_text(result)

    assert "kept.txt" in output
    assert "ignored.txt" not in output


@pytest.mark.skip(
    reason="TS 'should surface fd glob parse errors' asserts find('[') rejects with "
    "/error parsing glob|fd exited with code 1|fd error/i. This port never spawns fd; it walks "
    "the tree with fnmatch, for which '[' is simply a literal that matches nothing, so there is "
    "no parse error to surface."
)
async def test_surfaces_fd_glob_parse_errors() -> None: ...


async def test_flag_like_pattern_treated_as_search_text(tmp_path):
    tool = create_find_tool(str(tmp_path))
    (tmp_path / "a.txt").write_text("x")

    result = await tool.execute("call-3", {"pattern": "--help", "path": str(tmp_path)})

    assert get_text(result) == "No files found matching pattern"


async def test_basename_pattern_matches_any_depth(tmp_path):
    tool = create_find_tool(str(tmp_path))
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "deep.ts").write_text("x")
    (tmp_path / "top.ts").write_text("x")
    (tmp_path / "other.txt").write_text("x")

    result = await tool.execute("call-4", {"pattern": "*.ts", "path": str(tmp_path)})
    output_lines = lines_of(result)

    assert "top.ts" in output_lines
    assert "a/b/deep.ts" in output_lines
    assert not any(line.endswith("other.txt") for line in output_lines)


async def test_no_matches_found(tmp_path):
    tool = create_find_tool(str(tmp_path))
    (tmp_path / "a.txt").write_text("x")

    result = await tool.execute("call-5", {"pattern": "*.nonexistent", "path": str(tmp_path)})

    assert get_text(result) == "No files found matching pattern"


async def test_nonexistent_search_path_raises(tmp_path):
    tool = create_find_tool(str(tmp_path))
    missing = tmp_path / "no-such-dir"

    with pytest.raises(RuntimeError, match="Path not found"):
        await tool.execute("call-6", {"pattern": "*.txt", "path": str(missing)})


async def test_result_limit_reached(tmp_path):
    tool = create_find_tool(str(tmp_path))
    for i in range(10):
        (tmp_path / f"file{i:02d}.txt").write_text("x")

    result = await tool.execute("call-7", {"pattern": "*.txt", "path": str(tmp_path), "limit": 3})
    output = get_text(result)

    assert "results limit reached" in output
    assert result.details is not None
    assert result.details.result_limit_reached == 3


async def test_git_directory_always_excluded(tmp_path):
    tool = create_find_tool(str(tmp_path))
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("x")
    (tmp_path / "kept.txt").write_text("x")

    result = await tool.execute("call-8", {"pattern": "**/*", "path": str(tmp_path)})
    output = get_text(result)

    assert ".git" not in output
