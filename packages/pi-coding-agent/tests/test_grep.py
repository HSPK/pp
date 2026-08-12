"""Tests for the `grep` tool. Ported from `tools.test.ts` (describe("grep tool")).

Covers both the `rg`-subprocess path (used when `rg` is installed, as in
this environment) and, separately, the pure-Python fallback scan by forcing
`ensure_tool` to report `rg` as unavailable.
"""

from __future__ import annotations

import os
import stat

import pytest
from pi_coding_agent.tools.grep import create_grep_tool
from pi_coding_agent.utils import tools_manager


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


@pytest.fixture(params=["rg", "fallback"])
def grep_variant(request, monkeypatch):
    """Run each test twice: once using the real rg binary, once forcing the pure-Python fallback."""
    if request.param == "fallback":
        # Patch the resolver rather than `shutil.which`: `ensure_tool` would
        # otherwise try to download `rg` from GitHub during the test run.
        async def unavailable(tool, silent=True, bin_dir=None):
            return None

        monkeypatch.setattr(tools_manager, "ensure_tool", unavailable)
    return request.param


async def test_includes_filename_for_single_file_search(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    test_file = tmp_path / "example.txt"
    test_file.write_text("first line\nmatch line\nlast line")

    result = await tool.execute("call-1", {"pattern": "match", "path": str(test_file)})
    output = get_text(result)

    assert "example.txt:2: match line" in output


async def test_context_lines_and_limit(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    test_file = tmp_path / "context.txt"
    test_file.write_text("\n".join(["before", "match one", "after", "middle", "match two", "after two"]))

    result = await tool.execute("call-2", {"pattern": "match", "path": str(test_file), "limit": 1, "context": 1})
    output = get_text(result)

    assert "context.txt-1- before" in output
    assert "context.txt:2: match one" in output
    assert "context.txt-3- after" in output
    assert "[1 matches limit reached. Use limit=2 for more, or refine pattern]" in output
    assert "match two" not in output


async def test_flag_like_pattern_treated_as_search_text(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    marker = tmp_path / "grep-injection-marker"
    payload = tmp_path / "payload.sh"
    test_file = tmp_path / "target.txt"
    payload.write_text(f'#!/bin/sh\necho executed > {marker}\ncat "$1"\n')
    os.chmod(payload, os.stat(payload).st_mode | stat.S_IEXEC)
    test_file.write_text("target\n")

    result = await tool.execute("call-3", {"pattern": f"--pre={payload}", "path": str(tmp_path)})

    assert get_text(result) == "No matches found"
    assert not marker.exists()


async def test_no_matches_found(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    (tmp_path / "a.txt").write_text("nothing interesting here")

    result = await tool.execute("call-4", {"pattern": "zzz-not-present", "path": str(tmp_path)})

    assert get_text(result) == "No matches found"


async def test_nonexistent_path_raises(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    missing = tmp_path / "no-such"

    with pytest.raises(RuntimeError, match="Path not found"):
        await tool.execute("call-5", {"pattern": "x", "path": str(missing)})


async def test_ignore_case(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    test_file = tmp_path / "case.txt"
    test_file.write_text("Hello World\n")

    result = await tool.execute("call-6", {"pattern": "hello", "path": str(test_file), "ignoreCase": True})

    assert "Hello World" in get_text(result)


async def test_literal_pattern_treats_regex_metacharacters_as_text(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    test_file = tmp_path / "literal.txt"
    test_file.write_text("price: $5.00 (discounted)\n")

    result = await tool.execute("call-7", {"pattern": "$5.00 (discounted)", "path": str(test_file), "literal": True})

    assert "$5.00 (discounted)" in get_text(result)


async def test_glob_filter(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    (tmp_path / "a.ts").write_text("needle\n")
    (tmp_path / "b.txt").write_text("needle\n")

    result = await tool.execute("call-8", {"pattern": "needle", "path": str(tmp_path), "glob": "*.ts"})
    output = get_text(result)

    assert "a.ts" in output
    assert "b.txt" not in output


async def test_respects_gitignore(tmp_path, grep_variant):
    # rg only honors .gitignore by default when a `.git` directory is present
    # (matching this port's grep.py, which does not pass `--no-require-git`,
    # exactly like the TypeScript grep.ts); a bare `.git` dir is sufficient.
    (tmp_path / ".git").mkdir()
    tool = create_grep_tool(str(tmp_path))
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("needle\n")
    (tmp_path / "kept.txt").write_text("needle\n")

    result = await tool.execute("call-9", {"pattern": "needle", "path": str(tmp_path)})
    output = get_text(result)

    assert "kept.txt" in output
    assert "ignored.txt" not in output


async def test_long_line_truncated(tmp_path, grep_variant):
    tool = create_grep_tool(str(tmp_path))
    test_file = tmp_path / "long.txt"
    test_file.write_text(f"needle {'x' * 1000}\n")

    result = await tool.execute("call-10", {"pattern": "needle", "path": str(test_file)})
    output = get_text(result)

    assert "[truncated]" in output
    assert "Some lines truncated to 500 chars" in output
