"""Python port of `packages/coding-agent/test/suite/regressions/3302-find-path-glob.test.ts`.

The TypeScript `find` tool shells out to `fd`; the bug was that `fd --glob`
matches basenames only unless `--full-path` is set, so any pattern containing
a `/` returned nothing. This port walks the tree in Python
(`tools/find.py`) and switches to full-path matching when the pattern
contains `/`, so the same patterns must produce the same results.
"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.tools.find import create_find_tool


async def _run_find(root: Path, pattern: str) -> list[str]:
    tool = create_find_tool(str(root))
    assert tool.execute is not None
    result = await tool.execute("call-1", {"pattern": pattern}, None, None)
    text = result.content[0].text if result.content else ""
    if text == "No files found matching pattern":
        return []
    return [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("[")]


def _make_tree(tmp_path: Path) -> Path:
    (tmp_path / "some" / "parent" / "child").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "foo" / "bar").mkdir(parents=True, exist_ok=True)
    (tmp_path / "some" / "parent" / "child" / "file.ext").write_text("", encoding="utf-8")
    (tmp_path / "some" / "parent" / "child" / "test.spec.ts").write_text("", encoding="utf-8")
    (tmp_path / "src" / "foo" / "bar" / "example.spec.ts").write_text("", encoding="utf-8")
    return tmp_path


async def test_basename_pattern_still_matches(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    files = await _run_find(root, "*.spec.ts")
    assert sorted(files) == ["some/parent/child/test.spec.ts", "src/foo/bar/example.spec.ts"]


async def test_directory_prefixed_pattern_with_tail_matches_subtree(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    files = await _run_find(root, "some/parent/child/**")
    assert "some/parent/child/file.ext" in files
    assert "some/parent/child/test.spec.ts" in files


async def test_leading_wildcard_with_path_segments_matches(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    files = await _run_find(root, "**/parent/child/*")
    assert "some/parent/child/file.ext" in files
    assert "some/parent/child/test.spec.ts" in files


async def test_src_glob_matches_nested_spec_file(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    files = await _run_find(root, "src/**/*.spec.ts")
    assert files == ["src/foo/bar/example.spec.ts"]
