"""Python port of `packages/coding-agent/test/suite/regressions/3303-find-nested-gitignore.test.ts`.

Each `.gitignore` must apply only to its own subtree: `a/.gitignore` must not
filter files under a sibling `b/`. The TypeScript bug came from passing every
collected `.gitignore` to `fd --ignore-file` (one global source); this port
builds a hierarchical matcher in `tools/gitignore.py` instead, and the same
expectations must hold.
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
    return sorted(line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("["))


async def test_applies_a_gitignore_only_inside_a(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    (tmp_path / "a" / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "a" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "root.txt").write_text("", encoding="utf-8")

    files = await _run_find(tmp_path, "**/*.txt")
    assert files == ["a/kept.txt", "b/ignored.txt", "b/kept.txt", "root.txt"]


async def test_scopes_each_gitignore_to_its_own_subtree(tmp_path: Path) -> None:
    (tmp_path / "a" / "deep").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    (tmp_path / "a" / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "a" / "deep" / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (tmp_path / "a" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "deep" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "deep" / "secret.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "deep" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "root.txt").write_text("", encoding="utf-8")

    files = await _run_find(tmp_path, "**/*.txt")
    assert files == ["a/deep/kept.txt", "a/kept.txt", "b/ignored.txt", "b/kept.txt", "root.txt"]
