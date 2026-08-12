"""Python port of `packages/coding-agent/test/suite/regressions/6104-find-root-relativization.test.ts`.

The TypeScript regression is in `relativizeFindResultPath`, a helper that only
exists because the TS `find` shells out to `fd` and gets absolute paths back
that it slices at `searchPath.length + 1` -- which dropped the first character
of the first segment when the search path was a root (`/`, `I:\\`) because
`path.resolve` keeps the trailing separator there.

This port walks the tree itself and builds results with `os.path.relpath`, so
there is no slicing helper to unit test (see `find.py`'s module docstring). The
observable behavior is ported instead: results stay correctly relativized when
the search path carries a trailing separator, and a search path that is a name
prefix of a sibling directory does not swallow that sibling.

Disposition of all eleven TypeScript cases:

* "preserves the first segment and emits one trailing slash for fd directory
  output", "handles fd output that uses forward slashes under a drive root",
  "normalizes relative custom-glob results without corrupting them" (all under
  `Windows drive root`) -- Windows drive roots and backslash-separated `fd`
  output cannot occur here: `os.walk` yields native entries under a POSIX root
  and the helper under test does not exist.
* "keeps deeper search paths unchanged" ->
  `test_relativizes_against_a_deeper_search_path`.
* "does not relativize a sibling directory that shares a name prefix" ->
  `test_does_not_include_a_sibling_directory_that_shares_a_name_prefix`.
* "preserves the first segment for files under /" and "preserves the first
  segment and one trailing slash for directories under /" -> the trailing
  separator half is covered by
  `test_preserves_the_first_segment_when_the_search_path_has_a_trailing_separator`;
  literally searching `/` is not testable without walking the real filesystem
  root, and this port never emits directory results at all (the walk yields
  files only), so there is no trailing-slash form to assert.
* "preserves backslashes in POSIX filenames" ->
  `test_preserves_backslashes_in_posix_filenames`.
* "falls back to path.relative when the absolute paths do not share a prefix"
  and "keeps a trailing slash on directories resolved through path.relative"
  -- unreachable: every result comes from walking `search_path`, so no result
  can lie outside it and need a `../..` relative form.
* "relativizes custom glob results against a root search path" -- needs
  `FindOperations` (the injectable `exists`/`glob` used to delegate search to
  remote systems), which this port does not have.
"""

from __future__ import annotations

import os
from pathlib import Path

from pi_coding_agent.tools.find import create_find_tool


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")


async def test_preserves_the_first_segment_when_the_search_path_has_a_trailing_separator(tmp_path: Path) -> None:
    (tmp_path / "AI/Models/TextGen/gemma4").mkdir(parents=True)
    (tmp_path / "AI/Models/TextGen/gemma4/file.txt").write_text("x", encoding="utf-8")
    find = create_find_tool(str(tmp_path))

    result = await find.execute("call-1", {"pattern": "**/*.txt", "path": f"{tmp_path}{os.sep}"})

    assert _text(result) == "AI/Models/TextGen/gemma4/file.txt"


async def test_relativizes_against_a_deeper_search_path(tmp_path: Path) -> None:
    (tmp_path / "AI/Models/TextGen").mkdir(parents=True)
    (tmp_path / "AI/Models/TextGen/file.txt").write_text("x", encoding="utf-8")
    find = create_find_tool(str(tmp_path))

    result = await find.execute("call-2", {"pattern": "**/*.txt", "path": str(tmp_path / "AI")})

    assert _text(result) == "Models/TextGen/file.txt"


async def test_does_not_include_a_sibling_directory_that_shares_a_name_prefix(tmp_path: Path) -> None:
    (tmp_path / "AI/Models").mkdir(parents=True)
    (tmp_path / "AI/Models2").mkdir(parents=True)
    (tmp_path / "AI/Models/inside.txt").write_text("x", encoding="utf-8")
    (tmp_path / "AI/Models2/file.txt").write_text("x", encoding="utf-8")
    find = create_find_tool(str(tmp_path))

    result = await find.execute("call-3", {"pattern": "**/*.txt", "path": str(tmp_path / "AI/Models")})

    assert _text(result) == "inside.txt"


async def test_preserves_backslashes_in_posix_filenames(tmp_path: Path) -> None:
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "file\\").write_text("x", encoding="utf-8")
    find = create_find_tool(str(tmp_path))

    result = await find.execute("call-4", {"pattern": "*", "path": str(tmp_path / "home")})

    assert _text(result) == "file\\"
