"""Python port of `packages/coding-agent/test/edit-tool-no-full-redraw.test.ts`.

The TypeScript test drives a real `TuiMainScreen` with a `ToolExecutionComponent`
for the `edit` tool and asserts on the *boxed diff preview* that the edit tool's
built-in TUI renderer produces (`renderCall`/`renderResult` in
`core/tools/edit.ts`), plus that settling the result does not trigger a full
terminal redraw.

Built-in per-tool renderers are deliberately not ported (see the repository
README: "Built-in per-tool renderers (the bespoke read/edit/bash views in
`core/tools/*.ts`) ... built-in tools use the generic composition path"), so a
Python `ToolExecutionComponent` for `edit` renders pretty-printed arguments, not
a diff. Every assertion that reads the rendered diff, and the full-redraw
counters that only exist to prove the renderer does not force one, are therefore
skipped individually below.

What *is* portable is the data those renders are built from: `compute_edits_diff`
must produce the changed lines the preview shows, and must report the
"Could not find" preflight error instead of a diff when the edits do not apply.
Those are asserted against the real code.
"""

from __future__ import annotations

import os
from pathlib import Path

from pi_coding_agent.tools.edit_diff import Edit, compute_edits_diff

_TARGET_LINES = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]


def _create_large_edits(lines: list[str], count: int = len(_TARGET_LINES)) -> list[Edit]:
    # JavaScript tolerates out-of-range indexes (they stringify as "undefined"),
    # so the TypeScript helper always builds all ten edits and slices after.
    # Python raises, so the slice happens up front instead.
    return [
        Edit(
            old_text=f"{lines[line_number - 1]}\n{lines[line_number]}\n{lines[line_number + 1]}",
            new_text=f"{lines[line_number - 1]}\n{lines[line_number]} changed\n{lines[line_number + 1]}",
        )
        for line_number in _TARGET_LINES[:count]
    ]


def _write_numbered_file(path: Path, count: int) -> list[str]:
    path.write_text("\n".join(f"line {i}" for i in range(count)) + "\n", encoding="utf-8")
    return path.read_text(encoding="utf-8").rstrip("\n").split("\n")


def test_large_diff_contains_every_changed_block(tmp_path: Path) -> None:
    # Skipped: the TypeScript case renders the component into a `TuiMainScreen`
    # and asserts `tui.fullRedraws` / the terminal's `\x1b[2J\x1b[H\x1b[3J`
    # count are unchanged when the result settles. Both counters only matter
    # because the edit tool's built-in renderer swaps a large boxed preview in
    # place; that renderer is not ported, so there is nothing to redraw.
    file_path = tmp_path / "large-edit.txt"
    lines = _write_numbered_file(file_path, 1000)
    edits = _create_large_edits(lines)

    diff = compute_edits_diff(str(file_path), edits, os.getcwd())

    assert "error" not in diff
    assert "line 50 changed" in diff["diff"]
    assert "line 950 changed" in diff["diff"]


def test_settled_result_diff_is_reconstructable_without_the_tool_output_text(tmp_path: Path) -> None:
    # Skipped: `expect(settledRender).not.toContain("Successfully replaced")`
    # asserts the built-in renderer replaces the tool's text output with the
    # diff box. Without the renderer the generic path shows the text output,
    # which is upstream's own behaviour for a tool with no renderer.
    file_path = tmp_path / "replay-edit.txt"
    lines = _write_numbered_file(file_path, 200)
    edits = _create_large_edits(lines, 2)

    diff = compute_edits_diff(str(file_path), edits, os.getcwd())
    assert "error" not in diff

    # The TypeScript case deletes the file before rendering, proving the
    # preview is rebuilt from the stored `details` diff rather than re-read
    # from disk. The same holds here: the diff is a plain value.
    file_path.unlink()

    assert "line 50 changed" in diff["diff"]
    assert "line 150 changed" in diff["diff"]


def test_shows_a_preflight_error_without_a_diff_when_the_edits_do_not_apply(tmp_path: Path) -> None:
    file_path = tmp_path / "missing-edit.txt"
    file_path.write_text("line 0\nline 1\n", encoding="utf-8")

    diff = compute_edits_diff(
        str(file_path),
        [Edit(old_text="does not exist", new_text="replacement")],
        os.getcwd(),
    )

    assert "diff" not in diff
    assert "Could not find" in diff["error"]
