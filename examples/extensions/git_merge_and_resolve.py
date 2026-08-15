"""Merge and Resolve.

Python port of `packages/coding-agent/examples/extensions/git-merge-and-resolve.ts`.

Keeps the working branch up to date with its upstream tracking ref.
After each agent turn, fetches and merges. Clean merges complete
silently. When conflicts arise, the working tree is left dirty and
the agent receives a follow-up message listing each conflict block
with file, line range, and ours/theirs sections so it can resolve them.
Also re-sends unresolved conflicts from a previous incomplete merge.

Start pi with this extension::

    pi -e ./examples/extensions/git_merge_and_resolve.py
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import AgentEndEvent, ExtensionContext


@dataclass
class ConflictBlock:
    file: str
    start_line: int
    separator_line: int
    end_line: int


async def find_conflicts(pi: ExtensionAPI, cwd: str) -> list[ConflictBlock]:
    """Parse conflict markers from working tree files with unmerged paths."""
    result = await pi.exec("git", ["diff", "--name-only", "--diff-filter=U"])
    if result.code != 0 or not result.stdout.strip():
        return []

    blocks: list[ConflictBlock] = []
    for file in result.stdout.strip().split("\n"):
        with contextlib.suppress(OSError, UnicodeDecodeError):
            text = Path(cwd, file).read_text(encoding="utf-8")
            block_start: int | None = None
            separator_line: int | None = None
            for line_no, line in enumerate(text.split("\n"), start=1):
                if line.startswith("<<<<<<<"):
                    block_start = line_no
                    separator_line = None
                elif line.startswith("=======") and block_start is not None:
                    separator_line = line_no
                elif line.startswith(">>>>>>>") and block_start is not None and separator_line is not None:
                    blocks.append(
                        ConflictBlock(
                            file=file,
                            start_line=block_start,
                            separator_line=separator_line,
                            end_line=line_no,
                        )
                    )
                    block_start = None
                    separator_line = None
    return blocks


def format_range(start: int, end: int) -> str:
    if start > end:
        return "empty"
    if start == end:
        return str(start)
    return f"{start}-{end}"


def format_conflicts(ref: str, blocks: list[ConflictBlock]) -> str:
    lines = [f"Merged {ref} with conflicts:", ""]
    for b in blocks:
        ours = format_range(b.start_line + 1, b.separator_line - 1)
        theirs = format_range(b.separator_line + 1, b.end_line - 1)
        lines.append(f"  {b.file}:{b.start_line}-{b.end_line} (ours {ours}, theirs {theirs})")
    lines.extend(["", "Resolve these conflicts."])
    return "\n".join(lines)


def pi_extension(pi: ExtensionAPI) -> None:
    async def on_agent_end(_event: AgentEndEvent, ctx: ExtensionContext) -> None:
        rev_parse = await pi.exec("git", ["rev-parse", "--git-dir"])
        if rev_parse.code != 0:
            return

        ref = "MERGE_HEAD"

        # If not already in a merge, attempt one
        merge_head = await pi.exec("git", ["rev-parse", "MERGE_HEAD"])
        if merge_head.code != 0:
            # Only attempt a new merge if the working tree is clean
            status = await pi.exec("git", ["status", "--porcelain"])
            if status.stdout.strip():
                return

            upstream = await pi.exec("git", ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
            if upstream.code != 0:
                return

            ref = upstream.stdout.strip()
            remote = ref.split("/")[0]
            ctx.ui.notify(f"git-merge-and-resolve: fetching {remote}, merging {ref}", "info")

            fetch = await pi.exec("git", ["fetch", remote])
            if fetch.code != 0:
                ctx.ui.notify(f"git-merge-and-resolve: fetch failed: {fetch.stderr.strip()}", "warning")
                return

            merge = await pi.exec("git", ["merge", "--no-ff", ref])
            if merge.code == 0:
                return

        # Either we just merged with conflicts, or we were already in an unfinished merge
        conflicts = await find_conflicts(pi, ctx.cwd)
        if not conflicts:
            return

        pi.send_user_message(format_conflicts(ref, conflicts), deliver_as="followUp")

    pi.on("agent_end", on_agent_end)
