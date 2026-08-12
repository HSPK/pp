"""Find files by glob pattern, respecting `.gitignore`.

Python port of `packages/coding-agent/src/core/tools/find.ts`. The TypeScript
version shells out to `fd` (downloading it on demand via `ensureTool`); this
port always uses a pure-Python directory walk plus the local gitignore
matcher (`pi_coding_agent.tools.gitignore`), per the porting brief's
"no added dependency" rule.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.gitignore import build_matcher_for_tree, compile_glob
from pi_coding_agent.tools.path_utils import resolve_to_cwd
from pi_coding_agent.tools.truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head

_DEFAULT_LIMIT = 1000


@dataclass
class FindToolDetails:
    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


def _walk_matches(search_path: str, pattern: str, limit: int) -> tuple[list[str], bool]:
    """Walk `search_path`, applying gitignore filtering, and return files matching `pattern`.

    Returns `(relative_posix_paths, limit_reached)`. Matching mirrors the
    common glob convention: a pattern containing `/` matches the full
    relative path, otherwise only the basename is matched (so `*.ts` matches
    any `.ts` file at any depth, while `src/**/*.ts` is anchored).
    """
    matcher = build_matcher_for_tree(search_path)
    match_full_path = "/" in pattern
    regex = compile_glob(pattern)

    results: list[str] = []
    limit_reached = False

    for dirpath, dirnames, filenames in os.walk(search_path):
        rel_dir = os.path.relpath(dirpath, search_path)
        rel_dir_posix = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")

        # Prune ignored/VCS directories before descending, mirroring fd's default behavior.
        kept_dirnames = []
        for name in sorted(dirnames):
            if name == ".git":
                continue
            rel_child = name if not rel_dir_posix else f"{rel_dir_posix}/{name}"
            if matcher.is_ignored(rel_child, is_dir=True):
                continue
            kept_dirnames.append(name)
        dirnames[:] = kept_dirnames

        for name in sorted(filenames):
            rel_file = name if not rel_dir_posix else f"{rel_dir_posix}/{name}"
            if matcher.is_ignored(rel_file, is_dir=False):
                continue
            candidate = rel_file if match_full_path else name
            if not regex.match(candidate):
                continue
            results.append(rel_file)
            if len(results) >= limit:
                limit_reached = True
                return results, limit_reached

    return results, limit_reached


def create_find_tool(cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        pattern = params["pattern"]
        search_dir = params.get("path")
        limit = params.get("limit")
        search_path = resolve_to_cwd(search_dir or ".", cwd)
        effective_limit = limit if limit is not None else _DEFAULT_LIMIT

        if not os.path.exists(search_path):
            raise RuntimeError(f"Path not found: {search_path}")

        results, limit_reached = _walk_matches(search_path, pattern, effective_limit)

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        if not results:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")])

        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=2**31)
        output = truncation.content
        details = FindToolDetails()
        notices: list[str] = []
        if limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.result_limit_reached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.truncation is not None or details.result_limit_reached is not None
        return AgentToolResult(content=[TextContent(text=output)], details=details if has_details else None)

    return AgentTool(
        name="find",
        description=(
            "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
            f"Respects .gitignore. Output is truncated to {_DEFAULT_LIMIT} results or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'",
                },
                "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )
