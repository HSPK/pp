"""Minimal `.gitignore`-semantics pattern matcher.

The TypeScript coding-agent tools (`find.ts`, `grep.ts`) rely on the `ignore`
npm package (via `fd`/`rg`'s built-in git-awareness) to skip files excluded by
`.gitignore`. Python has no equivalent in the standard library and the porting
conventions forbid adding a third-party dependency for this, so this module
implements the subset of gitignore pattern syntax the tools need: `*`, `**`,
`?`, `!` negation, a leading `/` (anchor to the `.gitignore`'s directory), and
a trailing `/` (directory-only patterns). Character classes (`[abc]`) are also
supported since they fall out of the same translation naturally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _CompiledPattern:
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


def _translate_glob_segment(pattern: str) -> str:
    """Translate gitignore glob syntax into a regex fragment (unanchored)."""
    i = 0
    n = len(pattern)
    out: list[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _compile_pattern(raw: str) -> _CompiledPattern | None:
    pattern = raw.rstrip("\n\r")
    if not pattern or pattern.startswith("#"):
        return None

    negate = pattern.startswith("!")
    if negate:
        pattern = pattern[1:]
    # A backslash-escaped leading `!` or `#` loses its special meaning; only
    # relevant for hand-authored gitignore files, so handle the common case.
    if pattern.startswith("\\!") or pattern.startswith("\\#"):
        pattern = pattern[1:]

    dir_only = pattern.endswith("/") and not pattern.endswith("\\/")
    if dir_only:
        pattern = pattern[:-1]
    if not pattern:
        return None

    # A pattern is anchored to its `.gitignore` directory if it contains a
    # `/` anywhere except as the very last character (already stripped above).
    anchored = pattern.startswith("/") or "/" in pattern
    if pattern.startswith("/"):
        pattern = pattern[1:]

    translated = _translate_glob_segment(pattern)
    if anchored:
        regex = re.compile(f"^{translated}$")
    else:
        regex = re.compile(f"^(?:.*/)?{translated}$")

    return _CompiledPattern(regex=regex, negate=negate, dir_only=dir_only)


@dataclass
class GitignoreMatcher:
    """Matches POSIX-style relative paths against a set of gitignore patterns.

    Call :meth:`add` with the lines of one `.gitignore` file, in root-to-leaf
    order (patterns added later take precedence, matching git's rule that a
    deeper `.gitignore` overrides a shallower one). Patterns are matched
    against paths relative to the root the matcher was built for.
    """

    _patterns: list[_CompiledPattern] = field(default_factory=list)

    def add(self, lines: list[str]) -> None:
        for line in lines:
            compiled = _compile_pattern(line)
            if compiled is not None:
                self._patterns.append(compiled)

    def add_text(self, text: str) -> None:
        self.add(text.splitlines())

    def is_ignored(self, relative_posix_path: str, is_dir: bool) -> bool:
        """Whether ``relative_posix_path`` (posix separators, no leading `/`) is ignored.

        The last matching pattern wins, mirroring `.gitignore` precedence
        (later patterns, including `!` negations, override earlier ones).

        A trailing `/` marks the path as a directory and is stripped before
        matching, as the npm `ignore` package does: `ignores("venv/")` is true
        for the pattern `venv`.
        """
        if relative_posix_path.endswith("/"):
            relative_posix_path = relative_posix_path[:-1]
            is_dir = True
        if not relative_posix_path:
            return False
        ignored = False
        for compiled in self._patterns:
            if compiled.dir_only and not is_dir:
                continue
            if compiled.regex.match(relative_posix_path):
                ignored = not compiled.negate
        return ignored


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a `find`-tool glob pattern (e.g. ``*.ts``, ``**/*.json``) to a full-match regex."""
    return re.compile(f"^{_translate_glob_segment(pattern)}$")


def build_matcher_for_tree(root: str) -> GitignoreMatcher:
    """Build a matcher by collecting every `.gitignore` under ``root``.

    Patterns from a `.gitignore` in a subdirectory are prefixed with that
    subdirectory's path so they only apply within their own subtree, and are
    added after the root's patterns so they take precedence there, matching
    git's directory-scoped override behavior.
    """
    matcher = GitignoreMatcher()
    root_path = Path(root)
    if not root_path.is_dir():
        return matcher

    for gitignore_path in sorted(root_path.rglob(".gitignore"), key=lambda p: len(p.parts)):
        try:
            text = gitignore_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel_dir = gitignore_path.parent.relative_to(root_path).as_posix()
        prefix = "" if rel_dir == "." else f"{rel_dir}/"
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.startswith("#"):
                continue
            if not prefix:
                lines.append(line)
                continue
            negate = line.startswith("!")
            body = line[1:] if negate else line
            if body.startswith("/"):
                body = body[1:]
            # An unanchored pattern (no `/` except a trailing directory marker)
            # applies at every depth below its `.gitignore`, so prefixing it
            # with the directory must keep that "any depth" wildcard --
            # otherwise `a/.gitignore`'s `ignored.txt` would only match
            # `a/ignored.txt` and miss `a/deep/ignored.txt`.
            unanchored_body = body[:-1] if body.endswith("/") else body
            prefixed = f"{prefix}{body}" if "/" in unanchored_body else f"{prefix}**/{body}"
            lines.append(f"!{prefixed}" if negate else prefixed)
        matcher.add(lines)

    return matcher
