"""Pure utility functions for plan mode.

Python port of `packages/coding-agent/examples/extensions/plan-mode/utils.ts`.

Only the pure helpers are ported. The extension entry point
(`plan-mode/index.ts`) drives a progress widget and a `Ctrl+Alt+P` shortcut
through the extension UI host, which this port does not have (see the
top-level README's "Not ported, by decision" list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Destructive commands blocked in plan mode.
DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bmv\b", re.IGNORECASE),
    re.compile(r"\bcp\b", re.IGNORECASE),
    re.compile(r"\bmkdir\b", re.IGNORECASE),
    re.compile(r"\btouch\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"\bchgrp\b", re.IGNORECASE),
    re.compile(r"\bln\b", re.IGNORECASE),
    re.compile(r"\btee\b", re.IGNORECASE),
    re.compile(r"\btruncate\b", re.IGNORECASE),
    re.compile(r"\bdd\b", re.IGNORECASE),
    re.compile(r"\bshred\b", re.IGNORECASE),
    re.compile(r"(^|[^<])>(?!>)"),
    re.compile(r">>"),
    re.compile(r"\bnpm\s+(install|uninstall|update|ci|link|publish)", re.IGNORECASE),
    re.compile(r"\byarn\s+(add|remove|install|publish)", re.IGNORECASE),
    re.compile(r"\bpnpm\s+(add|remove|install|publish)", re.IGNORECASE),
    re.compile(r"\bpip\s+(install|uninstall)", re.IGNORECASE),
    re.compile(r"\bapt(-get)?\s+(install|remove|purge|update|upgrade)", re.IGNORECASE),
    re.compile(r"\bbrew\s+(install|uninstall|upgrade)", re.IGNORECASE),
    re.compile(
        r"\bgit\s+(add|commit|push|pull|merge|rebase|reset|checkout|branch\s+-[dD]|stash|cherry-pick|revert|tag|init|clone)",
        re.IGNORECASE,
    ),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bsu\b", re.IGNORECASE),
    re.compile(r"\bkill\b", re.IGNORECASE),
    re.compile(r"\bpkill\b", re.IGNORECASE),
    re.compile(r"\bkillall\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bsystemctl\s+(start|stop|restart|enable|disable)", re.IGNORECASE),
    re.compile(r"\bservice\s+\S+\s+(start|stop|restart)", re.IGNORECASE),
    re.compile(r"\b(vim?|nano|emacs|code|subl)\b", re.IGNORECASE),
)

# Safe read-only commands allowed in plan mode.
SAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*cat\b"),
    re.compile(r"^\s*head\b"),
    re.compile(r"^\s*tail\b"),
    re.compile(r"^\s*less\b"),
    re.compile(r"^\s*more\b"),
    re.compile(r"^\s*grep\b"),
    re.compile(r"^\s*find\b"),
    re.compile(r"^\s*ls\b"),
    re.compile(r"^\s*pwd\b"),
    re.compile(r"^\s*echo\b"),
    re.compile(r"^\s*printf\b"),
    re.compile(r"^\s*wc\b"),
    re.compile(r"^\s*sort\b"),
    re.compile(r"^\s*uniq\b"),
    re.compile(r"^\s*diff\b"),
    re.compile(r"^\s*file\b"),
    re.compile(r"^\s*stat\b"),
    re.compile(r"^\s*du\b"),
    re.compile(r"^\s*df\b"),
    re.compile(r"^\s*tree\b"),
    re.compile(r"^\s*which\b"),
    re.compile(r"^\s*whereis\b"),
    re.compile(r"^\s*type\b"),
    re.compile(r"^\s*env\b"),
    re.compile(r"^\s*printenv\b"),
    re.compile(r"^\s*uname\b"),
    re.compile(r"^\s*whoami\b"),
    re.compile(r"^\s*id\b"),
    re.compile(r"^\s*date\b"),
    re.compile(r"^\s*cal\b"),
    re.compile(r"^\s*uptime\b"),
    re.compile(r"^\s*ps\b"),
    re.compile(r"^\s*top\b"),
    re.compile(r"^\s*htop\b"),
    re.compile(r"^\s*free\b"),
    re.compile(r"^\s*git\s+(status|log|diff|show|branch|remote|config\s+--get)", re.IGNORECASE),
    re.compile(r"^\s*git\s+ls-", re.IGNORECASE),
    re.compile(r"^\s*npm\s+(list|ls|view|info|search|outdated|audit)", re.IGNORECASE),
    re.compile(r"^\s*yarn\s+(list|info|why|audit)", re.IGNORECASE),
    re.compile(r"^\s*node\s+--version", re.IGNORECASE),
    re.compile(r"^\s*python\s+--version", re.IGNORECASE),
    re.compile(r"^\s*curl\s", re.IGNORECASE),
    re.compile(r"^\s*wget\s+-O\s*-", re.IGNORECASE),
    re.compile(r"^\s*jq\b"),
    re.compile(r"^\s*sed\s+-n", re.IGNORECASE),
    re.compile(r"^\s*awk\b"),
    re.compile(r"^\s*rg\b"),
    re.compile(r"^\s*fd\b"),
    re.compile(r"^\s*bat\b"),
    re.compile(r"^\s*eza\b"),
)


def is_safe_command(command: str) -> bool:
    is_destructive = any(p.search(command) for p in DESTRUCTIVE_PATTERNS)
    is_safe = any(p.search(command) for p in SAFE_PATTERNS)
    return not is_destructive and is_safe


@dataclass
class TodoItem:
    step: int
    text: str
    completed: bool


_MARKDOWN_EMPHASIS = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_MARKDOWN_CODE = re.compile(r"`([^`]+)`")
_LEADING_ACTION_WORD = re.compile(
    r"^(Use|Run|Execute|Create|Write|Read|Check|Verify|Update|Modify|Add|Remove|Delete|Install)\s+(the\s+)?",
    re.IGNORECASE,
)
_WHITESPACE_RUN = re.compile(r"\s+")


def clean_step_text(text: str) -> str:
    cleaned = _MARKDOWN_EMPHASIS.sub(r"\1", text)
    cleaned = _MARKDOWN_CODE.sub(r"\1", cleaned)
    cleaned = _LEADING_ACTION_WORD.sub("", cleaned, count=1)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()

    if len(cleaned) > 0:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > 50:
        cleaned = f"{cleaned[:47]}..."
    return cleaned


_PLAN_HEADER = re.compile(r"\*{0,2}Plan:\*{0,2}\s*\n", re.IGNORECASE)
_NUMBERED_STEP = re.compile(r"^\s*(\d+)[.)]\s+\*{0,2}([^*\n]+)", re.MULTILINE)
_TRAILING_EMPHASIS = re.compile(r"\*{1,2}\Z")


def extract_todo_items(message: str) -> list[TodoItem]:
    items: list[TodoItem] = []
    header_match = _PLAN_HEADER.search(message)
    if not header_match:
        return items

    header = header_match.group(0)
    plan_section = message[message.index(header) + len(header) :]

    for match in _NUMBERED_STEP.finditer(plan_section):
        text = _TRAILING_EMPHASIS.sub("", match.group(2).strip()).strip()
        if len(text) > 5 and not text.startswith("`") and not text.startswith("/") and not text.startswith("-"):
            cleaned = clean_step_text(text)
            if len(cleaned) > 3:
                items.append(TodoItem(step=len(items) + 1, text=cleaned, completed=False))
    return items


_DONE_MARKER = re.compile(r"\[DONE:(\d+)\]", re.IGNORECASE)


def extract_done_steps(message: str) -> list[int]:
    return [int(match.group(1)) for match in _DONE_MARKER.finditer(message)]


def mark_completed_steps(text: str, items: list[TodoItem]) -> int:
    done_steps = extract_done_steps(text)
    for step in done_steps:
        item = next((t for t in items if t.step == step), None)
        if item:
            item.completed = True
    return len(done_steps)
