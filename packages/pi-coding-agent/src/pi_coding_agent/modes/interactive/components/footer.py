"""Footer showing cwd, git branch, token stats and context usage.

Ported from ``packages/coding-agent/src/modes/interactive/components/footer.ts``.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from pi_tui.component import Component
from pi_tui.utils import truncate_to_width, visible_width

from ....core.experimental import are_experimental_features_enabled
from ....core.usage_totals import add_usage_to_totals, create_usage_totals
from ....utils.js_number import js_round, to_fixed
from ..theme.theme import theme

if TYPE_CHECKING:
    from ....core.agent_session import AgentSession
    from ....core.footer_data_provider import FooterDataProvider

_CONTROL_WHITESPACE_RE = re.compile(r"[\r\n\t]")
_MULTI_SPACE_RE = re.compile(r" +")


def sanitize_status_text(text: str) -> str:
    """Flatten a status string onto one line."""
    return _MULTI_SPACE_RE.sub(" ", _CONTROL_WHITESPACE_RE.sub(" ", text)).strip()


def format_tokens(count: int) -> str:
    """Compact token count for the footer."""
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{to_fixed(count / 1000, 1)}k"
    if count < 1000000:
        return f"{js_round(count / 1000)}k"
    if count < 10000000:
        return f"{to_fixed(count / 1000000, 1)}M"
    return f"{js_round(count / 1000000)}M"


def format_cwd_for_footer(cwd: str, home: str | None) -> str:
    """Replace the home directory prefix with ``~``."""
    if not home:
        return cwd

    resolved_cwd = os.path.abspath(cwd)
    resolved_home = os.path.abspath(home)
    relative_to_home = os.path.relpath(resolved_cwd, resolved_home)
    if relative_to_home == ".":
        relative_to_home = ""
    is_inside_home = relative_to_home == "" or (
        relative_to_home != ".."
        and not relative_to_home.startswith(f"..{os.sep}")
        and not os.path.isabs(relative_to_home)
    )

    if not is_inside_home:
        return cwd
    return "~" if relative_to_home == "" else f"~{os.sep}{relative_to_home}"


class FooterComponent(Component):
    """Computes token/context stats from the session; git branch and extension
    statuses come from the footer data provider."""

    def __init__(self, session: AgentSession, footer_data: FooterDataProvider) -> None:
        self.session = session
        self.footer_data = footer_data
        self.auto_compact_enabled = True

    def set_session(self, session: AgentSession) -> None:
        self.session = session

    def set_auto_compact_enabled(self, enabled: bool) -> None:
        self.auto_compact_enabled = enabled

    def invalidate(self) -> None:
        """No-op: the git branch is cached and invalidated by the provider."""
        return None

    def dispose(self) -> None:
        """No-op: git watcher cleanup is handled by the provider."""
        return None

    def _collect_usage(self) -> tuple[object, float | None]:
        usage_totals = create_usage_totals()
        latest_cache_hit_rate: float | None = None

        for entry in self.session.session_manager.get_entries():
            entry_type = getattr(entry, "type", None)
            message = getattr(entry, "message", None)
            if entry_type == "message" and message is not None and message.role == "assistant":
                add_usage_to_totals(usage_totals, message.usage)
                latest_prompt_tokens = message.usage.input + message.usage.cache_read + message.usage.cache_write
                latest_cache_hit_rate = (
                    (message.usage.cache_read / latest_prompt_tokens) * 100 if latest_prompt_tokens > 0 else None
                )
            elif (
                entry_type == "message"
                and message is not None
                and message.role == "toolResult"
                and getattr(message, "usage", None)
            ):
                add_usage_to_totals(usage_totals, message.usage)
            elif entry_type in ("branch_summary", "compaction") and getattr(entry, "usage", None):
                add_usage_to_totals(usage_totals, entry.usage)

        return usage_totals, latest_cache_hit_rate

    def _pwd_line_text(self) -> str:
        pwd = format_cwd_for_footer(
            self.session.session_manager.get_cwd(),
            os.environ.get("HOME") or os.environ.get("USERPROFILE"),
        )
        branch = self.footer_data.get_git_branch()
        if branch:
            pwd = f"{pwd} ({branch})"
        session_name = self.session.session_manager.get_session_name()
        if session_name:
            pwd = f"{pwd} • {session_name}"
        return pwd

    def render(self, width: int) -> list[str]:
        state = self.session.state
        usage_totals, latest_cache_hit_rate = self._collect_usage()

        # Context usage comes from the session so compaction is handled: right
        # after a compaction the token count is unknown until the next response.
        # Mirrors the TS `??`/`!== null` chain exactly -- note that a *missing*
        # context usage yields "0.0", and only an explicitly null percent
        # (post-compaction, before the next response) yields "?".
        context_usage = self.session.get_context_usage()
        usage_context_window = getattr(context_usage, "context_window", None)
        model_context_window = getattr(state.model, "context_window", None) if state.model is not None else None
        if usage_context_window is not None:
            context_window = usage_context_window
        elif model_context_window is not None:
            context_window = model_context_window
        else:
            context_window = 0

        usage_percent = getattr(context_usage, "percent", None) if context_usage is not None else None
        context_percent_value = usage_percent if usage_percent is not None else 0.0
        context_percent = (
            "?" if (context_usage is not None and usage_percent is None) else to_fixed(context_percent_value, 1)
        )

        pwd = self._pwd_line_text()

        stats_parts: list[str] = []
        if usage_totals.input:
            stats_parts.append(f"↑{format_tokens(usage_totals.input)}")
        if usage_totals.output:
            stats_parts.append(f"↓{format_tokens(usage_totals.output)}")
        if usage_totals.cache_read:
            stats_parts.append(f"R{format_tokens(usage_totals.cache_read)}")
        if usage_totals.cache_write:
            stats_parts.append(f"W{format_tokens(usage_totals.cache_write)}")
        if (usage_totals.cache_read > 0 or usage_totals.cache_write > 0) and latest_cache_hit_rate is not None:
            stats_parts.append(f"CH{to_fixed(latest_cache_hit_rate, 1)}%")

        # Kimi Coding is subscription-backed despite using API-key auth.
        using_subscription = False
        if state.model is not None:
            using_subscription = (
                state.model.provider == "kimi-coding"
                or self.session.model_runtime.is_using_subscription(state.model.provider)
            )
        if usage_totals.cost or using_subscription:
            stats_parts.append(f"${to_fixed(usage_totals.cost, 3)}{' (sub)' if using_subscription else ''}")

        auto_indicator = " (auto)" if self.auto_compact_enabled else ""
        if context_percent == "?":
            context_percent_display = f"?/{format_tokens(context_window)}{auto_indicator}"
        else:
            context_percent_display = f"{context_percent}%/{format_tokens(context_window)}{auto_indicator}"
        if context_percent_value > 90:
            stats_parts.append(theme.fg("error", context_percent_display))
        elif context_percent_value > 70:
            stats_parts.append(theme.fg("warning", context_percent_display))
        else:
            stats_parts.append(context_percent_display)

        if are_experimental_features_enabled():
            stats_parts.append(f"{theme.fg('dim', '•')} {theme.bold(theme.fg('warning', 'xp'))}")

        stats_left = " ".join(stats_parts)
        model_name = state.model.id if state.model is not None else "no-model"

        stats_left_width = visible_width(stats_left)
        if stats_left_width > width:
            stats_left = truncate_to_width(stats_left, width, "...")
            stats_left_width = visible_width(stats_left)

        min_padding = 2

        right_side_without_provider = model_name
        if state.model is not None and getattr(state.model, "reasoning", None):
            thinking_level = state.thinking_level or "off"
            right_side_without_provider = (
                f"{model_name} • thinking off" if thinking_level == "off" else f"{model_name} • {thinking_level}"
            )

        right_side = right_side_without_provider
        if self.footer_data.get_available_provider_count() > 1 and state.model is not None:
            right_side = f"({state.model.provider}) {right_side_without_provider}"
            if stats_left_width + min_padding + visible_width(right_side) > width:
                right_side = right_side_without_provider

        right_side_width = visible_width(right_side)
        if stats_left_width + min_padding + right_side_width <= width:
            padding = " " * (width - stats_left_width - right_side_width)
            stats_line = stats_left + padding + right_side
        else:
            available_for_right = width - stats_left_width - min_padding
            if available_for_right > 0:
                truncated_right = truncate_to_width(right_side, available_for_right, "")
                padding = " " * max(0, width - stats_left_width - visible_width(truncated_right))
                stats_line = stats_left + padding + truncated_right
            else:
                stats_line = stats_left

        # `stats_left` can contain colour codes ending in a reset, which would
        # cancel an outer dim wrapper, so dim the two halves separately.
        dim_stats_left = theme.fg("dim", stats_left)
        dim_remainder = theme.fg("dim", stats_line[len(stats_left) :])

        lines = [
            truncate_to_width(theme.fg("dim", pwd), width, theme.fg("dim", "...")),
            dim_stats_left + dim_remainder,
        ]

        extension_statuses = self.footer_data.get_extension_statuses()
        if len(extension_statuses) > 0:
            status_line = " ".join(
                sanitize_status_text(text) for _key, text in sorted(extension_statuses.items(), key=lambda kv: kv[0])
            )
            lines.append(truncate_to_width(status_line, width, theme.fg("dim", "...")))

        return lines


__all__ = ["FooterComponent", "format_cwd_for_footer", "format_tokens", "sanitize_status_text"]
