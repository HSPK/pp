"""Python port of `packages/coding-agent/test/footer-width.test.ts`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pi_ai.types import Cost, Usage
from pi_coding_agent.modes.interactive.components.footer import FooterComponent, format_cwd_for_footer
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.utils import visible_width


def _usage(input_: int, output: int, cache_read: int, cache_write: int, cost_total: float) -> Usage:
    return Usage(
        input=input_,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        cost=Cost(total=cost_total),
    )


def _create_session(
    *,
    session_name: str,
    model_id: str = "test-model",
    provider: str = "test",
    reasoning: bool = False,
    thinking_level: str = "off",
    usage: Usage | None = None,
    branch_usage: Usage | None = None,
    compaction_usage: Usage | None = None,
    tool_usage: Usage | None = None,
    using_subscription: bool = False,
) -> Any:
    entries: list[Any] = []

    if usage is not None:
        entries.append(SimpleNamespace(type="message", message=SimpleNamespace(role="assistant", usage=usage)))
    if branch_usage is not None:
        entries.append(SimpleNamespace(type="branch_summary", usage=branch_usage))
    if compaction_usage is not None:
        entries.append(SimpleNamespace(type="compaction", usage=compaction_usage))
    if tool_usage is not None:
        entries.append(SimpleNamespace(type="message", message=SimpleNamespace(role="toolResult", usage=tool_usage)))

    return SimpleNamespace(
        state=SimpleNamespace(
            model=SimpleNamespace(
                id=model_id,
                provider=provider,
                context_window=200_000,
                reasoning=reasoning,
            ),
            thinking_level=thinking_level,
        ),
        session_manager=SimpleNamespace(
            get_entries=lambda: entries,
            get_session_name=lambda: session_name,
            get_cwd=lambda: "/tmp/project",
        ),
        get_context_usage=lambda: SimpleNamespace(context_window=200_000, percent=12.3),
        model_runtime=SimpleNamespace(is_using_subscription=lambda _provider: using_subscription),
    )


def _create_footer_data(provider_count: int) -> Any:
    return SimpleNamespace(
        get_git_branch=lambda: "main",
        get_extension_statuses=dict,
        get_available_provider_count=lambda: provider_count,
        on_branch_change=lambda callback: lambda: None,
    )


@pytest.fixture(autouse=True)
def _theme() -> None:
    # TypeScript passes `enableWatcher: false`; this port has no theme file
    # watcher (see `theme.py`'s module docstring), so there is nothing to disable.
    init_theme()


def test_does_not_abbreviate_sibling_paths_that_share_the_home_prefix() -> None:
    assert format_cwd_for_footer("/home/user2", "/home/user") == "/home/user2"


def test_abbreviates_the_home_directory_and_descendants() -> None:
    assert format_cwd_for_footer("/home/user", "/home/user") == "~"
    assert format_cwd_for_footer("/home/user/project", "/home/user") == "~/project"


def test_keeps_all_lines_within_width_for_wide_session_names() -> None:
    width = 93
    session = _create_session(session_name="한글" * 30)
    footer = FooterComponent(session, _create_footer_data(1))

    for line in footer.render(width):
        assert visible_width(line) <= width


def test_keeps_stats_line_within_width_for_wide_model_and_provider_names() -> None:
    width = 60
    session = _create_session(
        session_name="",
        model_id="模" * 30,
        provider="공급자",
        reasoning=True,
        thinking_level="high",
        usage=_usage(12_345, 6_789, 0, 0, 1.234),
    )
    footer = FooterComponent(session, _create_footer_data(2))

    for line in footer.render(width):
        assert visible_width(line) <= width


def test_includes_summary_and_tool_result_usage_in_the_total_cost() -> None:
    session = _create_session(
        session_name="",
        usage=_usage(100, 10, 0, 0, 0.5),
        branch_usage=_usage(20, 5, 0, 0, 0.25),
        compaction_usage=_usage(5, 2, 0, 0, 0.125),
        tool_usage=_usage(15, 3, 0, 0, 0.375),
    )
    footer = FooterComponent(session, _create_footer_data(1))

    stats_line = strip_ansi(footer.render(120)[1])
    assert "$1.250" in stats_line


def test_shows_the_latest_cache_hit_rate_when_cache_usage_is_present() -> None:
    session = _create_session(session_name="", usage=_usage(100, 10, 50, 50, 0.001))
    footer = FooterComponent(session, _create_footer_data(1))

    stats_line = strip_ansi(footer.render(120)[1])
    assert "CH25.0%" in stats_line


def test_marks_kimi_coding_costs_as_subscription_estimates() -> None:
    session = _create_session(session_name="", provider="kimi-coding", usage=_usage(100, 10, 0, 0, 1.234))
    footer = FooterComponent(session, _create_footer_data(1))

    assert "$1.234 (sub)" in strip_ansi(footer.render(120)[1])


def test_marks_explicitly_identified_subscription_auth() -> None:
    session = _create_session(session_name="", provider="anthropic", using_subscription=True)
    footer = FooterComponent(session, _create_footer_data(1))

    assert "$0.000 (sub)" in strip_ansi(footer.render(120)[1])


def test_does_not_mark_generic_oauth_sign_in_as_a_subscription() -> None:
    session = _create_session(session_name="", provider="openrouter", usage=_usage(100, 10, 0, 0, 1.234))
    footer = FooterComponent(session, _create_footer_data(1))
    stats = strip_ansi(footer.render(120)[1])

    assert "$1.234" in stats
    assert "(sub)" not in stats
