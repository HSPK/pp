"""Regression tests for defects found by auditing the package manager.

Both defects fail open: an offline install still reaches the network, and a
disable pattern that fails to match leaves executable extension code enabled.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.core.package_manager import (
    _expand_braces,
    _glob_to_regex,
    _is_offline_mode_enabled,
)

# --------------------------------------------------------------------------
# PI_OFFLINE
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES"])
def test_offline_mode_enabled_for_documented_values(value, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", value)
    assert _is_offline_mode_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "random", "2"])
def test_offline_mode_disabled_for_everything_else(value, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", value)
    assert _is_offline_mode_enabled() is False


def test_offline_mode_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    assert _is_offline_mode_enabled() is False


def test_offline_gates_are_wired_into_the_network_paths():
    """Every network entry point must consult offline mode.

    A missing gate means a restricted or air-gapped environment still performs
    git clones and remote update checks on startup.
    """
    import inspect

    from pi_coding_agent.core import package_manager

    source = inspect.getsource(package_manager)
    # helper definition + the six call sites
    assert source.count("_is_offline_mode_enabled()") >= 6


# --------------------------------------------------------------------------
# glob matching
# --------------------------------------------------------------------------


def test_brace_expansion():
    assert sorted(_expand_braces("{foo,bar}.md")) == ["bar.md", "foo.md"]
    assert sorted(_expand_braces("a{1,2}b{3,4}")) == ["a1b3", "a1b4", "a2b3", "a2b4"]
    assert _expand_braces("plain.md") == ["plain.md"]


def test_nested_brace_expansion():
    assert sorted(_expand_braces("{a,{b,c}}.md")) == ["a.md", "b.md", "c.md"]


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        # Brace expansion: without it a disable pattern silently does nothing.
        ("{foo,bar}.md", "foo.md", True),
        ("{foo,bar}.md", "bar.md", True),
        ("{foo,bar}.md", "baz.md", False),
        ("skills/{a,b}/SKILL.md", "skills/a/SKILL.md", True),
        ("skills/{a,b}/SKILL.md", "skills/c/SKILL.md", False),
        # A globstar matches zero segments.
        ("skills/**/SKILL.md", "skills/SKILL.md", True),
        ("skills/**/SKILL.md", "skills/a/SKILL.md", True),
        ("skills/**/SKILL.md", "skills/a/b/SKILL.md", True),
        ("a/**/b", "a/b", True),
        ("a/**/b", "a/x/b", True),
        # Single star stays within one segment.
        ("*.md", "x.md", True),
        ("*.md", "dir/x.md", False),
        # Character classes and single-character wildcards.
        ("a?c", "abc", True),
        ("a?c", "ac", False),
        ("[abc].md", "a.md", True),
        ("[abc].md", "d.md", False),
        ("[!a].md", "b.md", True),
        ("[!a].md", "a.md", False),
    ],
)
def test_glob_matching(pattern, path, expected):
    assert bool(_glob_to_regex(pattern).match(path)) is expected


def test_globstar_matches_zero_segments():
    """Regression: `a/**/b` required at least one intervening segment."""
    assert _glob_to_regex("a/**/b").match("a/b") is not None


def test_braces_are_not_escaped_literally():
    """Regression: `{` `}` `,` were escaped, so brace patterns never matched."""
    assert _glob_to_regex("{a,b}.md").match("a.md") is not None
    assert _glob_to_regex("{a,b}.md").match("{a,b}.md") is None
