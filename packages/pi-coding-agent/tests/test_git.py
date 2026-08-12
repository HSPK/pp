"""Tests for pi_coding_agent.utils.git.

Ported from packages/coding-agent/test/git-ssh-url.test.ts. No `hosted-git-
info`-specific provider-alias shorthand cases are ported (e.g. bare
`github:user/repo`, unqualified `user/repo` defaulting to github.com) -- see
git.py's module docstring for why that npm dependency was dropped: every
case in the upstream test suite is satisfied by the generic host/path URL
parser alone.
"""

import pytest
from pi_coding_agent.utils.git import parse_git_url


def test_parse_https_url():
    result = parse_git_url("https://github.com/user/repo")
    assert result is not None
    assert result.host == "github.com"
    assert result.path == "user/repo"
    assert result.repo == "https://github.com/user/repo"


def test_parse_ssh_protocol_url():
    result = parse_git_url("ssh://git@github.com/user/repo")
    assert result is not None
    assert result.host == "github.com"
    assert result.path == "user/repo"
    assert result.repo == "ssh://git@github.com/user/repo"


def test_parse_protocol_url_with_ref():
    result = parse_git_url("https://github.com/user/repo@v1.0.0")
    assert result is not None
    assert result.host == "github.com"
    assert result.path == "user/repo"
    assert result.ref == "v1.0.0"
    assert result.repo == "https://github.com/user/repo"


def test_parse_git_at_host_colon_path_with_git_prefix():
    result = parse_git_url("git:git@github.com:user/repo")
    assert result is not None
    assert result.host == "github.com"
    assert result.path == "user/repo"
    assert result.repo == "git@github.com:user/repo"


def test_parse_host_path_shorthand_with_git_prefix():
    result = parse_git_url("git:github.com/user/repo")
    assert result is not None
    assert result.host == "github.com"
    assert result.path == "user/repo"
    assert result.repo == "https://github.com/user/repo"


def test_parse_shorthand_with_ref_and_git_prefix():
    result = parse_git_url("git:git@github.com:user/repo@v1.0.0")
    assert result is not None
    assert result.host == "github.com"
    assert result.path == "user/repo"
    assert result.ref == "v1.0.0"
    assert result.repo == "git@github.com:user/repo"


@pytest.mark.parametrize(
    "source",
    [
        "git:git@evil.example:../../victim/repo",
        "https://evil.example/..%2F..%2Fvictim/repo",
        "https://evil.example/..%2F..%2Fvictim/repo%",
        "git:git@evil.example:/absolute/repo",
        "git:git@evil.example:user\\repo/name",
        "git:git@evil.example:user/repo\0name",
    ],
)
def test_rejects_unsafe_git_install_path_inputs(source):
    assert parse_git_url(source) is None


def test_rejects_git_at_host_colon_path_without_git_prefix():
    assert parse_git_url("git@github.com:user/repo") is None


def test_rejects_host_path_shorthand_without_git_prefix():
    assert parse_git_url("github.com/user/repo") is None


def test_rejects_user_repo_shorthand():
    assert parse_git_url("user/repo") is None
