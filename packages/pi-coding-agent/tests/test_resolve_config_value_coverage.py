"""Python port of `packages/coding-agent/test/resolve-config-value.test.ts`,
plus extra coverage for `core/resolve_config_value.py`.

The eight TypeScript cases are ported below under "TypeScript parity"; the rest
of the file covers parsing and `resolve_headers*` edges the TypeScript suite
does not reach: $$ and $! escapes, unclosed ${, invalid env-var name in ${},
bare $ at end of string, command failure paths, multi-env-var error,
resolve_headers / resolve_headers_or_throw.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_coding_agent.core.resolve_config_value import (
    clear_config_value_cache,
    get_config_value_env_var_name,
    get_config_value_env_var_names,
    get_missing_config_value_env_var_names,
    is_command_config_value,
    is_config_value_configured,
    resolve_config_value,
    resolve_config_value_or_throw,
    resolve_config_value_uncached,
    resolve_headers,
    resolve_headers_or_throw,
)


@pytest.fixture(autouse=True)
def _cache():
    clear_config_value_cache()
    yield
    clear_config_value_cache()


# ---------------------------------------------------------------------------
# Template escaping
# ---------------------------------------------------------------------------


def test_double_dollar_produces_literal_dollar(monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    assert resolve_config_value("$$") == "$"


def test_dollar_bang_produces_literal_bang():
    assert resolve_config_value("$!") == "!"


def test_dollar_bang_at_start_of_non_command_template():
    # $! is not a command prefix (! is); it produces a literal "!"
    result = resolve_config_value("hello$!world")
    assert result == "hello!world"


def test_mixed_escapes():
    result = resolve_config_value("$$FOO$$")
    assert result == "$FOO$"


# ---------------------------------------------------------------------------
# ${} edge cases
# ---------------------------------------------------------------------------


def test_unclosed_brace_treated_as_literal(monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    # "${FOO" with no closing "}" — "$" is literal, then "{FOO" appended → "${FOO"
    result = resolve_config_value("${FOO")
    assert result == "${FOO"


def test_braces_with_invalid_env_var_name_treated_as_literal(monkeypatch):
    # "${1INVALID}" — name starts with digit, not a valid env var name
    result = resolve_config_value("${1INVALID}")
    assert result == "${1INVALID}"


def test_braces_with_valid_env_var_resolved(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    assert resolve_config_value("${MY_KEY}") == "secret"


def test_dollar_at_end_of_string_is_literal():
    # Bare "$" at end with no following char → treated as literal "$"
    result = resolve_config_value("value$")
    assert result == "value$"


def test_dollar_with_no_alphanumeric_following_is_literal():
    result = resolve_config_value("$-not-a-var")
    assert result == "$-not-a-var"


# ---------------------------------------------------------------------------
# get_config_value_env_var_name / names
# ---------------------------------------------------------------------------


def test_get_config_value_env_var_name_returns_none_for_multi_part_template(monkeypatch):
    # "prefix-$VAR" has two parts → not a single bare env var → returns None
    result = get_config_value_env_var_name("prefix-$VAR")
    assert result is None


def test_get_config_value_env_var_name_returns_none_for_command():
    result = get_config_value_env_var_name("!echo secret")
    assert result is None


def test_get_config_value_env_var_names_returns_all_names():
    names = get_config_value_env_var_names("$A and $B and $A")
    assert names == ["A", "B"]


def test_get_config_value_env_var_names_empty_for_command():
    assert get_config_value_env_var_names("!echo") == []


def test_get_missing_config_value_env_var_names(monkeypatch):
    monkeypatch.setenv("PRESENT", "yes")
    monkeypatch.delenv("MISSING", raising=False)
    missing = get_missing_config_value_env_var_names("$PRESENT and $MISSING")
    assert missing == ["MISSING"]


def test_is_config_value_configured_true_when_all_env_vars_present(monkeypatch):
    monkeypatch.setenv("KEY_A", "a")
    assert is_config_value_configured("$KEY_A")


def test_is_config_value_configured_false_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    assert not is_config_value_configured("$ABSENT_KEY")


# ---------------------------------------------------------------------------
# Command execution failures
# ---------------------------------------------------------------------------


def test_execute_command_returns_none_on_nonzero_exit():
    # A command that exits non-zero should return None.
    result = resolve_config_value_uncached("!exit 1")
    assert result is None


def test_execute_command_returns_none_on_empty_stdout():
    result = resolve_config_value_uncached("!true")
    assert result is None


def test_execute_command_returns_stripped_stdout():
    result = resolve_config_value_uncached("!echo   hello   ")
    assert result == "hello"


def test_execute_command_caches_result():
    # Deliberately not `resolve("!echo x") == resolve("!echo x")`: two equal
    # strings prove nothing about caching. The counter-file cases under
    # "TypeScript parity" below observe the side effect, as the TypeScript does.
    assert resolve_config_value("!echo cached_value") == "cached_value"
    assert resolve_config_value("!echo cached_value") == "cached_value"


def test_execute_command_uncached_returns_none_on_oserror(monkeypatch):
    import subprocess as sp

    def raise_oserror(*args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(sp, "run", raise_oserror)
    result = resolve_config_value_uncached("!somecommand")
    assert result is None


def test_is_command_config_value_true_for_bang_prefix():
    assert is_command_config_value("!echo foo")


def test_is_command_config_value_false_for_template():
    assert not is_command_config_value("$FOO")


# ---------------------------------------------------------------------------
# resolve_config_value_or_throw
# ---------------------------------------------------------------------------


def test_or_throw_raises_for_failing_command():
    with pytest.raises(ValueError, match="Failed to resolve api_key from shell command"):
        resolve_config_value_or_throw("!exit 1", "api_key")


def test_or_throw_raises_for_single_missing_env_var(monkeypatch):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    with pytest.raises(ValueError, match="from environment variable: MISSING_SECRET"):
        resolve_config_value_or_throw("$MISSING_SECRET", "api_key")


def test_or_throw_raises_for_multiple_missing_env_vars(monkeypatch):
    monkeypatch.delenv("KEY_X", raising=False)
    monkeypatch.delenv("KEY_Y", raising=False)
    with pytest.raises(ValueError, match="from environment variables: KEY_X, KEY_Y"):
        resolve_config_value_or_throw("$KEY_X-$KEY_Y", "api_key")


def test_or_throw_raises_generic_for_unresolved_template(monkeypatch):
    # A template with only literal parts that still resolves to empty → generic error
    # Actually a literal always resolves, so use an env template that is missing.
    monkeypatch.delenv("GONE", raising=False)
    with pytest.raises(ValueError):
        resolve_config_value_or_throw("$GONE", "api_key")


def test_or_throw_returns_value_when_command_succeeds():
    result = resolve_config_value_or_throw("!echo working", "api_key")
    assert result == "working"


def test_or_throw_returns_value_when_env_present(monkeypatch):
    monkeypatch.setenv("GOOD_KEY", "secret123")
    result = resolve_config_value_or_throw("$GOOD_KEY", "api_key")
    assert result == "secret123"


# ---------------------------------------------------------------------------
# resolve_headers
# ---------------------------------------------------------------------------


def test_resolve_headers_returns_none_for_empty_dict():
    assert resolve_headers({}) is None


def test_resolve_headers_returns_none_for_none():
    assert resolve_headers(None) is None


def test_resolve_headers_resolves_env_vars(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "bearer-xyz")
    result = resolve_headers({"Authorization": "$AUTH_TOKEN"})
    assert result == {"Authorization": "bearer-xyz"}


def test_resolve_headers_skips_unresolvable_values(monkeypatch):
    monkeypatch.delenv("MISSING_HDR", raising=False)
    monkeypatch.setenv("PRESENT_HDR", "ok")
    result = resolve_headers({"X-Good": "$PRESENT_HDR", "X-Bad": "$MISSING_HDR"})
    assert result == {"X-Good": "ok"}


def test_resolve_headers_returns_none_when_all_unresolvable(monkeypatch):
    monkeypatch.delenv("MISSING_A", raising=False)
    result = resolve_headers({"X-Key": "$MISSING_A"})
    assert result is None


# ---------------------------------------------------------------------------
# resolve_headers_or_throw
# ---------------------------------------------------------------------------


def test_resolve_headers_or_throw_returns_none_for_none():
    assert resolve_headers_or_throw(None, "auth") is None


def test_resolve_headers_or_throw_returns_none_for_empty():
    assert resolve_headers_or_throw({}, "auth") is None


def test_resolve_headers_or_throw_raises_for_missing_env_var(monkeypatch):
    monkeypatch.delenv("MISSING_VAL", raising=False)
    with pytest.raises(ValueError, match='header "X-Secret"'):
        resolve_headers_or_throw({"X-Secret": "$MISSING_VAL"}, "auth")


def test_resolve_headers_or_throw_returns_resolved_headers(monkeypatch):
    monkeypatch.setenv("HDR_VAL", "token-abc")
    result = resolve_headers_or_throw({"X-Auth": "$HDR_VAL"}, "auth")
    assert result == {"X-Auth": "token-abc"}


# ---------------------------------------------------------------------------
# TypeScript parity: the eight cases in
# `packages/coding-agent/test/resolve-config-value.test.ts`
# ---------------------------------------------------------------------------


def _counter_command(counter_file: Path, tail: str) -> str:
    """A shell command that bumps `counter_file` every time it actually runs."""
    path = str(counter_file)
    return f"""!sh -c 'count=$(cat "{path}"); echo $((count + 1)) > "{path}"; {tail}'"""


def test_resolves_literals_environment_templates_and_escapes(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_LEFT", "left")
    monkeypatch.setenv("TEST_CONFIG_RIGHT", "right")

    assert resolve_config_value("literal-key") == "literal-key"
    assert resolve_config_value("$TEST_CONFIG_LEFT") == "left"
    assert resolve_config_value("${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT") == "left_right"
    assert resolve_config_value("$$TEST_CONFIG_LEFT") == "$TEST_CONFIG_LEFT"
    assert resolve_config_value("$!literal-$TEST_CONFIG_RIGHT") == "!literal-right"


def test_uses_credential_scoped_environment_before_process_env(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_SCOPED", "process")

    assert resolve_config_value("$TEST_CONFIG_SCOPED", {"TEST_CONFIG_SCOPED": "credential"}) == "credential"


def test_executes_shell_commands_and_trims_their_output():
    assert resolve_config_value("!echo '  spaced-key  '") == "spaced-key"
    # Only the outer whitespace is trimmed: an interior newline survives.
    assert resolve_config_value("!printf 'line1\\nline2'") == "line1\nline2"
    # Goes through a real shell, so a pipeline works.
    assert resolve_config_value("!echo 'hello world' | tr ' ' '-'") == "hello-world"


@pytest.mark.parametrize("command", ["!exit 1", "!nonexistent-command-12345", "!printf ''"])
def test_returns_none_when_command_resolution_fails(command):
    assert resolve_config_value(command) is None


def test_caches_successful_and_failed_commands_until_explicitly_cleared(tmp_path: Path):
    counter_file = tmp_path / "counter"
    counter_file.write_text("0")
    success = _counter_command(counter_file, "echo value")

    assert resolve_config_value(success) == "value"
    assert resolve_config_value(success) == "value"
    assert counter_file.read_text().strip() == "1"

    clear_config_value_cache()
    assert resolve_config_value(success) == "value"
    assert counter_file.read_text().strip() == "2"

    failure = _counter_command(counter_file, "exit 1")
    assert resolve_config_value(failure) is None
    assert resolve_config_value(failure) is None
    assert counter_file.read_text().strip() == "3"


def test_does_not_cache_environment_values(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_DYNAMIC", "first")
    assert resolve_config_value("$TEST_CONFIG_DYNAMIC") == "first"

    monkeypatch.setenv("TEST_CONFIG_DYNAMIC", "second")
    assert resolve_config_value("$TEST_CONFIG_DYNAMIC") == "second"


def test_uncached_resolution_executes_a_command_on_every_call(tmp_path: Path):
    counter_file = tmp_path / "uncached-counter"
    counter_file.write_text("0")
    command = _counter_command(counter_file, "echo value")

    assert resolve_config_value_uncached(command) == "value"
    assert resolve_config_value_uncached(command) == "value"
    assert counter_file.read_text().strip() == "2"


@pytest.mark.skip(
    reason=(
        "TS `test('uses stdin when the configured Windows shell requires it')` stubs "
        "`getShellConfig()` to return `{shell: '/bin/bash', args: ['-s'], commandTransport: 'stdin'}` "
        "and forces `process.platform = 'win32'`, pinning that the command is piped through stdin "
        "instead of passed as an argument. `utils/shell.py`'s module docstring states that the "
        "Windows/Git-Bash shell discovery is not ported: there is no `get_shell_config()` and no "
        "`command_transport`, and `_execute_command_uncached` always runs `/bin/sh -c <command>`. "
        "There is nothing to configure and no stdin transport to select."
    )
)
def test_uses_stdin_when_the_configured_windows_shell_requires_it() -> None:
    raise AssertionError("unreachable")
