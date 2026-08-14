"""Guards that this package's test doubles have the same shape as production.

The TypeScript suite gets this for free: its terminal doubles are declared
`class TestTerminal implements Terminal` (see
`packages/tui/test/terminal-colors.test.ts`), so the compiler rejects a double
that is missing a member or declares a method with the wrong signature. Python
has no equivalent check at test time -- a `Protocol` is not enforced on
duck-typed doubles -- so a double that is narrower, or that implements an
`async def` collaborator as a plain `def`, would silently make tests pass where
production code would break.

These tests assert the conformance the TypeScript compiler would have.
"""

from __future__ import annotations

import inspect
from typing import Protocol, get_type_hints

from pi_tui.autocomplete import CombinedAutocompleteProvider
from pi_tui.components.editor import _AutocompleteProviderLike
from pi_tui.terminal import StdinSource, Terminal, TerminalIo
from pi_tui.testing import FakeStdinSource, FakeTerminal, FakeTerminalIo
from test_editor import FakeAutocompleteProvider, apply_completion_replace_prefix
from test_editor_autocomplete_edges import Provider as EdgeCaseProvider


def _protocol_members(protocol: type) -> list[str]:
    """Public members a `Protocol` declares, excluding Protocol machinery."""
    declared = set(vars(protocol)) | set(getattr(protocol, "__annotations__", {}))
    return sorted(name for name in declared if not name.startswith("_") and name not in {"mro", "register"})


def _is_property(owner: type, name: str) -> bool:
    return isinstance(inspect.getattr_static(owner, name, None), property)


class TestFakeTerminalMatchesTerminal:
    def test_implements_every_member_of_the_terminal_protocol(self) -> None:
        missing = [name for name in _protocol_members(Terminal) if not hasattr(FakeTerminal, name)]
        assert missing == []

    def test_properties_stay_properties(self) -> None:
        # `columns`/`rows`/`kitty_protocol_active` are read as attributes by
        # production code; a double exposing them as methods would make
        # `terminal.rows` a bound method and silently truthy.
        terminal = FakeTerminal(columns=42, rows=7)
        assert terminal.columns == 42
        assert terminal.rows == 7
        assert terminal.kitty_protocol_active is False

    def test_drain_input_is_a_coroutine_function_like_the_real_terminal(self) -> None:
        assert inspect.iscoroutinefunction(Terminal.drain_input)
        assert inspect.iscoroutinefunction(FakeTerminal.drain_input)

    def test_synchronous_members_stay_synchronous(self) -> None:
        for name in _protocol_members(Terminal):
            if name == "drain_input" or _is_property(Terminal, name):
                continue
            assert not inspect.iscoroutinefunction(getattr(FakeTerminal, name)), name

    def test_method_signatures_match_the_protocol(self) -> None:
        for name in _protocol_members(Terminal):
            if _is_property(Terminal, name):
                continue
            expected = inspect.signature(getattr(Terminal, name))
            actual = inspect.signature(getattr(FakeTerminal, name))
            assert list(actual.parameters) == list(expected.parameters), name


class TestFakeTerminalIoMatchesTerminalIo:
    def test_builds_a_terminal_io_with_every_field_populated(self) -> None:
        built = FakeTerminalIo().build()
        assert isinstance(built, TerminalIo)
        for field_name in get_type_hints(TerminalIo):
            assert getattr(built, field_name) is not None, field_name

    def test_fake_stdin_source_implements_the_stdin_source_protocol(self) -> None:
        missing = [name for name in _protocol_members(StdinSource) if not hasattr(FakeStdinSource, name)]
        assert missing == []
        for name in _protocol_members(StdinSource):
            assert not inspect.iscoroutinefunction(getattr(FakeStdinSource, name)), name


class TestFakeAutocompleteProviderMatchesProvider:
    def test_implements_every_member_of_the_provider_protocol(self) -> None:
        missing = [
            name for name in _protocol_members(_AutocompleteProviderLike) if not hasattr(FakeAutocompleteProvider, name)
        ]
        assert missing == []

    def test_get_suggestions_is_a_coroutine_function(self) -> None:
        # The editor awaits `provider.get_suggestions(...)`. A double defining
        # it with plain `def` would return a value the editor cannot await --
        # or, worse, hide an implementation that forgot to await.
        assert inspect.iscoroutinefunction(_AutocompleteProviderLike.get_suggestions)
        assert inspect.iscoroutinefunction(FakeAutocompleteProvider.get_suggestions)

    def test_apply_completion_stays_synchronous(self) -> None:
        assert not inspect.iscoroutinefunction(_AutocompleteProviderLike.apply_completion)
        assert not inspect.iscoroutinefunction(FakeAutocompleteProvider.apply_completion)

    def test_get_suggestions_accepts_the_keyword_arguments_the_editor_passes(self) -> None:
        parameters = inspect.signature(FakeAutocompleteProvider.get_suggestions).parameters
        assert list(parameters) == ["self", "lines", "cursor_line", "cursor_col", "signal", "force"]
        assert parameters["signal"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["force"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_apply_completion_helper_matches_the_protocol_signature(self) -> None:
        expected = list(inspect.signature(_AutocompleteProviderLike.apply_completion).parameters)
        expected.remove("self")
        assert list(inspect.signature(apply_completion_replace_prefix).parameters) == expected

    def test_the_autocomplete_edge_case_double_has_the_same_shape(self) -> None:
        missing = [name for name in _protocol_members(_AutocompleteProviderLike) if not hasattr(EdgeCaseProvider, name)]
        assert missing == []
        assert inspect.iscoroutinefunction(EdgeCaseProvider.get_suggestions)
        assert not inspect.iscoroutinefunction(EdgeCaseProvider.apply_completion)
        assert list(inspect.signature(EdgeCaseProvider.get_suggestions).parameters) == [
            "self",
            "lines",
            "cursor_line",
            "cursor_col",
            "signal",
            "force",
        ]

    def test_the_real_combined_provider_still_matches_the_protocol(self) -> None:
        # Rule of thumb: prefer the real collaborator. `CombinedAutocompleteProvider`
        # is offline, so pin its shape too -- if production drifts, the doubles
        # above are no longer standing in for anything real.
        missing = [
            name
            for name in _protocol_members(_AutocompleteProviderLike)
            if not hasattr(CombinedAutocompleteProvider, name)
        ]
        assert missing == []
        assert inspect.iscoroutinefunction(CombinedAutocompleteProvider.get_suggestions)
        assert not inspect.iscoroutinefunction(CombinedAutocompleteProvider.apply_completion)


class TestProtocolHelperItself:
    def test_lists_members_of_a_protocol(self) -> None:
        class Sample(Protocol):
            def visible(self) -> None: ...

            def _hidden(self) -> None: ...

        assert _protocol_members(Sample) == ["visible"]
