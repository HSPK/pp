"""Python port of `packages/coding-agent/test/suite/regressions/7443-model-command-cached-match.test.ts`."""

from __future__ import annotations

from pathlib import Path

import pytest
from interactive_harness import make_interactive_mode


async def test_matches_the_availability_snapshot_without_starting_a_catalog_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    try:
        model_runtime = mode.session.model_runtime
        refreshes: list[None] = []

        def refresh() -> None:
            refreshes.append(None)

        monkeypatch.setattr(model_runtime, "refresh", refresh)
        statuses: list[str] = []
        monkeypatch.setattr(mode, "show_status", statuses.append)

        cached = model_runtime.get_available_snapshot()[0]
        model = mode._find_exact_model_match(cached.id)

        assert model is not None
        assert model.id == cached.id
        assert refreshes == []
        assert statuses == []
    finally:
        mode.session.dispose()


async def test_falls_back_to_the_selector_after_a_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Port of "uses a caller-owned deadline only after a cache miss".

    TypeScript's cache miss runs `modelRuntime.refresh({ signal })` behind a
    15-second `AbortController` deadline and shows "Refreshing model
    catalogs…". This port has no remote catalog refresh at all
    (`core/model_runtime.py`: "`refresh()` only rebuilds the *local*
    composition ... it never makes a network call"), so these three
    assertions are skipped:

        expect(refresh).toHaveBeenCalledOnce();
        expect(refresh.mock.calls[0]?.[0]?.signal).toBeInstanceOf(AbortSignal);
        expect(context.showStatus).toHaveBeenCalledWith("Refreshing model catalogs…");

    What stays assertable is the outcome the deadline exists to reach: an
    unknown reference resolves to no model, so `/model <term>` opens the
    selector pre-filled with the term instead of switching silently.
    """
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    try:
        assert mode._find_exact_model_match("not-cached") is None

        selector_searches: list[str | None] = []
        monkeypatch.setattr(mode, "show_model_selector", selector_searches.append)
        set_models: list[object] = []

        async def set_model(model: object) -> None:
            set_models.append(model)

        monkeypatch.setattr(mode.session, "set_model", set_model)

        await mode._handle_model_command("not-cached")

        assert selector_searches == ["not-cached"]
        assert set_models == []
    finally:
        mode.session.dispose()


async def test_model_command_switches_directly_on_a_cached_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The behaviour `findExactModelMatch` feeds: `handleModelCommand` sets the
    model and never opens the selector when the term matches exactly."""
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    try:
        selector_searches: list[str | None] = []
        monkeypatch.setattr(mode, "show_model_selector", selector_searches.append)
        set_models: list[object] = []

        async def set_model(model: object) -> None:
            set_models.append(model)

        monkeypatch.setattr(mode.session, "set_model", set_model)
        statuses: list[str] = []
        monkeypatch.setattr(mode, "show_status", statuses.append)

        cached = mode.session.model_runtime.get_available_snapshot()[0]
        await mode._handle_model_command(cached.id)

        assert selector_searches == []
        assert [getattr(model, "id", None) for model in set_models] == [cached.id]
        assert statuses == [f"Model: {cached.id}"]
    finally:
        mode.session.dispose()


async def test_model_command_without_an_argument_opens_the_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = await make_interactive_mode(tmp_path, monkeypatch)
    try:
        opened: list[object] = []
        monkeypatch.setattr(mode, "show_model_selector", lambda *args: opened.append(args))

        await mode._handle_model_command(None)

        assert opened == [()]
    finally:
        mode.session.dispose()
