"""Python port of `packages/coding-agent/test/remote-catalog-provider.test.ts`.

`src/core/remote-catalog-provider.ts` is not ported. `withRemoteCatalog`
wraps a provider so it fetches a keyed model catalog from `pi.dev`, revalidates
it with `ETag`/`Last-Modified`, persists it through a `ModelsStore` and
publishes the merged overlay back into the registry. That whole layer -- the
dynamic model catalog/registry/store, named in the README as unported -- has no
counterpart here:

- `pi_ai.registry.Provider` has no `refresh_models`, and there is no
  `RefreshModelsContext` / `ModelsPublication` / `create_models(...).refresh()`.
- `pi_ai.models_store` does port `ModelsStore` / `ModelsStoreEntry` /
  `InMemoryModelsStore` (including the `etag`, `last_modified` and `checked_at`
  validator fields), so the *storage* half exists and is asserted below; the
  fetching/merging half does not.
- The version header these tests check (`User-Agent: pi/<VERSION>`) is a
  `pi.dev` catalog-request concern; this port does not query `pi.dev` at all
  (see the README's "Version checks use GitHub Releases, not pi.dev").

Every TypeScript case is recorded below with what it asserts.
"""

from __future__ import annotations

import pytest
from pi_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pi_ai.registry import Provider
from pi_ai.types import Model

_REASON = (
    "`src/core/remote-catalog-provider.ts` is not ported: it needs the dynamic "
    "model catalog/registry/store refresh layer (Provider.refreshModels, "
    "RefreshModelsContext, ModelsPublication), which this port omits -- see the "
    "README's unported list."
)


def test_provider_has_no_refresh_models_hook() -> None:
    """Pins the documented boundary the skips below rely on."""
    assert not hasattr(Provider, "refresh_models")


async def test_models_store_round_trips_the_validator_state_the_catalog_needs() -> None:
    """The storage half of the feature that *is* ported.

    `withRemoteCatalog` persists exactly these three fields between refreshes;
    the cases below all read them back through the store.
    """
    store = InMemoryModelsStore()
    model = Model(
        id="dynamic",
        name="dynamic",
        api="openai-completions",
        provider="test-provider",
        base_url="https://example.test/v1",
        reasoning=False,
        context_window=1000,
        max_tokens=100,
    )
    entry = ModelsStoreEntry(models=[model], etag='"catalog-1"', last_modified=1_700_000_000, checked_at=1_700_000_001)

    await store.write("test-provider", entry)
    stored = await store.read("test-provider")
    assert stored is not None
    assert [m.id for m in stored.models] == ["dynamic"]
    assert stored.etag == '"catalog-1"'
    assert stored.last_modified == 1_700_000_000
    assert stored.checked_at == 1_700_000_001

    # Reads are copies: mutating one must not corrupt the stored entry.
    stored.models.clear()
    reread = await store.read("test-provider")
    assert reread is not None
    assert [m.id for m in reread.models] == ["dynamic"]

    await store.delete("test-provider")
    assert await store.read("test-provider") is None


@pytest.mark.skip(reason=_REASON)
def test_parses_keyed_catalogs_sends_version_headers_observes_ttl_and_supports_force() -> None:
    """`it("parses keyed catalogs, sends version headers, observes the refresh TTL, and supports forced refreshes")`.

    Three refreshes (plain, plain, forced) against a catalog body keyed by model
    id: `getModels()` == `["static", "dynamic"]` (static base model first,
    remote overlay appended), the store holds only `["dynamic"]`, `fetch` ran
    exactly **twice** (the second plain refresh is inside the TTL and skipped;
    `force: true` bypasses it), and the first request's headers include a
    `User-Agent` containing `pi/<VERSION>`.
    """


@pytest.mark.skip(reason=_REASON)
def test_prefers_the_newer_of_the_generated_and_remote_catalogs() -> None:
    """`it("prefers the newer of the generated and remote catalogs")`.

    With `localGeneratedAt` set, a remote catalog whose `last-modified` is
    *older* is ignored (`getModels() == ["static"]`); a forced refresh with a
    `last-modified` 60s *newer* is applied (`["static", "newer"]`) and the store
    records `lastModified == Date.parse(newerHeader)`.
    """


@pytest.mark.skip(reason=_REASON)
def test_revalidates_a_stored_catalog_with_its_etag_and_keeps_the_overlay_on_304() -> None:
    """`it("revalidates a stored catalog with its etag and keeps the overlay on 304")`.

    First request sends **no** `if-none-match` and stores `etag: '"catalog-1"'`.
    The forced second request sends `if-none-match: '"catalog-1"'`; on a 304 the
    overlay survives (`["static", "dynamic"]`), the store still holds
    `["dynamic"]` with the same etag, and `checkedAt` moved forward (>= the
    previous value).
    """


@pytest.mark.skip(reason=_REASON)
def test_drops_a_stale_etag_when_the_overlay_becomes_unavailable() -> None:
    """`it("drops a stale etag when the overlay becomes unavailable")`.

    After a successful catalog fetch stores an etag, a forced refresh answered
    with `501` clears it: `store.read(...).etag` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_keeps_the_etag_and_overlay_after_a_transient_failure() -> None:
    """`it("keeps the etag and overlay after a transient failure")`.

    A forced refresh that exhausts its retries on `429` **rejects** with an
    error matching `/429/`, yet leaves the stored etag and models intact
    (unlike the `501` case above -- transient failures must not evict the
    cache). The next forced refresh re-sends `if-none-match: '"catalog-1"'` and
    restores `["static", "dynamic"]`. Note the retry count: the three `429`
    responses are consumed by one refresh call.
    """


@pytest.mark.skip(reason=_REASON)
def test_lets_a_newer_request_bypass_a_stalled_older_request_without_stale_publication() -> None:
    """`it("lets a newer catalog request bypass a stalled older request without stale publication")`.

    Two concurrent `models.refresh({force: true})` calls where the first fetch
    hangs: the second completes and publishes `["static", "newer"]`. When the
    stalled first request finally resolves with an older catalog, it must
    **not** overwrite it -- `getModels()` stays `["static", "newer"]` and the
    store still holds only `["newer"]`.
    """


@pytest.mark.skip(reason=_REASON)
def test_treats_unimplemented_pi_dev_catalog_routes_as_an_unavailable_overlay() -> None:
    """`it("treats unimplemented pi.dev catalog routes as an unavailable overlay")`.

    A `501` on the very first refresh **resolves** (does not throw),
    `getModels()` stays `["static"]`, and the store records an empty
    `models: []` with a numeric `checkedAt` so the TTL still applies.
    """
