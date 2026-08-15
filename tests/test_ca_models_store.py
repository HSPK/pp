"""Python port of `packages/coding-agent/test/models-store.test.ts`.

The whole TypeScript file exercises `FileModelsStore` from
`src/core/models-store.ts`, the on-disk, `proper-lockfile`-guarded,
revision-tracked model-catalog cache. This port does not have it.
`core/model_runtime.py`'s module docstring records the omission by name
("No locked, revision-tracked `ModelsStore`"), and the boundary test at the
bottom of this file pins that fact so the four skips below cannot silently
outlive the reason for them.

Every one of the four TypeScript cases asserts on file I/O, cross-process
advisory locking, or the in-flight-read coalescing that only the file-backed
store has, so each is skipped individually at the case it corresponds to
rather than rewritten against a store that has none of that machinery. The
shared `ModelsStore` contract that *is* ported --
`pi_ai.models_store.InMemoryModelsStore`'s read/write/delete, its
deep-copy-on-read/write, and its abort-signal handling -- is pinned for real
in `packages/pi-ai/tests/test_models_store.py`, which is where the ported code
lives.
"""

from __future__ import annotations

import pytest
from pi_ai import models_store

FILE_MODELS_STORE_NOT_PORTED = (
    "src/core/models-store.ts's FileModelsStore is not ported: this port has no "
    "on-disk model-catalog cache, no file-revision tracking and no "
    "proper-lockfile equivalent (see core/model_runtime.py's module docstring). "
    "Only pi_ai.models_store.InMemoryModelsStore exists."
)


@pytest.mark.skip(reason=FILE_MODELS_STORE_NOT_PORTED)
def test_persists_provider_catalogs_without_replacing_unrelated_providers() -> None:
    """`it("persists provider catalogs without replacing unrelated providers")`.

    Writes provider `one` then `two` through one `FileModelsStore`, builds a
    *second* `FileModelsStore` over the same path, and asserts the reload sees
    `read("one").models.map(id) === ["m1"]`, `read("one").checkedAt === 100`
    and `read("two").models.map(id) === ["m2"]`; then `delete("one")` leaves
    `read("one")` undefined while `read("two")` still resolves `["m2"]`.

    The assertion is specifically that a *separate instance over the same file*
    sees the earlier writes, so it cannot be expressed without file backing.
    """
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=FILE_MODELS_STORE_NOT_PORTED)
def test_coalesces_file_reloads_across_concurrent_readers_and_interleaved_instances() -> None:
    """`it("coalesces file reloads across concurrent readers and interleaved storage instances")`.

    Spies on `lockfile.lock` and asserts the exact call counts as reads are
    interleaved: three concurrent reads across two instances take the lock
    `toHaveBeenCalledTimes(1)`; a follow-up `second.read("one")` still `1`; two
    reads through a third instance on a *different* path plus a reload of the
    shared path bring it to `3`. Every assertion counts `proper-lockfile`
    acquisitions, which have no counterpart here.
    """
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=FILE_MODELS_STORE_NOT_PORTED)
def test_keeps_a_coalesced_reload_alive_while_another_reader_is_still_waiting() -> None:
    """`it("keeps a coalesced reload alive while another reader is still waiting")`.

    Two readers join one in-flight reload; aborting the first must reject it
    with `{name: "AbortError"}` without cancelling the shared reload, so the
    second still resolves `{models: [{id: "stored"}]}`, with
    `lockSpy` called once and `release` called once. This is the reload
    refcounting inside `FileModelsStore`.
    """
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=FILE_MODELS_STORE_NOT_PORTED)
def test_cancels_a_catalog_write_waiting_for_a_held_file_lock_without_writing_later() -> None:
    """`it("cancels a catalog write waiting for a held file lock without writing later")`.

    Holds a real `proper-lockfile` lock on the store file, starts a `write`,
    aborts it, releases the lock, and asserts the write never lands: the
    on-disk JSON still has `one` and never gains `two`. Cross-process advisory
    locking has no counterpart in this port.
    """
    raise AssertionError("unreachable")


def test_port_has_no_file_backed_models_store() -> None:
    """Pins the boundary the four skips above depend on.

    If a `FileModelsStore` (or any other non-in-memory `ModelsStore`) is ever
    added, this fails and those four cases must be written for real.
    """
    assert not hasattr(models_store, "FileModelsStore")
    assert not hasattr(models_store, "InMemoryCodingAgentModelsStore")
    # `ModelsStore` itself is the Protocol every implementation satisfies.
    implementations = [
        name
        for name in dir(models_store)
        if name.endswith("ModelsStore") and name != "ModelsStore" and isinstance(getattr(models_store, name), type)
    ]
    assert implementations == ["InMemoryModelsStore"]
