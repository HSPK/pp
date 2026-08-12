"""Python port of `packages/coding-agent/test/suite/regressions/7301-stalled-availability-refresh.test.ts`.

TypeScript's `ModelRuntime` keeps a mutable availability *snapshot*
(`snapshot.available` / `configuredProviders` / `storedProviders` / `auth`)
plus an `availabilityError`, both written asynchronously by
`refresh({allowNetwork})` / `getAvailable()`. Because two refreshes can
overlap, TS guards every write with a monotonically increasing
`availabilityErrorSeq` so a slow in-flight refresh cannot overwrite a newer
snapshot or a newer (cleared) error. All three cases stall the credential read,
start a second refresh, and assert the stale one is ignored.

This port has no cached availability state: `get_available()` delegates
straight to `Models.get_available()` and stores nothing but the failure of the
newest pass, and `get_provider_auth_status()` / `has_configured_auth()` read
the credential store synchronously on every call. `refresh()` is synchronous,
takes no `allow_network`, and only reloads `models.json`, so there is no
snapshot to go stale -- but the availability *error* can, so
`ModelRuntime.get_available` carries the same monotonic sequence guard as TS.

That makes the *mechanism* unportable, but the *outcome* each case asserts is
not: a stalled or failing availability read must still leave auth status and
`get_error()` reporting the newest state. That is what the ported cases pin
below, by stalling `ModelRuntime.credentials.get` (this port's equivalent of
TypeScript's stalled `authStorage.list`, since `Models.get_available` reads
credentials through `get`; note the runtime wraps `AuthStorage` in a
`RuntimeCredentials` overlay, so the overlay is the object that must be
stalled -- patching the underlying `AuthStorage` intercepts nothing). The one assertion that cannot survive the translation --
`refresh({allowNetwork: false})` being awaited to observe a *second* credential
list -- is called out at the exact spot it would go.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harness import create_harness
from pi_ai.auth.types import Credential
from pi_ai.models import ModelsError


class _StalledCredentialGet:
    """Stalls the first credential-store `get` call on an event this test controls.

    Port of TS `stallNextCredentialList`. Shaped like the real coroutine method
    it replaces (`async def get(provider_id) -> Credential | None`) so it
    cannot be satisfied more easily than `RuntimeCredentials`; the release/fail gate is
    an `asyncio.Event` rather than a wall-clock wait, so no test here sleeps.
    """

    def __init__(self, storage: object) -> None:
        self._storage = storage
        self._original = storage.get
        self.started = asyncio.Event()
        self._gate = asyncio.Event()
        self._error: Exception | None = None
        self._should_stall = True
        self.call_count = 0

        async def stalling_get(provider_id: str) -> Credential | None:
            entry = await self._original(provider_id)
            self.call_count += 1
            if not self._should_stall:
                return entry
            self._should_stall = False
            self.started.set()
            await self._gate.wait()
            if self._error is not None:
                raise self._error
            return entry

        storage.get = stalling_get  # type: ignore[method-assign]

    def release(self) -> None:
        self._gate.set()

    def fail(self, error: Exception) -> None:
        self._error = error
        self._gate.set()

    def restore(self) -> None:
        self._storage.get = self._original  # type: ignore[method-assign]


async def test_recovers_without_letting_the_stalled_refresh_overwrite_the_newer_snapshot(
    tmp_path: Path,
) -> None:
    # TS uses `withConfiguredAuth: false` and stalls `authStorage.list()`, which
    # its refresh calls unconditionally. This port stalls `AuthStorage.get`,
    # which `Models.get_available` only calls per registered provider, so the
    # faux provider has to be present for the stall to have anything to stall.
    harness = await create_harness(tmp_path)
    runtime = harness.model_runtime

    await harness.auth_storage.set("stale-provider", Credential(type="api_key", key="stale-key"))
    runtime.refresh()
    status = runtime.get_provider_auth_status("stale-provider")
    assert status.configured is True
    # TypeScript reports `source: "stored"`; this port's `AuthCheck` spells the
    # same state "stored credential" (`ModelRuntime.get_provider_auth_status`).
    assert status.source == "stored credential"

    stalled = _StalledCredentialGet(runtime.credentials)
    try:
        stale_refresh = asyncio.ensure_future(runtime.get_available())
        await stalled.started.wait()

        await harness.auth_storage.delete("stale-provider")
        await harness.auth_storage.set("current-provider", Credential(type="api_key", key="current-key"))

        # TS waits here for `refresh({allowNetwork: false})` to drive a *second*
        # credential list before asserting. Skipped: this port's `refresh()`
        # takes no options and never lists credentials, and auth status is read
        # from the store on every call, so the newest state is observable with
        # the stale read still in flight -- which is the stronger assertion.
        assert runtime.get_provider_auth_status("stale-provider").configured is False
        current = runtime.get_provider_auth_status("current-provider")
        assert current.configured is True
        assert current.source == "stored credential"

        stalled.release()
        await stale_refresh

        assert runtime.get_provider_auth_status("stale-provider").configured is False
        current = runtime.get_provider_auth_status("current-provider")
        assert current.configured is True
        assert current.source == "stored credential"
    finally:
        stalled.release()
        stalled.restore()


async def test_does_not_let_a_stale_failure_overwrite_newer_availability_error_state(
    tmp_path: Path,
) -> None:
    harness = await create_harness(tmp_path)
    runtime = harness.model_runtime
    assert runtime.get_error() is None

    stalled = _StalledCredentialGet(runtime.credentials)
    try:
        stale_refresh = asyncio.ensure_future(runtime.get_available())
        await stalled.started.wait()
        assert runtime.get_error() is None

        # TS drives its recovery pass with `refresh({allowNetwork: false})` and
        # waits for a *second* credential list. This port's `refresh()` takes no
        # options and never lists credentials, so the newer availability pass is
        # a second `get_available()` -- the same thing TS's recovery refresh
        # ultimately performs, and what bumps the sequence counter that makes
        # the stale failure below stale.
        await runtime.get_available()
        assert runtime.get_error() is None

        stalled.fail(RuntimeError("stale credential list failure"))
        # TS's stale pass is `refresh({allowNetwork: false})`, which fails while
        # *listing* credentials, so its rejection carries the raw message. This
        # port substitutes `get_available()` (see the note above), which reads a
        # credential and therefore goes through `Models._read_credential` --
        # the port of `models.ts`'s `readCredential` -- so the raw error arrives
        # wrapped in `ModelsError`. The substituted path, not the assertion, is
        # what changes the exception type; the original message is preserved.
        with pytest.raises(ModelsError, match="stale credential list failure") as excinfo:
            await stale_refresh
        # TS's `{ cause: error }` guarantees the original is preserved, not just
        # quoted into the message. Matching the substring alone would still pass
        # if the wrapper dropped the cause and only interpolated its text.
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert str(excinfo.value.__cause__) == "stale credential list failure"

        # The superseded pass must not overwrite the newer, error-free state.
        assert runtime.get_error() is None
    finally:
        stalled.release()
        stalled.restore()


async def test_does_not_let_a_stale_provider_scoped_failure_overwrite_a_newer_availability_pass(
    tmp_path: Path,
) -> None:
    harness = await create_harness(tmp_path)
    runtime = harness.model_runtime
    model = harness.get_model()
    assert model is not None

    models = runtime.models
    original_get_available = models.get_available
    started = asyncio.Event()
    gate = asyncio.Event()
    stall = True

    async def get_available(provider_id: str | None = None) -> list:
        nonlocal stall
        if not provider_id or not stall:
            return await original_get_available(provider_id)
        stall = False
        started.set()
        await gate.wait()
        raise RuntimeError("stale provider availability failure")

    models.get_available = get_available  # type: ignore[method-assign]
    try:
        stale_refresh = asyncio.ensure_future(runtime.get_available(model.provider))
        await started.wait()

        available = await runtime.get_available()
        assert model.id in [entry.id for entry in available]
        assert runtime.get_error() is None

        gate.set()
        with pytest.raises(RuntimeError, match="stale provider availability failure"):
            await stale_refresh
        assert runtime.get_error() is None
    finally:
        gate.set()
        models.get_available = original_get_available  # type: ignore[method-assign]
