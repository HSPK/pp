"""Python port of `packages/coding-agent/test/suite/regressions/7027-credential-refresh-hang.test.ts`.

Both TypeScript cases exercise the *network* model-catalog refresh, which this
port deliberately omits (see `core/model_runtime.py`, "No remote model catalog
refresh"). Each case is skipped individually below with the specific API it
needs and why that API is absent, rather than skipping the module wholesale.

Verified against `src/` while writing this file:

* `ModelRuntime.refresh` is `def refresh(self) -> None:` -- synchronous, no
  options, no `signal`, no `ModelsRefreshResult`. There is no in-flight
  refresh a login could queue behind.
* `register_native_provider` does not exist anywhere in `src/`.
* `Provider.refresh_models` does not exist; the only `refresh_models` is
  `modes/interactive/components/model_selector.py`, an unrelated UI helper.
* `complete_provider_authentication` does not exist; `_start_provider_login`
  persists the credential and updates the provider count with no bounded
  background refresh to warn about.
* `ModelRuntime.login(provider_id, api_key)` takes the key directly, rather
  than TypeScript's `login(id, "api_key", {prompt, notify})` interaction flow.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "Needs runtime.register_native_provider, Provider.refresh_models and "
        "refresh({allow_network, providers}); this port's refresh() is synchronous and local-only."
    )
)
def test_does_not_hold_login_behind_an_older_stalled_network_catalog_refresh() -> None:
    """TypeScript pins, against a provider whose `refreshModels` never settles when
    `allowNetwork` is true:

        await expect(runtime.login(provider.id, "api_key", {...}))
            .resolves.toEqual({ type: "api_key", key: "secret" });
        expect(runtime.getAvailableSnapshot().map((m) => m.id)).toContain("dynamic");
        expect(await credentials.read(provider.id)).toEqual({ type: "api_key", key: "secret" });
        await expect(stalledRefresh).resolves.toMatchObject({ aborted: false });

    Unportable: the stall is created by `runtime.refresh({allowNetwork: true,
    providers: [...]})` returning a pending promise. `refresh()` here returns
    `None` synchronously and never touches the network, so there is no pending
    refresh to stall, no per-provider operation queue for a login to jump, and
    no `ModelsRefreshResult` carrying `aborted`.
    """
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "Needs InteractiveMode.completeProviderAuthentication and an abortable, "
        "timeout-bounded background refresh; neither exists in this port."
    )
)
def test_completes_interactive_login_before_its_bounded_background_refresh() -> None:
    """TypeScript pins:

        expect(runtime.refresh).toHaveBeenCalledWith({
            providers: ["stalled-login"], signal: expect.any(AbortSignal),
        });
        expect(showWarning).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(15_000);
        expect(showWarning).toHaveBeenCalledWith(
            "Saved API key for Stalled Login, but its model catalog refresh timed out; using cached models.",
        );

    Unportable: `completeProviderAuthentication` has no counterpart, and the
    warning string it emits exists nowhere in `src/` because a synchronous
    local `refresh()` cannot time out. Asserting it would mean building the
    remote-catalog subsystem, not testing existing behavior.
    """
    raise AssertionError("unreachable")
