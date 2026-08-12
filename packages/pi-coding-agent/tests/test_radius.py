"""Python port of `packages/coding-agent/test/radius.test.ts`.

Most of the upstream file exercises the remote model-catalog layer that
`pi_coding_agent.core.model_runtime` deliberately omits (`ModelsStore`,
`allowModelNetwork`, `withRemoteCatalog`, and the network half of the Radius
gateway catalog fetch -- see the module docstring of `core/model_runtime.py`).

`ModelRuntime.create` *does* now port the "legacy credential catalog" restore
that TS `radius.ts`'s `refreshModels` performs before any network call: an
OAuth credential caches the gateway's last `/v1/config` response in
`credential.data["gatewayConfig"]` (Python) / `credential.gatewayConfig` (TS),
so a stored credential makes Radius models available immediately with no
network access at all. That path needed no `ModelsStore` -- it is a pure,
synchronous seed of the built-in `radius` provider from the stored credential
-- so it is ported and exercised below rather than skipped. The cases that
depend only on the ported composition layer are kept; the ones needing a live
(or mocked) network fetch and `ModelsStore` persistence are skipped with an
explicit reason at that exact spot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.provider_attribution import RADIUS_PROVIDER_ID


async def _runtime(tmp_path: Path, models_json: dict[str, object] | None = None) -> ModelRuntime:
    models_path = tmp_path / "models.json"
    if models_json is not None:
        models_path.write_text(json.dumps(models_json))
    return await ModelRuntime.create(
        agent_dir=str(tmp_path),
        auth_path=str(tmp_path / "auth.json"),
        models_path=str(models_path),
        env={},
    )


def _write_radius_credential(auth_path: Path, gateway_base_url: str) -> None:
    """Port of TS test helper `radiusOAuthCredential`, written straight to `auth.json`.

    `FileCredentialStore` (de)serializes provider-specific extras under a
    `"data"` sub-key rather than flattening them onto the credential the way
    TypeScript's `OAuthCredentials` index signature does, so `gatewayConfig`
    lives at `data.gatewayConfig` here (see `Credential.data`'s docstring).
    """
    auth_path.write_text(
        json.dumps(
            {
                RADIUS_PROVIDER_ID: {
                    "type": "oauth",
                    "key": None,
                    "env": {},
                    "data": {
                        "gatewayConfig": {
                            "baseUrl": gateway_base_url,
                            "models": [
                                {
                                    "id": "auto",
                                    "name": "Radius Auto",
                                    "reasoning": False,
                                    "input": ["text"],
                                    "cost": {"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2},
                                    "contextWindow": 128000,
                                    "maxTokens": 16384,
                                }
                            ],
                        }
                    },
                }
            }
        )
    )


async def test_restores_the_legacy_credential_catalog_without_network_access(tmp_path: Path) -> None:
    """Port of TS 'restores the legacy credential catalog without network access'.

    TS builds the runtime with `allowModelNetwork: false` and a credential
    whose `gatewayConfig` already caches the gateway's model list, then
    asserts the model, provider name, and configured-auth status are all
    available with no network call. This port never makes a Radius network
    call at all (see the module docstring), so there is no `allowModelNetwork`
    flag to pass; the credential-seed path is unconditional.
    """
    auth_path = tmp_path / "auth.json"
    _write_radius_credential(auth_path, "https://radius.example.com/v1")

    runtime = await ModelRuntime.create(agent_dir=str(tmp_path), auth_path=str(auth_path), env={})

    model = runtime.get_model(RADIUS_PROVIDER_ID, "auto")
    assert model is not None
    assert model.api == "pi-messages"
    assert model.base_url == "https://radius.example.com/v1"
    provider = runtime.get_provider(RADIUS_PROVIDER_ID)
    assert provider is not None
    assert provider.name == "Radius"
    assert runtime.has_configured_auth(RADIUS_PROVIDER_ID) is True


@pytest.mark.skip(
    reason=(
        "TS 'fetches and stores the catalog for configured Radius auth' needs a mocked "
        "`fetch` of https://radius.pi.dev/v1/config, `allowModelNetwork: true`, and "
        "`ModelsStore.read()` to check the persisted catalog -- none of which "
        "`ModelRuntime` composes (see the module docstring)."
    )
)
def test_fetches_and_stores_the_catalog_for_configured_radius_auth() -> None:
    raise AssertionError("unreachable")


async def test_does_not_refresh_catalogs_over_the_network_by_default(tmp_path: Path) -> None:
    """Port of TS 'does not refresh catalogs over the network by default'.

    TS mocks `fetch` to reject and asserts it is never called while the model
    is still available from the credential's cached `gatewayConfig`. This port
    has no `fetch`/network path for Radius at all to spy on (see the module
    docstring), so the only portable half is that the model remains available
    without one.
    """
    auth_path = tmp_path / "auth.json"
    _write_radius_credential(auth_path, "https://radius.example.com/v1")

    runtime = await ModelRuntime.create(agent_dir=str(tmp_path), auth_path=str(auth_path), env={})

    assert runtime.get_model(RADIUS_PROVIDER_ID, "auto") is not None


async def test_does_not_fetch_or_expose_radius_models_without_configured_auth(tmp_path: Path) -> None:
    """Port of TS 'does not fetch or expose Radius models without configured auth'.

    The TS `fetchSpy.mock.calls...toBe(false)` half is trivially true here:
    this port never issues a Radius network request at all (see the module
    docstring), so only the model-list assertion is kept.
    """
    runtime = await _runtime(tmp_path)

    assert runtime.get_models(RADIUS_PROVIDER_ID) == []


async def test_supports_custom_radius_gateways_from_models_json(tmp_path: Path) -> None:
    """Port of TS 'supports custom Radius gateways from models.json'.

    TS mocks the gateway's `/v1/config` fetch, so its model list comes over the
    network; this port has no Radius network path (see the module docstring),
    so the model list is seeded from the stored credential's cached
    `gatewayConfig` instead -- the same catalog, restored rather than fetched.
    Everything else the TS case asserts (the gateway is promoted to a real
    Radius provider carrying the `pi-messages` api, the configured baseUrl, and
    the configured display name) is pinned exactly.
    """
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "radius-dev": {
                    "type": "oauth",
                    "key": None,
                    "env": {},
                    "data": {
                        "gatewayConfig": {
                            "baseUrl": "http://localhost:8788/v1",
                            "models": [
                                {
                                    "id": "auto",
                                    "name": "Radius Auto",
                                    "reasoning": False,
                                    "input": ["text"],
                                    "cost": {"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2},
                                    "contextWindow": 128000,
                                    "maxTokens": 16384,
                                }
                            ],
                        }
                    },
                }
            }
        )
    )
    models_path = tmp_path / "models.json"
    models_path.write_text(
        json.dumps(
            {
                "providers": {
                    "radius-dev": {
                        "name": "Radius (dev)",
                        "baseUrl": "http://localhost:8788",
                        "oauth": "radius",
                    }
                }
            }
        )
    )

    runtime = await ModelRuntime.create(
        agent_dir=str(tmp_path), auth_path=str(auth_path), models_path=str(models_path), env={}
    )

    assert runtime.get_error() is None
    model = runtime.get_model("radius-dev", "auto")
    assert model is not None
    assert model.api == "pi-messages"
    assert model.base_url == "http://localhost:8788/v1"
    provider = runtime.get_provider("radius-dev")
    assert provider is not None
    assert provider.name == "Radius (dev)"


async def test_custom_radius_gateways_offer_the_radius_oauth_flow(tmp_path: Path) -> None:
    """A `models.json` gateway is promoted to a full Radius provider.

    TypeScript's `configureRadiusProviders` replaces the config entry with
    `radiusProvider({id, name, gateway})`, so the gateway gets Radius's OAuth
    method and `pi-messages` api rather than composing as a plain config
    provider.
    """
    runtime = await _runtime(
        tmp_path,
        {
            "providers": {
                "radius-dev": {"name": "Radius (dev)", "baseUrl": "http://localhost:8788/v1", "oauth": "radius"}
            }
        },
    )

    provider = runtime.get_provider("radius-dev")
    assert provider is not None
    assert provider.auth is not None
    assert provider.auth.oauth is not None
    assert provider.auth.oauth.name == "Radius (dev)"


async def test_registers_the_radius_provider(tmp_path: Path) -> None:
    runtime = await _runtime(tmp_path)

    provider = runtime.get_provider(RADIUS_PROVIDER_ID)
    assert provider is not None
    assert provider.name == "Radius"


async def test_does_not_expose_radius_models_without_configured_auth(tmp_path: Path) -> None:
    runtime = await _runtime(tmp_path)

    assert runtime.get_models(RADIUS_PROVIDER_ID) == []
    assert runtime.has_configured_auth(RADIUS_PROVIDER_ID) is False


async def test_requires_base_url_for_custom_radius_gateways(tmp_path: Path) -> None:
    runtime = await _runtime(tmp_path, {"providers": {"radius-dev": {"oauth": "radius"}}})

    error = runtime.get_error()
    assert error is not None
    assert '"baseUrl" is required when "oauth" is set' in error


async def test_a_broken_provider_entry_does_not_break_the_runtime(tmp_path: Path) -> None:
    """TypeScript records composition failures per provider instead of throwing."""
    runtime = await _runtime(tmp_path, {"providers": {"radius-dev": {"oauth": "radius"}}})

    assert runtime.get_provider("radius-dev") is None
    assert runtime.get_provider(RADIUS_PROVIDER_ID) is not None
