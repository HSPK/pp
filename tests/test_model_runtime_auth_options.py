"""Python port of `packages/coding-agent/test/model-runtime-auth-options.test.ts`.

TypeScript's helper builds "auth options" by flattening
`runtime.getProviders()` into one entry per provider auth method; the same
flattening is done here by `_auth_options`.

Six of the eleven TypeScript cases exercise `runtime.registerProvider(...)`,
the extension-provider layer. `core/model_runtime.py`'s module docstring
records that layer as deliberately unported ("No extension providers
(`ProviderConfigInput`, `validateExtensionProvider`, ...)"), so those cases are
skipped individually at the bottom of this file with the reason on each marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pi_ai.auth.types import Credential, CredentialStore, InMemoryCredentialStore
from pi_ai.models import ModelsError
from pi_ai.registry import Provider
from pi_ai.types import now_ms

from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime


class _AuthOption:
    def __init__(self, option_type: Literal["oauth", "api_key"], provider: Provider, method: object) -> None:
        self.type = option_type
        self.provider = provider
        self.method = method


def _auth_options(runtime: ModelRuntime, option_type: str | None = None) -> list[_AuthOption]:
    options: list[_AuthOption] = []
    for provider in runtime.get_providers():
        oauth = getattr(provider.auth, "oauth", None)
        api_key = getattr(provider.auth, "api_key", None)
        if oauth is not None and option_type in (None, "oauth"):
            options.append(_AuthOption("oauth", provider, oauth))
        if api_key is not None and option_type in (None, "api_key"):
            options.append(_AuthOption("api_key", provider, api_key))
    return options


async def _create(tmp_path: Path, credentials: CredentialStore) -> ModelRuntime:
    return await ModelRuntime.create(
        agent_dir=str(tmp_path),
        credentials=credentials,
        models_path=str(tmp_path / "models.json"),
        env={},
    )


async def test_accepts_a_pi_ai_credential_store(tmp_path: Path) -> None:
    credentials = InMemoryCredentialStore()
    await credentials.set("anthropic", Credential(type="api_key", key="stored-key"))
    runtime = await _create(tmp_path, credentials)

    auth = await runtime.get_auth("anthropic")
    assert auth is not None
    assert auth.auth.api_key == "stored-key"


async def test_scopes_provider_availability_reads_and_records_refresh_failures(tmp_path: Path) -> None:
    base = InMemoryCredentialStore()
    reads: list[str] = []
    fail_reads = {"value": False}

    class _SpyStore(CredentialStore):
        async def get(self, provider_id: str) -> Credential | None:
            reads.append(provider_id)
            if fail_reads["value"]:
                raise RuntimeError(f"read failed for {provider_id}")
            return await base.get(provider_id)

        async def set(self, provider_id: str, credential: Credential) -> None:
            await base.set(provider_id, credential)

        async def delete(self, provider_id: str) -> None:
            await base.delete(provider_id)

        async def list(self):
            return await base.list()

    runtime = await _create(tmp_path, _SpyStore())

    reads.clear()
    await runtime.get_available("anthropic")
    assert set(reads) == {"anthropic"}

    fail_reads["value"] = True
    with pytest.raises(ModelsError, match="Credential store read failed for anthropic"):
        await runtime.get_available("anthropic")
    error = runtime.get_error()
    assert error is not None
    assert "Availability refresh: Credential store read failed for anthropic" in error

    fail_reads["value"] = False
    await runtime.get_available()
    assert runtime.get_error() is None


async def test_projects_provider_owned_methods_names_and_status(tmp_path: Path) -> None:
    runtime = await _create(tmp_path, AuthStorage.in_memory())
    options = _auth_options(runtime)

    def _find(provider_id: str, option_type: str) -> _AuthOption:
        return next(o for o in options if o.provider.id == provider_id and o.type == option_type)

    bedrock = _find("amazon-bedrock", "api_key")
    assert bedrock.provider.name == "Amazon Bedrock"
    assert bedrock.method.name == "AWS credentials or bearer token"

    vertex = _find("google-vertex", "api_key")
    assert vertex.provider.name == "Google Vertex AI"
    assert vertex.method.name == "Google Cloud credentials"

    assert _find("anthropic", "oauth").provider.name == "Anthropic"
    assert _find("cloudflare-ai-gateway", "api_key").provider.name == "Cloudflare AI Gateway"
    assert _find("cloudflare-workers-ai", "api_key").provider.name == "Cloudflare Workers AI"

    assert all(o.type == "api_key" for o in _auth_options(runtime, "api_key"))
    assert all(o.type == "oauth" for o in _auth_options(runtime, "oauth"))

    # TS additionally asserts no `openai-codex` api_key option exists (upstream
    # declares that provider OAuth-only). `pi_ai.providers.openai_codex` gives
    # it a placeholder `ApiKeyAuth` with no env vars because
    # `pi_ai.auth.types.ProviderAuth` requires an `api_key` entry -- that
    # decision belongs to `pi-ai`, which is outside this package.


async def test_attaches_the_providers_active_auth_status_to_every_method_option(tmp_path: Path) -> None:
    runtime = await _create(
        tmp_path,
        AuthStorage.in_memory(
            {
                "anthropic": Credential(
                    type="oauth",
                    data={"access": "access", "refresh": "refresh", "expires": now_ms() + 60_000},
                )
            }
        ),
    )

    options = [o for o in _auth_options(runtime) if o.provider.id == "anthropic"]
    assert len(options) == 2

    check = await runtime.check_auth("anthropic")
    assert check is not None
    assert check.type == "oauth"


async def test_distinguishes_subscription_oauth_from_generic_oauth_sign_in(tmp_path: Path) -> None:
    runtime = await _create(
        tmp_path,
        AuthStorage.in_memory(
            {
                "anthropic": Credential(
                    type="oauth",
                    data={
                        "access": "anthropic-access",
                        "refresh": "anthropic-refresh",
                        "expires": now_ms() + 60 * 60_000,
                    },
                ),
                "openrouter": Credential(
                    type="oauth",
                    data={"access": "openrouter-key", "refresh": "", "expires": 2**53 - 1},
                ),
                "radius": Credential(
                    type="oauth",
                    data={
                        "access": "radius-access",
                        "refresh": "radius-refresh",
                        "expires": now_ms() + 60 * 60_000,
                    },
                ),
            }
        ),
    )

    assert runtime.is_using_oauth("anthropic") is True
    assert runtime.is_using_subscription("anthropic") is True
    assert runtime.is_using_oauth("openrouter") is True
    assert runtime.is_using_subscription("openrouter") is False
    assert runtime.is_using_oauth("radius") is True
    assert runtime.is_using_subscription("radius") is False


_EXTENSION_PROVIDER_REASON = (
    "needs `runtime.registerProvider(...)`, the extension-provider layer that "
    "`core/model_runtime.py`'s module docstring records as deliberately unported "
    "(no `ProviderConfigInput`/`validateExtensionProvider`/`extensionProviders` map)."
)


@pytest.mark.skip(
    reason="`constructs an API key method for an extension API-key provider` " + _EXTENSION_PROVIDER_REASON
)
def test_constructs_an_api_key_method_for_an_extension_api_key_provider() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason="`resolves configured auth from request-scoped environment overrides` "
    + _EXTENSION_PROVIDER_REASON
    + " It also needs `getAuth(providerId, { env })`; this port's `get_auth` takes no "
    "per-request env override, because the env lookup is fixed at `ModelRuntime.create`."
)
def test_resolves_configured_auth_from_request_scoped_environment_overrides() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason="`lets an explicit Authorization header override authHeader case-insensitively` "
    + _EXTENSION_PROVIDER_REASON
)
def test_lets_an_explicit_authorization_header_override_auth_header() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason="`transforms fully assembled headers once without forwarding the transform` " + _EXTENSION_PROVIDER_REASON
)
def test_transforms_fully_assembled_headers_once() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason="`forwards cancellation to extension OAuth refresh` " + _EXTENSION_PROVIDER_REASON)
def test_forwards_cancellation_to_extension_oauth_refresh() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason="`does not fabricate an API key method for an extension OAuth-only provider` " + _EXTENSION_PROVIDER_REASON
)
def test_does_not_fabricate_an_api_key_method_for_an_extension_oauth_only_provider() -> None:
    raise AssertionError("unreachable")
