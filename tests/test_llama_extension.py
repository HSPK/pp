"""Python port of `packages/coding-agent/test/llama-extension.test.ts`.

The llama.cpp local-model extension (`src/extensions/llama/*`) is not ported.
The README names it explicitly as the largest unported piece of the coding
agent, and it depends on three further layers this port also lacks:

- `pi.registerProvider` / `runtime.pendingNativeProviderRegistrations` --
  extension provider registration (see `core/extensions/types.py`).
- `Provider.refreshModels` + `ModelsPublication` -- the dynamic model
  catalog/registry/store refresh layer. `pi_ai.models_store` has the
  `ModelsStore`/`ModelsStoreEntry` types, but no provider exposes
  `refresh_models` and there is no `RefreshModelsContext`/`ModelsPublication`.
- The llama router's HTTP/SSE client and its Hugging Face client.

Every TypeScript case is recorded below with what it asserts, so the gap stays
visible and a future port has the spec in one place.
"""

from __future__ import annotations

import importlib.util

import pytest

_REASON = (
    "The llama.cpp extension (src/extensions/llama/*) is not ported; see the "
    "README's unported list. It also needs extension provider registration and "
    "the Provider.refreshModels/ModelsPublication catalog layer, neither of "
    "which exists in this port."
)


def test_llama_extension_module_is_absent() -> None:
    """Pins the documented boundary the skips below rely on."""
    assert importlib.util.find_spec("pi_coding_agent.extensions") is None


@pytest.mark.skip(reason=_REASON)
def test_registers_a_native_provider_and_llama_command() -> None:
    """`it("registers a native provider and /llama command")`.

    After `loadExtensionFromFactory(llamaExtension, ...)`:
    `extension.commands.get("llama").description == "Manage llama.cpp router models"`
    and `runtime.pendingNativeProviderRegistrations` holds exactly one entry
    whose `provider.id` is `LLAMA_PROVIDER_ID`.
    """


@pytest.mark.skip(reason=_REASON)
def test_normalizes_management_and_inference_urls() -> None:
    """`it("normalizes management and inference URLs")`.

    `normalizeLlamaServerUrl("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"`,
    `normalizeLlamaServerUrl("https://example.com/prefix/v1") == "https://example.com/prefix"`,
    and `normalizeLlamaServerUrl("file:///tmp/llama")` throws with a message
    containing `"http or https"`.
    """


@pytest.mark.skip(reason=_REASON)
def test_exposes_only_loaded_models_with_router_metadata() -> None:
    """`it("exposes only loaded models with router metadata")`.

    Given a catalog of `loaded`/`unloaded`/`loading` entries,
    `controller.provider.getModels()` returns exactly one model: id `loaded`,
    `baseUrl` = catalog URL + `/v1`, `contextWindow == 65536` (from
    `meta.n_ctx`, *not* `n_ctx_train`), `maxTokens == 65536`, and
    `input == ["text", "image"]` (from `architecture.input_modalities`).
    """


@pytest.mark.skip(reason=_REASON)
def test_persists_and_restores_loaded_models_for_cache_only_startup_refreshes() -> None:
    """`it("persists and restores loaded models for cache-only startup refreshes")`.

    First refresh with `allowNetwork: true` against a live `/models` endpoint
    yields `["loaded"]` from `getModels()` and persists `["loaded"]` into the
    publication's `persist` entry. A second, fresh provider refreshed with
    `allowNetwork: false` and only the cached entry still reports the `loaded`
    model with `baseUrl == url + "/v1"` and `contextWindow == 32768`.
    """


@pytest.mark.skip(reason=_REASON)
def test_stays_dormant_until_configured_and_stores_url_plus_optional_key() -> None:
    """`it("stays dormant until configured and stores URL plus optional key")`.

    With an empty `AuthContext`, both `auth.check(...)` and `auth.resolve(...)`
    are `undefined`. After `auth.login` answers the two prompts with the server
    URL then `"secret"`, the credential is exactly
    `{type: "api_key", key: "secret", env: {LLAMA_BASE_URL: url}}`, and
    resolving it yields
    `{auth: {apiKey: "secret", baseUrl: url + "/v1"}, env: {LLAMA_BASE_URL: url}, source: "stored credential"}`.
    The probe request during login sends a redacted `authorization: ******`
    header.
    """


@pytest.mark.skip(reason=_REASON)
def test_searches_hugging_face_and_reads_quantizations_plus_access_requirements() -> None:
    """`it("searches Hugging Face and reads quantizations plus access requirements")`.

    `client.search("qwen coder")` hits `/api/models?` with
    `search=qwen coder`, `filter=gguf`, `sort=downloads` and returns the raw
    list. `client.details("owner/model-GGUF")` returns `gated: "manual"` plus
    quantizations sorted by name with **split shards summed**
    (`Q4_K_M-00001-of-00002` 2000 + `Q4_K_M-00002-of-00002` 3000 -> 5000) and
    `mmproj-*` excluded. `findHuggingFaceToken({HF_TOKEN: " hf-secret "})`
    trims to `"hf-secret"`.
    """


@pytest.mark.skip(reason=_REASON)
def test_loads_with_sse_progress_and_waits_for_the_loaded_catalog_state() -> None:
    """`it("loads with SSE progress and waits for the loaded catalog state")`.

    `LlamaClient.loadAndWait("test-model", onProgress)` POSTs `/models/load`,
    consumes `/models/sse` `status_change` events, resolves only once the
    catalog reports `status.value == "loaded"`, and reports the stage-derived
    progress message `"Loading text model"`.
    """


@pytest.mark.skip(reason=_REASON)
def test_downloads_with_byte_progress_and_returns_the_refreshed_catalog() -> None:
    """`it("downloads with byte progress and returns the refreshed catalog")`.

    `LlamaClient.downloadAndWait("owner/repo:Q4_K_M", onProgress)` resolves to
    the refreshed catalog `[{id: "owner/repo:Q4_K_M", status: {value: "unloaded"}}]`,
    and emits a progress entry exactly equal to
    `{message: "Downloading model", ratio: 0.5, detail: "512 B / 1.00 KiB"}`
    (note the byte formatting: decimal-free `B`, two-decimal `KiB`).
    """
