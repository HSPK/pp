"""Python port of `packages/coding-agent/test/model-runtime-cloudflare-compat.test.ts`.

Cloudflare AI Gateway is the awkward provider: its base URL is a *template*
(`.../v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat`) filled in from
the credential's `env`, and it authenticates the gateway with a
`cf-aig-authorization` header while explicitly nulling `Authorization` and
`x-api-key` so the upstream provider key can travel separately. The TypeScript
test pins that both the URL and those headers survive the trip from
`ModelRuntime` down to the OpenAI client, both through `ModelRuntime` streaming
and through "extension-style" auth resolution
(`ModelRegistry.getApiKeyAndHeaders` -> `compat.complete`).

Two adaptations:

- `ModelRegistry` is not ported (see `core/model_runtime.py` -- this port has no
  dynamic registry/catalog layer), so the second scenario resolves auth with
  `ModelRuntime.get_auth`, which is the method the missing registry delegated
  to, and then calls `pi_ai.compat.complete` exactly as the TypeScript does.
- TypeScript mocks the whole `openai` package and inspects the constructor
  options. This port speaks HTTP directly through `httpx`, so the assertions are
  made against the real outgoing request captured by an `httpx.MockTransport` --
  the same facts (materialized URL, `cf-aig-authorization`, no `Authorization`,
  no `x-api-key`), observed one layer closer to the wire.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from pi_ai.auth.types import Credential
from pi_ai.compat import complete, reset_api_providers
from pi_ai.types import Context, StreamOptions
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime

CLOUDFLARE_PROVIDER = "cloudflare-ai-gateway"
CLOUDFLARE_MODEL = "workers-ai/@cf/moonshotai/kimi-k2.5"
EXPECTED_BASE_URL = "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway/compat"

_COMPLETION_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": CLOUDFLARE_MODEL,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _sse_stream_body() -> str:
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": CLOUDFLARE_MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


class _RequestRecorder:
    """Captures the one outgoing request so it can be asserted on."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("accept") == "text/event-stream" or b'"stream": true' in request.content:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=_sse_stream_body(),
            )
        return httpx.Response(200, json=_COMPLETION_BODY)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    @property
    def only(self) -> httpx.Request:
        assert len(self.requests) == 1, f"expected exactly one request, got {len(self.requests)}"
        return self.requests[0]


async def _create_cloudflare_runtime(tmp_path: Any) -> ModelRuntime:
    auth_storage = AuthStorage.in_memory()
    await auth_storage.set(
        CLOUDFLARE_PROVIDER,
        Credential(
            type="api_key",
            key="test-token",
            env={"CLOUDFLARE_ACCOUNT_ID": "test-account", "CLOUDFLARE_GATEWAY_ID": "test-gateway"},
        ),
    )
    return await ModelRuntime.create(
        agent_dir=str(tmp_path),
        credentials=auth_storage,
        models_path=str(tmp_path / "models.json"),
        env={}.get,  # type: ignore[arg-type]
    )


def test_materializes_the_cloudflare_endpoint_through_model_runtime_streaming(tmp_path: Any) -> None:
    async def scenario() -> None:
        runtime = await _create_cloudflare_runtime(tmp_path)
        model = runtime.get_model(CLOUDFLARE_PROVIDER, CLOUDFLARE_MODEL)
        assert model is not None

        reset_api_providers()
        recorder = _RequestRecorder()
        async with recorder.client() as client:
            stream = await runtime.stream_simple(model, Context(messages=[]), None, client=client)
            await stream.result()

        request = recorder.only
        assert str(request.url).startswith(EXPECTED_BASE_URL)
        assert request.headers.get("cf-aig-authorization") == "Bearer test-token"

    asyncio.run(scenario())


def test_materializes_the_cloudflare_endpoint_after_extension_style_auth_resolution(tmp_path: Any) -> None:
    async def scenario() -> None:
        runtime = await _create_cloudflare_runtime(tmp_path)
        # `ModelRegistry.find` in TypeScript; this port resolves through the
        # runtime directly because `ModelRegistry` is not ported.
        model = runtime.get_model(CLOUDFLARE_PROVIDER, CLOUDFLARE_MODEL)
        assert model is not None

        reset_api_providers()
        # `ModelRegistry.getApiKeyAndHeaders(model)` -> `ModelRuntime.get_auth`.
        auth = await runtime.get_auth(model)
        assert auth is not None
        assert auth.auth is not None
        assert auth.auth.headers == {
            "cf-aig-authorization": "Bearer test-token",
            "Authorization": None,
            "x-api-key": None,
        }

        recorder = _RequestRecorder()
        async with recorder.client() as client:
            await complete(
                model,
                Context(messages=[]),
                StreamOptions(api_key=auth.auth.api_key, headers=auth.auth.headers, env=auth.env),
                client=client,
            )

        request = recorder.only
        assert str(request.url).startswith(EXPECTED_BASE_URL)
        assert request.headers.get("cf-aig-authorization") == "Bearer test-token"
        # `Authorization: null` / `x-api-key: null` mean "do not send"; nothing
        # may fall back to the stored key in those slots.
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers

    asyncio.run(scenario())


@pytest.mark.skip(
    reason=(
        "ModelRegistry is not ported (see core/model_runtime.py: this port has "
        "no dynamic model-registry/catalog layer), so there is no "
        "ModelRegistry.find / getApiKeyAndHeaders whose `ok`/`error` result "
        "shape could be asserted. The underlying auth resolution it delegates "
        "to is covered by the test above."
    )
)
def test_model_registry_get_api_key_and_headers_reports_ok() -> None:
    """`expect(auth.ok).toBe(true)`; `if (!auth.ok) throw new Error(auth.error)`."""
    raise AssertionError("unreachable")
