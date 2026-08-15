"""Python port of `packages/coding-agent/test/suite/regressions/6768-copilot-compaction-base-url.test.ts`.

TS registers a hand-written Copilot-shaped provider into the live
`modelRuntime` (`registerNativeProvider` + `refresh({ providers })`). This port
has no runtime provider-registration API, so the same provider is handed to
`create_harness(provider_override=...)`, which is where the harness puts it
into `ModelRuntime.create(providers=[...])`. The behaviour under test -- the
OAuth-resolved base URL reaching the request model through the plain provider
dispatch -- is identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from harness import Harness, create_harness
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthResult,
    Credential,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.registry import Model, Provider
from pi_ai.types import Cost, DoneEvent, TextContent, Usage, UserMessage, now_ms
from pi_ai.utils.event_stream import create_assistant_message_event_stream

INDIVIDUAL_BASE_URL = "https://api.individual.githubcopilot.com"
ENTERPRISE_BASE_URL = "https://api.enterprise.githubcopilot.com"


def seed_compactable_session(harness: Harness) -> None:
    harness.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    now = now_ms()
    harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
    )
    model = harness.get_model()
    assert model is not None
    assistant = faux_assistant_message("assistant response to compact", timestamp=now - 500)
    assistant.api = model.api
    assistant.provider = model.provider
    assistant.model = model.id
    assistant.usage = Usage(
        input=100,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=100,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )
    harness.session_manager.append_message(assistant)
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages


@pytest.fixture
def harnesses() -> list[Harness]:
    created: list[Harness] = []
    yield created
    while created:
        created.pop().cleanup()


async def test_uses_the_auth_resolved_base_url_through_the_stream_wrapper(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    request_base_urls: list[str | None] = []

    class _Api:
        def _respond(self, request_model: Model, *_args: Any, **_kwargs: Any):
            request_base_urls.append(request_model.base_url)
            stream = create_assistant_message_event_stream()
            message = faux_assistant_message("summary")
            message.api = request_model.api
            message.provider = request_model.provider
            message.model = request_model.id
            stream.push(DoneEvent(reason="stop", message=message))
            stream.end(message)
            return stream

        def stream(self, model: Model, context: Any, options: Any = None, **kwargs: Any):
            return self._respond(model, context, options, **kwargs)

        def stream_simple(self, model: Model, context: Any, options: Any = None, **kwargs: Any):
            return self._respond(model, context, options, **kwargs)

    def build_provider(faux_provider: Provider) -> Provider:
        catalog_model = faux_provider.models[0]
        catalog_model = type(catalog_model)(  # type: ignore[call-arg]
            **{**catalog_model.__dict__, "base_url": INDIVIDUAL_BASE_URL}
        )

        async def resolve(*, credential: Credential | None = None, env: Any = None) -> AuthResult | None:
            if credential is not None and credential.key:
                return AuthResult(auth=ResolvedAuth(api_key=credential.key), source="explicit token")
            return None

        async def login(_interaction: Any) -> Credential:
            raise RuntimeError("unused")

        async def refresh(credential: Credential, _signal: Any = None) -> Credential:
            return credential

        async def to_auth(credential: Credential) -> ResolvedAuth:
            return ResolvedAuth(api_key=credential.access, base_url=ENTERPRISE_BASE_URL)

        return Provider(
            id=faux_provider.id,
            name="Copilot regression provider",
            base_url=INDIVIDUAL_BASE_URL,
            auth=ProviderAuth(
                api_key=ApiKeyAuth(name="Copilot token", resolve=resolve),
                oauth=OAuthAuth(name="Copilot OAuth", login=login, refresh=refresh, to_auth=to_auth),
            ),
            api=_Api(),
            models=[catalog_model],
        )

    harness = await create_harness(tmp_path, provider_override=build_provider, with_configured_auth=False)
    harnesses.append(harness)
    seed_compactable_session(harness)

    catalog_model = harness.session.model_runtime.get_model(harness.models[0].provider, harness.models[0].id)
    assert catalog_model is not None
    assert catalog_model.base_url == INDIVIDUAL_BASE_URL
    harness.session.agent.state.model = catalog_model

    await harness.auth_storage.set(
        catalog_model.provider,
        Credential(
            type="oauth",
            access="enterprise-token",
            refresh="refresh-token",
            expires=now_ms() + 60 * 60_000,
        ),
    )

    model_runtime = harness.session.model_runtime
    harness.session.agent.stream_function = model_runtime.stream_simple

    await harness.session.compact()

    assert request_base_urls and request_base_urls[-1] == ENTERPRISE_BASE_URL
