"""Python port of `packages/coding-agent/test/runtime-credentials.test.ts`."""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.auth.types import Credential, CredentialInfo
from pi_ai.types import now_ms
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.runtime_credentials import RuntimeCredentials


async def test_runtime_overrides_mask_stored_credentials_without_persisting() -> None:
    storage = AuthStorage.in_memory({"anthropic": Credential(type="api_key", key="stored-key")})
    credentials = RuntimeCredentials(storage)

    credentials.set_runtime_api_key("anthropic", "runtime-key")
    assert await credentials.get("anthropic") == Credential(type="api_key", key="runtime-key")
    assert await storage.get("anthropic") == Credential(type="api_key", key="stored-key")

    credentials.remove_runtime_api_key("anthropic")
    assert await credentials.get("anthropic") == Credential(type="api_key", key="stored-key")


async def test_enumeration_merges_overrides_without_exposing_keys() -> None:
    storage = AuthStorage.in_memory(
        {
            "anthropic": Credential(
                type="oauth",
                data={"access": "access", "refresh": "refresh", "expires": now_ms() + 60_000},
            )
        }
    )
    credentials = RuntimeCredentials(storage)
    credentials.set_runtime_api_key("anthropic", "runtime-key")
    credentials.set_runtime_api_key("openai", "other-runtime-key")

    assert await credentials.list() == [
        CredentialInfo(provider_id="anthropic", type="api_key"),
        CredentialInfo(provider_id="openai", type="api_key"),
    ]


# The TypeScript case "forwards operation signals to the persistent store" has
# no counterpart: TypeScript's `CredentialStore` takes a per-operation
# `{ signal }` options bag on `read`/`list`/`modify`/`delete`, and the test
# asserts `RuntimeCredentials` forwards it unchanged. This port's
# `pi_ai.auth.types.CredentialStore` protocol is `get`/`set`/`delete` with no
# options argument (cancellation is `asyncio.CancelledError` on the awaiting
# task, per the README's `AbortSignal` convention), so there is no signal to
# forward and nothing to assert.


@pytest.mark.skip(
    reason=(
        "TS 'forwards operation signals to the persistent store' needs the "
        "per-operation `{ signal }` options bag on `CredentialStore`, which this "
        "port replaces with `asyncio.CancelledError` on the awaiting task."
    )
)
def test_forwards_operation_signals_to_the_persistent_store() -> None:
    raise AssertionError("unreachable")


async def test_keeps_a_runtime_override_when_persistent_deletion_fails() -> None:
    """Port of "keeps a runtime override when persistent deletion is cancelled".

    TypeScript rejects the wrapped `delete` with an `AbortError`; the Python
    equivalent of an aborted await is `asyncio.CancelledError`, and the
    behaviour under test is the same either way: the override survives because
    `RuntimeCredentials.delete` only clears it *after* the wrapped store
    succeeds.
    """
    storage = AuthStorage.in_memory({"anthropic": Credential(type="api_key", key="stored-key")})
    delete_calls: list[str] = []

    async def failing_delete(provider_id: str) -> None:
        delete_calls.append(provider_id)
        raise asyncio.CancelledError

    storage.delete = failing_delete  # type: ignore[method-assign]
    credentials = RuntimeCredentials(storage)
    credentials.set_runtime_api_key("anthropic", "runtime-key")

    with pytest.raises(asyncio.CancelledError):
        await credentials.delete("anthropic")

    assert delete_calls == ["anthropic"]
    assert await credentials.get("anthropic") == Credential(type="api_key", key="runtime-key")


async def test_delete_clears_both_the_override_and_persisted_credential() -> None:
    storage = AuthStorage.in_memory({"anthropic": Credential(type="api_key", key="stored-key")})
    credentials = RuntimeCredentials(storage)
    credentials.set_runtime_api_key("anthropic", "runtime-key")

    await credentials.delete("anthropic")

    assert await credentials.get("anthropic") is None
    assert await credentials.list() == []
