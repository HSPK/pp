"""In-memory credential overrides layered over a persistent store.

Ported from ``packages/coding-agent/src/core/runtime-credentials.ts``.

``pi --api-key sk-...`` and a ``/login`` that the user does not want saved must
authenticate the current process without ever touching ``auth.json``.
`RuntimeCredentials` wraps the real store and answers from an in-memory
override first, falling back to disk for every provider that has no override.
Writes (`set`) still go to the wrapped store, and `delete` clears both, so a
logout cannot leave a live override behind.
"""

from __future__ import annotations

from pi_ai.auth.types import Credential, CredentialInfo, CredentialStore


class RuntimeCredentials(CredentialStore):
    def __init__(self, store: CredentialStore) -> None:
        self._store = store
        self._overrides: dict[str, str] = {}

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        self._overrides[provider_id] = api_key

    def remove_runtime_api_key(self, provider_id: str) -> None:
        self._overrides.pop(provider_id, None)

    def has_runtime_api_key(self, provider_id: str) -> bool:
        return provider_id in self._overrides

    async def get(self, provider_id: str) -> Credential | None:
        override = self._overrides.get(provider_id)
        if override:
            return Credential(type="api_key", key=override)
        return await self._store.get(provider_id)

    async def set(self, provider_id: str, credential: Credential) -> None:
        await self._store.set(provider_id, credential)

    async def delete(self, provider_id: str) -> None:
        await self._store.delete(provider_id)
        self._overrides.pop(provider_id, None)

    def has_sync(self, provider_id: str) -> bool:
        """Synchronous existence check across both layers.

        `has_configured_auth` and the login/logout selectors call this on every
        render, so the overlay has to answer synchronously too; without it a
        wrapped store looks unconfigured.
        """
        if provider_id in self._overrides:
            return True
        has_sync = getattr(self._store, "has_sync", None)
        return bool(has_sync(provider_id)) if has_sync is not None else False

    def get_sync(self, provider_id: str) -> Credential | None:
        override = self._overrides.get(provider_id)
        if override:
            return Credential(type="api_key", key=override)
        get_sync = getattr(self._store, "get_sync", None)
        return get_sync(provider_id) if get_sync is not None else None

    async def list(self) -> list[CredentialInfo]:
        """Provider ids with a credential *and their type*, from either layer.

        Port of TS `RuntimeCredentials.list`: a runtime `--api-key` override
        always reports as `api_key`, shadowing any stored OAuth entry.
        """
        entries: dict[str, CredentialInfo] = {}
        for info in await self._store.list():
            entries[info.provider_id] = info
        for provider_id in self._overrides:
            entries[provider_id] = CredentialInfo(provider_id=provider_id, type="api_key")
        return list(entries.values())


__all__ = ["RuntimeCredentials"]
