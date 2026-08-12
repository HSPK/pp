"""Credential storage backed by ``auth.json``.

Port of `packages/coding-agent/src/core/auth-storage.ts`. Provider auth
orchestration belongs to `model_runtime.py` / pi-ai `Models`; this module only
owns reading and writing the credential file.

Simplifications versus the TypeScript original:

- `pi_ai.auth.types.CredentialStore` (the interface this module implements)
  was already simplified in this port to ``get``/``set``/``delete`` only -
  there is no ``read``/``modify``/``list``/`AuthOperationOptions`
  cancellation surface to fill in. That removes the need for TypeScript's
  read-modify-write ``modify()`` closures; ``set()`` here simply
  read-locks-merges-writes.
- File locking uses the POSIX-only stdlib `fcntl.flock` (advisory,
  process-local exclusive lock on the open file descriptor) instead of the
  `proper-lockfile` npm package's marker-file/staleness protocol. This is
  sufficient for this port's target platform (Linux) and test usage
  (single-process); it does not protect against a process that crashes while
  holding the lock leaving a stale marker (proper-lockfile's main extra
  feature), which is an accepted simplification. The async lock path holds
  the file lock for the duration of the (already-local, non-networked)
  read-transform-write critical section rather than offloading to a worker
  thread; TypeScript's async retry/backoff/abort-signal-aware lock
  acquisition loop is dropped as unneeded for a single-process Python target.
- TypeScript's `AuthStorage` deduplicates concurrent async reloads across
  callers via a shared `AuthFileReload` (readers counter + abort-on-last-
  reader) and a process-wide `sharedAuthFileReadState`. This port drops that
  de-duplication: each `AuthStorage` instance keeps its own revision-cached
  snapshot (`get_file_revision`-gated, same as TypeScript) but concurrent
  reloads are not coalesced. Correctness is unaffected (each caller still
  reads a consistent snapshot); at most it means redundant reads under heavy
  concurrent contention, which does not occur in normal CLI/agent usage.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_ai.auth.types import Credential, CredentialInfo, CredentialStore

from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resolve_config_value import is_command_config_value, resolve_config_value
from pi_coding_agent.utils.paths import get_file_revision, normalize_path

_AUTH_FILE_MODE = 0o600
_AUTH_DIR_MODE = 0o700

AuthStorageData = dict[str, Credential]


def _default_auth_path() -> str:
    return os.path.join(get_agent_dir(), "auth.json")


@dataclass
class LockResult:
    """Result of a `with_lock`/`with_lock_async` transform.

    `next_content` is `None` to leave the file untouched, or the new raw file
    content to write back (mirrors TypeScript's `LockResult.next` being
    `undefined`-or-a-string).
    """

    result: Any
    next_content: str | None = None


class AuthStorageBackend:
    """Where `AuthStorage` reads/writes its raw JSON text under a lock."""

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        raise NotImplementedError

    async def with_lock_async(self, fn: Callable[[str | None], LockResult]) -> Any:
        raise NotImplementedError


class FileAuthStorageBackend(AuthStorageBackend):
    """Backend that locks and reads/writes a real file on disk."""

    def __init__(self, auth_path: str | None = None) -> None:
        self.auth_path = normalize_path(auth_path or _default_auth_path())

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(self.auth_path)
        if parent:
            os.makedirs(parent, mode=_AUTH_DIR_MODE, exist_ok=True)

    def _ensure_file_exists(self) -> None:
        if not os.path.exists(self.auth_path):
            fd = os.open(self.auth_path, os.O_CREAT | os.O_WRONLY, _AUTH_FILE_MODE)
            try:
                os.write(fd, b"{}")
            finally:
                os.close(fd)
            os.chmod(self.auth_path, _AUTH_FILE_MODE)

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        self._ensure_parent_dir()
        self._ensure_file_exists()
        with open(self.auth_path, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                current = handle.read()
                outcome = fn(current if current else None)
                if outcome.next_content is not None:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(outcome.next_content)
                    handle.flush()
                    os.fchmod(handle.fileno(), _AUTH_FILE_MODE)
                return outcome.result
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    async def with_lock_async(self, fn: Callable[[str | None], Any]) -> Any:
        self._ensure_parent_dir()
        self._ensure_file_exists()
        with open(self.auth_path, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                current = handle.read()
                outcome = await fn(current if current else None)
                if outcome.next_content is not None:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(outcome.next_content)
                    handle.flush()
                    os.fchmod(handle.fileno(), _AUTH_FILE_MODE)
                return outcome.result
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class InMemoryAuthStorageBackend(AuthStorageBackend):
    """Backend that keeps the raw JSON text in memory, for tests."""

    def __init__(self) -> None:
        self._value: str | None = None
        self._lock = asyncio.Lock()

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        outcome = fn(self._value)
        if outcome.next_content is not None:
            self._value = outcome.next_content
        return outcome.result

    async def with_lock_async(self, fn: Callable[[str | None], Any]) -> Any:
        async with self._lock:
            outcome = await fn(self._value)
            if outcome.next_content is not None:
                self._value = outcome.next_content
            return outcome.result


def _parse_storage_data(content: str | None) -> AuthStorageData:
    """Lenient parse (mirrors TypeScript's `parseStorageData`: blind JSON.parse)."""
    if not content:
        return {}
    raw = json.loads(content)
    return {provider_id: _credential_from_dict(value) for provider_id, value in raw.items()}


def _credential_from_dict(value: dict[str, Any]) -> Credential:
    return Credential(
        type=value.get("type", "api_key"),
        key=value.get("key"),
        env=dict(value.get("env") or {}),
        data=dict(value.get("data") or {}),
        access=value.get("access"),
        refresh=value.get("refresh"),
        expires=value.get("expires"),
    )


def _credential_to_dict(credential: Credential) -> dict[str, Any]:
    if credential.type == "oauth":
        out: dict[str, Any] = {
            "type": "oauth",
            "access": credential.access,
            "refresh": credential.refresh,
            "expires": credential.expires,
        }
    else:
        out = {"type": "api_key"}
        if credential.key is not None:
            out["key"] = credential.key
        if credential.env:
            out["env"] = credential.env
    if credential.data:
        out["data"] = credential.data
    return out


def _serialize_storage_data(data: AuthStorageData) -> str:
    return json.dumps({provider_id: _credential_to_dict(c) for provider_id, c in data.items()}, indent=2)


def _validate_credential_dict(provider_id: str, value: Any) -> None:
    """Strict validation used only by `ReadOnlyAuthStorage.load()` (matches TS)."""
    if not isinstance(value, dict):
        raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')
    if value.get("type") == "api_key":
        key = value.get("key")
        valid_key = key is None or isinstance(key, str)
        env = value.get("env")
        valid_env = env is None or (isinstance(env, dict) and all(isinstance(v, str) for v in env.values()))
        if valid_key and valid_env:
            return
    elif (
        value.get("type") == "oauth"
        and isinstance(value.get("access"), str)
        and isinstance(value.get("refresh"), str)
        and isinstance(value.get("expires"), (int, float))
    ):
        return
    raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')


def _resolve_read_only_api_key_credential(credential: Credential) -> Credential:
    """Port of `ReadOnlyAuthStorage.read()`'s resolution rule: skips resolving
    command-backed (`!cmd`) keys -- unlike `AuthStorage.get()` (see
    `_resolve_api_key_credential` below), matching TypeScript's two distinct
    `read()` implementations exactly (`auth-storage.ts` lines ~259-268 vs.
    ~442-448)."""
    if credential.type != "api_key" or not credential.key or is_command_config_value(credential.key):
        return credential
    resolved = resolve_config_value(credential.key, credential.env)
    return Credential(type=credential.type, key=resolved, env=credential.env, data=credential.data)


def _resolve_api_key_credential(credential: Credential) -> Credential:
    """Port of `AuthStorage.read()`'s resolution rule: always resolves an
    `api_key` credential's `key` (env-interpolated or, notably, command-backed
    `!cmd` values too) when a key is present, unlike `ReadOnlyAuthStorage`'s
    stricter variant above."""
    if credential.type != "api_key" or credential.key is None:
        return credential
    resolved = resolve_config_value(credential.key, credential.env)
    return Credential(type=credential.type, key=resolved, env=credential.env, data=credential.data)


class ReadOnlyAuthStorage(CredentialStore):
    """Strictly-validated, load-once view of `auth.json`. Cannot write."""

    def __init__(self, auth_path: str | None = None) -> None:
        self.auth_path = normalize_path(auth_path or _default_auth_path())
        self._data: AuthStorageData | None = None

    def _load(self) -> AuthStorageData:
        if self._data is not None:
            return self._data
        try:
            with open(self.auth_path, encoding="utf-8") as handle:
                raw_text = handle.read()
        except FileNotFoundError:
            self._data = {}
            return self._data
        except OSError as error:
            raise ValueError(f"Failed to read auth.json: {error}") from error

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Failed to read auth.json: {error}") from error
        if not isinstance(parsed, dict):
            raise ValueError("Invalid auth.json: expected an object")
        for provider_id, credential in parsed.items():
            _validate_credential_dict(provider_id, credential)

        self._data = {provider_id: _credential_from_dict(value) for provider_id, value in parsed.items()}
        return self._data

    async def get(self, provider_id: str) -> Credential | None:
        credential = self._load().get(provider_id)
        if credential is None:
            return None
        return _resolve_read_only_api_key_credential(credential)

    async def set(self, provider_id: str, credential: Credential) -> None:
        raise NotImplementedError("Read-only credential storage cannot modify auth.json")

    async def delete(self, provider_id: str) -> None:
        raise NotImplementedError("Read-only credential storage cannot modify auth.json")

    async def list(self) -> list[CredentialInfo]:
        return [CredentialInfo(provider_id=pid, type=c.type) for pid, c in self._load().items()]


class AuthStorage(CredentialStore):
    """Credential storage backed by a JSON file (or an in-memory backend)."""

    def __init__(self, storage: AuthStorageBackend, auth_path: str | None = None) -> None:
        self._storage = storage
        self._auth_path = auth_path
        self._data: AuthStorageData = {}
        self._revision: str | None = None
        self.reload()

    @classmethod
    def create(cls, auth_path: str | None = None) -> AuthStorage:
        normalized = normalize_path(auth_path or _default_auth_path())
        return cls(FileAuthStorageBackend(normalized), normalized)

    @classmethod
    def from_storage(cls, storage: AuthStorageBackend) -> AuthStorage:
        return cls(storage)

    @classmethod
    def in_memory(cls, data: AuthStorageData | None = None) -> AuthStorage:
        storage = InMemoryAuthStorageBackend()
        storage.with_lock(lambda _current: LockResult(result=None, next_content=_serialize_storage_data(data or {})))
        return cls.from_storage(storage)

    def _update_read_state(self, data: AuthStorageData, revision: str | None = None) -> None:
        self._data = data
        self._revision = revision

    def reload(self) -> None:
        """Reload credentials from storage (preserves the last valid snapshot on error)."""
        try:
            content_holder: dict[str, str | None] = {}

            def _read(current: str | None) -> LockResult:
                content_holder["content"] = current
                return LockResult(result=None)

            self._storage.with_lock(_read)
            revision = get_file_revision(self._auth_path) if self._auth_path else None
            self._update_read_state(_parse_storage_data(content_holder.get("content")), revision)
        except Exception:
            pass

    async def _read_latest_data(self) -> AuthStorageData:
        if self._auth_path:
            revision = get_file_revision(self._auth_path)
            if revision is not None and revision == self._revision:
                return self._data

        async def _read(current: str | None) -> LockResult:
            data = _parse_storage_data(current)
            return LockResult(result=data)

        data = await self._storage.with_lock_async(_read)
        revision = get_file_revision(self._auth_path) if self._auth_path else None
        self._update_read_state(data, revision)
        return data

    async def get(self, provider_id: str) -> Credential | None:
        credential = (await self._read_latest_data()).get(provider_id)
        if credential is None:
            return None
        return _resolve_api_key_credential(credential)

    async def set(self, provider_id: str, credential: Credential) -> None:
        latest_data = self._data

        async def _write(current: str | None) -> LockResult:
            nonlocal latest_data
            current_data = _parse_storage_data(current)
            merged = {**current_data, provider_id: credential}
            latest_data = merged
            return LockResult(result=None, next_content=_serialize_storage_data(merged))

        await self._storage.with_lock_async(_write)
        self._update_read_state(latest_data)

    async def delete(self, provider_id: str) -> None:
        latest_data = self._data

        async def _write(current: str | None) -> LockResult:
            nonlocal latest_data
            current_data = _parse_storage_data(current)
            current_data.pop(provider_id, None)
            latest_data = current_data
            return LockResult(result=None, next_content=_serialize_storage_data(current_data))

        await self._storage.with_lock_async(_write)
        self._update_read_state(latest_data)

    async def list(self) -> list[CredentialInfo]:
        data = await self._read_latest_data()
        return [CredentialInfo(provider_id=pid, type=c.type) for pid, c in data.items()]

    def _latest_data_sync(self) -> AuthStorageData:
        if self._auth_path:
            revision = get_file_revision(self._auth_path)
            if revision is None or revision != self._revision:
                self.reload()
        return self._data

    def has_sync(self, provider_id: str) -> bool:
        """Synchronous existence check, used by `ModelRuntime.has_configured_auth`.

        Without it a runtime built over an `AuthStorage` reports a provider as
        unconfigured immediately after `login()` succeeded, because the
        synchronous check can only consult stores exposing this method.
        """
        return provider_id in self._latest_data_sync()

    def get_sync(self, provider_id: str) -> Credential | None:
        """Synchronous read, used by the login/logout selectors on every render."""
        credential = self._latest_data_sync().get(provider_id)
        return _resolve_api_key_credential(credential) if credential is not None else None


def read_stored_credential(provider_id: str, auth_path: str | None = None) -> Credential | None:
    """One-off synchronous read of a stored credential, without a store instance."""
    path = normalize_path(auth_path or _default_auth_path())
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        value = data.get(provider_id)
        return _credential_from_dict(value) if value is not None else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


__all__ = [
    "AuthStorage",
    "AuthStorageBackend",
    "AuthStorageData",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
    "LockResult",
    "ReadOnlyAuthStorage",
    "read_stored_credential",
]
