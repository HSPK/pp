"""Tests for pi_coding_agent.core.auth_storage.

Ported from packages/coding-agent/test/auth-storage.test.ts. Cases relying on
`proper-lockfile`'s stale-lock retry/compromise machinery (TypeScript's
"retries a briefly contended file lock" and "surfaces a compromised file
storage lock") are skipped: this port's `fcntl.flock`-based locking has no
retry/backoff or staleness surface. The `AbortSignal`-cancellation cases are
skipped because no `AuthOperationOptions` cancellation exists in this port's
simplified `CredentialStore` interface (see auth_storage.py's module
docstring). The concurrent-reload-coalescing cases are skipped for the same
documented reason (no shared reload de-duplication in this port).
"""

import asyncio
import fcntl
import json
import os
import time

import pytest
from pi_ai.auth.types import (
    ApiKeyAuth,
    Credential,
    CredentialStore,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.models import ModelsError
from pi_ai.registry import Models, create_provider

from pi_coding_agent.core.auth_storage import (
    AuthStorage,
    FileAuthStorageBackend,
    InMemoryAuthStorageBackend,
    ReadOnlyAuthStorage,
    read_stored_credential,
)


class _FakeApi:
    def stream(self, model, context, options=None, **kwargs):
        raise AssertionError("not used")

    def stream_simple(self, model, context, options=None, **kwargs):
        raise AssertionError("not used")


def _write_auth_json(path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


async def test_reads_and_resolves_stored_api_key_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_AUTH_STORAGE_KEY", "environment-key")
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "$TEST_AUTH_STORAGE_KEY"}})

    storage = AuthStorage.create(auth_path)
    credential = await storage.get("anthropic")

    assert credential.type == "api_key"
    assert credential.key == "environment-key"


async def test_resolves_command_backed_api_key_credentials(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "!printf 'command-key'"}})

    storage = AuthStorage.create(auth_path)
    credential = await storage.get("anthropic")

    assert credential.type == "api_key"
    assert credential.key == "command-key"


async def test_returns_oauth_credentials_unchanged():
    credential = Credential(type="oauth", access="access-token", refresh="refresh-token", expires=9999999999)
    storage = AuthStorage.in_memory({"anthropic": credential})

    result = await storage.get("anthropic")

    assert result == credential


async def test_credential_scoped_env_takes_precedence_and_remains_inspectable(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(
        auth_path,
        {
            "anthropic": {
                "type": "api_key",
                "key": "$SCOPED_KEY",
                "env": {"SCOPED_KEY": "scoped-value", "REGION": "test-region"},
            }
        },
    )

    storage = AuthStorage.create(auth_path)
    credential = await storage.get("anthropic")

    assert credential.key == "scoped-value"
    assert credential.env == {"SCOPED_KEY": "scoped-value", "REGION": "test-region"}


async def test_reads_pick_up_external_writes_across_storage_instances(tmp_path):
    """The portable half of TS's "coalesces file reloads across concurrent readers
    and storage instances". The reload *coalescing* (one shared lock acquisition
    for all in-flight readers) is not ported -- see the module docstring -- but
    every value assertion in that TS case is: after an external write, both an
    already-constructed `AuthStorage` and a second one on the same path must read
    the new values, and `list()` must report both providers in on-disk key order.
    """
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "old"}})
    first = AuthStorage.create(auth_path)
    second = AuthStorage.create(auth_path)
    assert (await first.get("anthropic")).key == "old"

    _write_auth_json(
        auth_path,
        {
            "anthropic": {"type": "api_key", "key": "new"},
            "openai": {"type": "api_key", "key": "openai-key"},
        },
    )

    anthropic, openai, credentials = await asyncio.gather(first.get("anthropic"), second.get("openai"), first.list())
    assert anthropic == Credential(type="api_key", key="new")
    assert openai == Credential(type="api_key", key="openai-key")
    assert [(c.provider_id, c.type) for c in credentials] == [("anthropic", "api_key"), ("openai", "api_key")]
    assert (await second.get("anthropic")).key == "new"

    # A third instance created after yet another external write, plus the two
    # existing ones, all converge on the newest value.
    third = AuthStorage.create(auth_path)
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "newest"}})
    first_reload, third_reload = await asyncio.gather(first.get("anthropic"), third.get("anthropic"))
    assert first_reload == Credential(type="api_key", key="newest")
    assert third_reload == Credential(type="api_key", key="newest")


async def test_set_persists_a_credential_while_preserving_unrelated_external_edits(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "old"}})
    storage = AuthStorage.create(auth_path)

    # Another process/writer touches the file after construction but before `set()`.
    _write_auth_json(
        auth_path,
        {"anthropic": {"type": "api_key", "key": "old"}, "openai": {"type": "api_key", "key": "external"}},
    )

    await storage.set("anthropic", Credential(type="api_key", key="new"))

    with open(auth_path) as f:
        on_disk = json.load(f)
    assert on_disk == {
        "anthropic": {"type": "api_key", "key": "new"},
        "openai": {"type": "api_key", "key": "external"},
    }


async def test_delete_removes_one_credential_while_preserving_others(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(
        auth_path,
        {
            "anthropic": {"type": "api_key", "key": "anthropic-key"},
            "openai": {"type": "api_key", "key": "openai-key"},
        },
    )
    storage = AuthStorage.create(auth_path)
    # External edit adds a third credential before delete() re-reads the file.
    _write_auth_json(
        auth_path,
        {
            "anthropic": {"type": "api_key", "key": "anthropic-key"},
            "openai": {"type": "api_key", "key": "openai-key"},
            "google": {"type": "api_key", "key": "external-key"},
        },
    )

    await storage.delete("anthropic")

    credentials = await storage.list()
    # TS pins the order too: `[{openai}, {google}]`, i.e. the on-disk key order with
    # the deleted entry removed, not a sorted set.
    assert [(c.provider_id, c.type) for c in credentials] == [
        ("openai", "api_key"),
        ("google", "api_key"),
    ]
    assert await storage.get("anthropic") is None
    openai_cred = await storage.get("openai")
    assert openai_cred.key == "openai-key"
    google_cred = await storage.get("google")
    assert google_cred.key == "external-key"


async def test_in_memory_storage_implements_the_same_credential_store_behavior():
    storage = AuthStorage.in_memory({"anthropic": Credential(type="api_key", key="initial")})

    initial = await storage.get("anthropic")
    assert initial.key == "initial"

    await storage.set("anthropic", Credential(type="api_key", key="updated"))
    updated = await storage.get("anthropic")
    assert updated.key == "updated"

    await storage.delete("anthropic")
    assert await storage.list() == []


async def test_does_not_overwrite_malformed_auth_files(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    storage = AuthStorage.create(auth_path)

    with open(auth_path, "w") as f:
        f.write("{invalid-json")

    with pytest.raises(json.JSONDecodeError):
        await storage.set("openai", Credential(type="api_key", key="new"))

    with open(auth_path) as f:
        assert f.read() == "{invalid-json"


# ---------------------------------------------------------------------------
# ReadOnlyAuthStorage: strict validation, no write access.
# ---------------------------------------------------------------------------


async def test_read_only_auth_storage_reads_valid_credentials(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})

    storage = ReadOnlyAuthStorage(auth_path)
    credential = await storage.get("anthropic")

    assert credential.key == "stored"


async def test_read_only_auth_storage_does_not_resolve_command_backed_keys(tmp_path):
    """`ReadOnlyAuthStorage.get()` intentionally does NOT run the shell command
    (unlike `AuthStorage.get()`, see `test_resolves_command_backed_api_key_credentials`),
    matching TypeScript's two distinct `read()` implementations exactly."""
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "!printf 'command-key'"}})

    storage = ReadOnlyAuthStorage(auth_path)
    credential = await storage.get("anthropic")

    assert credential.key == "!printf 'command-key'"


async def test_read_only_auth_storage_rejects_writes(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {})
    storage = ReadOnlyAuthStorage(auth_path)

    with pytest.raises(NotImplementedError):
        await storage.set("anthropic", Credential(type="api_key", key="x"))

    with pytest.raises(NotImplementedError):
        await storage.delete("anthropic")


async def test_read_only_auth_storage_raises_on_malformed_credential_shape(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": 12345}})

    storage = ReadOnlyAuthStorage(auth_path)
    with pytest.raises(ValueError):
        await storage.get("anthropic")


async def test_read_only_auth_storage_treats_missing_file_as_empty(tmp_path):
    auth_path = str(tmp_path / "does-not-exist.json")
    storage = ReadOnlyAuthStorage(auth_path)

    assert await storage.get("anthropic") is None
    assert await storage.list() == []


# ---------------------------------------------------------------------------
# read_stored_credential(): one-off synchronous read.
# ---------------------------------------------------------------------------


def test_read_stored_credential_reads_an_existing_provider(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})

    credential = read_stored_credential("anthropic", auth_path)

    assert credential is not None
    assert credential.key == "stored"


def test_read_stored_credential_returns_none_for_missing_provider_or_file(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})

    assert read_stored_credential("openai", auth_path) is None
    assert read_stored_credential("anthropic", str(tmp_path / "missing.json")) is None


# ---------------------------------------------------------------------------
# Backends directly (FileAuthStorageBackend / InMemoryAuthStorageBackend).
# ---------------------------------------------------------------------------


def test_file_backend_creates_parent_dir_and_file_with_restrictive_permissions(tmp_path):
    auth_path = str(tmp_path / "nested" / "auth.json")
    backend = FileAuthStorageBackend(auth_path)

    from pi_coding_agent.core.auth_storage import LockResult

    result = backend.with_lock(lambda current: LockResult(result=current))

    assert result == "{}"
    assert os.path.isfile(auth_path)
    mode = os.stat(auth_path).st_mode & 0o777
    assert mode == 0o600


async def test_in_memory_backend_round_trips_content():
    from pi_coding_agent.core.auth_storage import LockResult

    backend = InMemoryAuthStorageBackend()

    async def _write(current):
        assert current is None
        return LockResult(result=None, next_content='{"anthropic": {"type": "api_key", "key": "x"}}')

    await backend.with_lock_async(_write)

    async def _read(current):
        return LockResult(result=current)

    content = await backend.with_lock_async(_read)
    assert content == '{"anthropic": {"type": "api_key", "key": "x"}}'


async def test_serializes_concurrent_modifications(tmp_path):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {})
    first = AuthStorage.create(auth_path)
    second = AuthStorage.create(auth_path)

    await asyncio.gather(
        first.set("anthropic", Credential(type="api_key", key="anthropic-key")),
        second.set("openai", Credential(type="api_key", key="openai-key")),
    )

    with open(auth_path) as f:
        assert json.load(f) == {
            "anthropic": {"type": "api_key", "key": "anthropic-key"},
            "openai": {"type": "api_key", "key": "openai-key"},
        }


async def test_serializes_in_memory_mutations_across_providers(tmp_path):
    # TS blocks inside a `modify()` closure and asserts the *second* provider's mutation
    # has not started. This port has no read-modify-write closure, so the equivalent
    # observation point is the callback `with_lock_async` runs while holding the lock.
    started: list[str] = []
    release = asyncio.Event()

    class _BlockingBackend(InMemoryAuthStorageBackend):
        async def with_lock_async(self, fn):
            async def _wrapped(current):
                started.append("call")
                if len(started) == 1:
                    await release.wait()
                return await fn(current)

            return await super().with_lock_async(_wrapped)

    storage = AuthStorage.from_storage(_BlockingBackend())

    first = asyncio.create_task(storage.set("anthropic", Credential(type="api_key", key="anthropic-key")))
    while not started:
        await asyncio.sleep(0)
    second = asyncio.create_task(storage.set("openai", Credential(type="api_key", key="openai-key")))
    for _ in range(10):
        await asyncio.sleep(0)
    assert started == ["call"]

    release.set()
    await asyncio.gather(first, second)
    assert started == ["call", "call"]

    assert await storage.get("anthropic") == Credential(type="api_key", key="anthropic-key")
    assert await storage.get("openai") == Credential(type="api_key", key="openai-key")
    assert [info.provider_id for info in await storage.list()] == ["anthropic", "openai"]


async def test_does_not_write_after_lock_acquisition_failure_and_recovers_on_retry(tmp_path, monkeypatch):
    auth_path = str(tmp_path / "auth.json")
    _write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    storage = AuthStorage.create(auth_path)

    real_flock = fcntl.flock
    calls = {"n": 0}

    def _failing_flock(fd, operation):
        if operation == fcntl.LOCK_EX and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("lock unavailable")
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", _failing_flock)
    with pytest.raises(OSError, match="lock unavailable"):
        await storage.set("openai", Credential(type="api_key", key="new"))

    with open(auth_path) as f:
        assert json.load(f) == {"anthropic": {"type": "api_key", "key": "stored"}}

    monkeypatch.setattr(fcntl, "flock", real_flock)
    await storage.set("openai", Credential(type="api_key", key="new"))

    with open(auth_path) as f:
        assert json.load(f) == {
            "anthropic": {"type": "api_key", "key": "stored"},
            "openai": {"type": "api_key", "key": "new"},
        }


async def test_translates_a_credential_store_refresh_failure_and_allows_a_later_retry():
    provider_id = "oauth-provider"
    base = AuthStorage.in_memory(
        {provider_id: Credential(type="oauth", access="expired-access", refresh="refresh-token", expires=0)}
    )
    fail_next_write = {"value": True}

    class _FlakyStore(CredentialStore):
        async def get(self, pid: str) -> Credential | None:
            return await base.get(pid)

        async def set(self, pid: str, credential: Credential) -> None:
            if fail_next_write["value"]:
                fail_next_write["value"] = False
                raise RuntimeError("credential store unavailable")
            await base.set(pid, credential)

        async def delete(self, pid: str) -> None:
            await base.delete(pid)

        async def list(self):
            return await base.list()

    async def _login(interaction):
        raise AssertionError("not used")

    async def _refresh(credential: Credential, signal=None) -> Credential:
        return Credential(
            type="oauth",
            access="refreshed-access",
            refresh=credential.refresh,
            expires=int(time.time() * 1000) + 60_000,
        )

    async def _to_auth(credential: Credential) -> ResolvedAuth:
        return ResolvedAuth(api_key=credential.access)

    provider = create_provider(
        id=provider_id,
        name="OAuth Provider",
        auth=ProviderAuth(
            api_key=ApiKeyAuth(name=provider_id, env_vars=("OAUTH_PROVIDER_API_KEY",)),
            oauth=OAuthAuth(name="OAuth", login=_login, refresh=_refresh, to_auth=_to_auth),
        ),
        api=_FakeApi(),
        models=[],
        base_url="https://oauth-provider.invalid/v1",
    )
    models = Models(credential_store=_FlakyStore(), env={}.get)
    models.add(provider)

    with pytest.raises(ModelsError) as exc_info:
        await models.get_auth(provider_id)
    assert exc_info.value.code == "auth"

    result = await models.get_auth(provider_id)
    assert result is not None
    assert result.auth.api_key == "refreshed-access"


# --------------------------------------------------------------------------
# Deliberately-omitted surfaces (see the module docstring)
# --------------------------------------------------------------------------


@pytest.mark.skip(
    reason="`coalesces file reloads across concurrent readers and storage instances` and `keeps a "
    "coalesced reload alive while another credential reader is waiting`: TypeScript shares one "
    "`AuthFileReload` (readers counter, abort-on-last-reader) through a process-wide "
    "`sharedAuthFileReadState`, and both cases assert on `lockfile.lock` call counts. "
    "auth_storage.py's module docstring records dropping that de-duplication -- each "
    "`AuthStorage` keeps its own revision-gated snapshot instead, so there is no shared lock "
    "count to assert. Every *value* assertion in the first case is ported in "
    "`test_reads_pick_up_external_writes_across_storage_instances` above."
)
def test_coalesces_file_reloads() -> None:
    pass


@pytest.mark.skip(
    reason="`modify with undefined leaves the current credential unchanged`: this port's "
    "`CredentialStore` is get/set/delete only, with no read-modify-write `modify()` closure to "
    "return `undefined` from (see auth_storage.py's module docstring)."
)
def test_modify_with_undefined_leaves_the_credential_unchanged() -> None:
    pass


@pytest.mark.skip(
    reason="`retries a briefly contended file lock` and `surfaces a compromised file storage "
    "lock`: both pin `proper-lockfile` behavior (ELOCKED retry with jittered backoff, the "
    "`onCompromised` callback). `fcntl.flock` has neither -- it is a blocking advisory lock with "
    "no staleness protocol."
)
def test_file_lock_retry_and_compromise() -> None:
    pass


@pytest.mark.skip(
    reason="The six `AbortSignal` cases (`pre-aborted file operations do not create the backing "
    "file or run the mutation`, `aborts while waiting for a held file lock without running the "
    "mutation later`, `releases a file lock acquired concurrently with cancellation before "
    "mutation`, `holds the file lock until a cancelled active callback settles without committing "
    "it`, `cancels a signalled credential read waiting for a held file lock`, `cancels a queued "
    "in-memory mutation without running it later`) and `preserves the stored credential after "
    "cancelling an active refresh mutation`: `AuthOperationOptions` does not exist in this port's "
    "`CredentialStore`, so no auth operation takes a cancellation signal."
)
def test_cancellable_auth_operations() -> None:
    pass
