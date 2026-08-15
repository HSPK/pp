"""Python port of `packages/coding-agent/test/suite/regressions/7572-provider-retry-settings-merge.test.ts`."""

from __future__ import annotations

import json

from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager


def test_preserves_global_provider_settings_not_overridden_by_the_project() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global", lambda _current: json.dumps({"retry": {"provider": {"timeoutMs": 30000, "maxRetryDelayMs": 45000}}})
    )
    storage.with_lock("project", lambda _current: json.dumps({"retry": {"provider": {"maxRetries": 2}}}))

    settings_manager = SettingsManager.from_storage(storage)

    assert settings_manager.get_provider_retry_settings() == {
        "timeoutMs": 30000,
        "maxRetries": 2,
        "maxRetryDelayMs": 45000,
    }
