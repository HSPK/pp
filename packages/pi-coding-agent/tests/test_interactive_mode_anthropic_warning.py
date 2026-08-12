"""Python port of `packages/coding-agent/test/interactive-mode-anthropic-warning.test.ts`.

`_maybe_warn_about_anthropic_subscription_auth` warns at most once, only for
Anthropic models, and only when the `anthropicExtraUsage` warning is enabled.
Called against a stand-in `self`, mirroring the TypeScript prototype call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pi_ai.auth.types import AuthCheck, AuthResult, ResolvedAuth
from pi_ai.registry import Model
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode


class _ModelRuntime:
    """Mirrors the `ModelRuntime` methods under test, including their async-ness.

    `check_auth`/`get_auth` are `async def` on the real runtime and return the
    real `AuthCheck`/`AuthResult`, so this stand-in uses both rather than a
    looser local shape.
    """

    def __init__(self, credential: AuthCheck | None, api_key: str | None = None) -> None:
        self._credential = credential
        self._api_key = api_key
        self.check_auth_calls = 0
        self.get_auth_calls = 0

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        self.check_auth_calls += 1
        return self._credential

    async def get_auth(self, target: str | Model, *, min_oauth_validity_ms: int | None = None) -> AuthResult | None:
        self.get_auth_calls += 1
        if self._api_key is None:
            return None
        return AuthResult(auth=ResolvedAuth(api_key=self._api_key), source="stored credential")


class _SettingsManager:
    def __init__(self, warnings: dict[str, Any]) -> None:
        self._warnings = warnings

    def get_warnings(self) -> dict[str, Any]:
        return self._warnings


@dataclass
class _Session:
    model_runtime: _ModelRuntime
    model: Model | None = None


@dataclass
class _Fake:
    session: _Session
    settings_manager: _SettingsManager
    anthropic_subscription_warning_shown: bool = False
    warnings: list[str] = field(default_factory=list)

    def show_warning(self, message: str) -> None:
        self.warnings.append(message)


def _model(provider: str) -> Model:
    return Model(id="test-model", name="Test", api="anthropic-messages", provider=provider, base_url="https://x")


def _fake(runtime: _ModelRuntime, warnings: dict[str, Any] | None = None) -> _Fake:
    return _Fake(session=_Session(model_runtime=runtime), settings_manager=_SettingsManager(warnings or {}))


async def test_warns_once_when_anthropic_subscription_auth_is_detected() -> None:
    runtime = _ModelRuntime(None, "sk-ant-oat01-test")
    fake = _fake(runtime)

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, _model("anthropic"))
    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, _model("anthropic"))

    assert len(fake.warnings) == 1
    assert runtime.get_auth_calls == 1


async def test_warns_when_anthropic_oauth_is_stored_even_if_token_refresh_would_fail() -> None:
    runtime = _ModelRuntime(AuthCheck(configured=True, type="oauth"))
    fake = _fake(runtime)

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, _model("anthropic"))

    assert len(fake.warnings) == 1
    assert runtime.get_auth_calls == 0


async def test_does_not_warn_for_non_anthropic_models() -> None:
    runtime = _ModelRuntime(None)
    fake = _fake(runtime)

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, _model("openai"))

    assert fake.warnings == []
    assert runtime.get_auth_calls == 0


async def test_does_not_warn_when_anthropic_extra_usage_warning_is_disabled() -> None:
    runtime = _ModelRuntime(None)
    fake = _fake(runtime, {"anthropicExtraUsage": False})

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, _model("anthropic"))

    assert fake.warnings == []
    assert runtime.check_auth_calls == 0
    assert runtime.get_auth_calls == 0
