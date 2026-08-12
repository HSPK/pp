"""Python port of `packages/coding-agent/test/oauth-selector.test.ts`."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pytest
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthCheck,
    Credential,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.registry import Provider
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.oauth_selector import (
    AuthSelectorProvider,
    AuthType,
    OAuthSelectorComponent,
)
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.keybindings import set_keybindings


@pytest.fixture(autouse=True)
def _theme_and_keybindings() -> None:
    init_theme("dark")
    set_keybindings(KeybindingsManager.create())


async def _login(_interaction: object) -> Credential:
    """`OAuthAuth.login` is an `async def` in production, so the stub is too."""
    return Credential(type="oauth")


async def _refresh(credential: Credential, _signal: object) -> Credential:
    return credential


async def _to_auth(_credential: Credential) -> ResolvedAuth:
    return ResolvedAuth()


def _oauth(name: str) -> OAuthAuth:
    return OAuthAuth(name=name, login=_login, refresh=_refresh, to_auth=_to_auth)


def _provider(provider_id: str, name: str, auth: ProviderAuth) -> Provider:
    return Provider(id=provider_id, name=name, auth=auth, api=None)


@dataclass
class _FakeModelRuntime:
    """Shaped like the slice of `ModelRuntime` that `get_login_provider_options` calls.

    `auth_status` is a real `pi_ai.auth.types.AuthCheck` because that is what
    `ModelRuntime.get_provider_auth_status` actually returns. TypeScript's
    `getProviderAuthStatus` returns `{configured, source, label?}` and the
    caller collapses it with `label ?? source`; this port does that collapse
    inside `get_provider_auth_status` itself (see the `models_json_key` /
    label assertions in test_model_registry.py), so there is no separate
    `label` to prefer downstream. Returning a `label`-carrying stub here would
    make `check.label or check.source` in `get_login_provider_options` look
    correct while crashing against the real runtime.
    """

    providers: list[Provider]
    auth_status: AuthCheck = field(default_factory=lambda: AuthCheck(configured=False))
    using_oauth: bool = False

    def get_providers(self) -> list[Provider]:
        return self.providers

    def get_provider_auth_status(self, _provider_id: str) -> AuthCheck:
        return self.auth_status

    def is_using_oauth(self, _provider_id: str) -> bool:
        return self.using_oauth


class _FakeSession:
    def __init__(self, model_runtime: _FakeModelRuntime) -> None:
        self.model_runtime = model_runtime


class _LoginOptionsContext:
    get_login_provider_options = InteractiveMode.get_login_provider_options

    def __init__(self, runtime: _FakeModelRuntime) -> None:
        self.session = _FakeSession(runtime)


def _render(selector: OAuthSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(120)))


def _selector(provider: AuthSelectorProvider) -> OAuthSelectorComponent:
    return OAuthSelectorComponent(
        "login",
        [provider],
        lambda _provider_id, _auth_type: None,
        lambda: None,
    )


def _status(status_type: AuthType, source: str) -> AuthCheck:
    return AuthCheck(configured=True, type=status_type, source=source)


class TestOAuthSelectorComponent:
    def test_projects_provider_owned_auth_options_without_provider_specific_filtering(self) -> None:
        providers = [
            _provider(
                "anthropic",
                "Anthropic",
                ProviderAuth(
                    api_key=ApiKeyAuth(name="Anthropic API key"),
                    oauth=_oauth("Anthropic (Claude Pro/Max)"),
                ),
            ),
            _provider(
                "google-vertex",
                "Google Vertex AI",
                ProviderAuth(api_key=ApiKeyAuth(name="Google Cloud credentials")),
            ),
        ]
        context = _LoginOptionsContext(_FakeModelRuntime(providers))

        api_key_options = context.get_login_provider_options("api_key")
        assert [(option.id, option.name, option.auth_type, option.method.name) for option in api_key_options] == [
            ("anthropic", "Anthropic", "api_key", "Anthropic API key"),
            ("google-vertex", "Google Vertex AI", "api_key", "Google Cloud credentials"),
        ]

        oauth_options = context.get_login_provider_options("oauth")
        assert [(option.id, option.name, option.auth_type, option.method.name) for option in oauth_options] == [
            ("anthropic", "Anthropic", "oauth", "Anthropic (Claude Pro/Max)")
        ]
        # Both options are unconfigured here, matching `getProviderAuthStatus`
        # returning `{configured: false}` in the TypeScript test.
        assert [option.status for option in api_key_options] == [None, None]

    def test_renders_an_option_without_compiled_auth_status_as_unconfigured(self) -> None:
        selector = _selector(AuthSelectorProvider(id="google", name="Google", auth_type="api_key", status=None))

        output = _render(selector)
        assert "unconfigured" in output
        assert "✓ configured" not in output

    def test_shows_oauth_auth_distinctly_in_the_api_key_selector(self) -> None:
        selector = _selector(
            AuthSelectorProvider(
                id="anthropic", name="Anthropic", auth_type="api_key", status=_status("oauth", "OAuth")
            )
        )

        assert "subscription configured" in _render(selector)

    def test_shows_environment_api_key_auth_as_configured(self) -> None:
        selector = _selector(
            AuthSelectorProvider(
                id="openai", name="OpenAI", auth_type="api_key", status=_status("api_key", "OPENAI_API_KEY")
            )
        )

        output = _render(selector)
        assert "✓ env: OPENAI_API_KEY" in output
        assert "unconfigured" not in output

    def test_shows_models_json_api_key_auth_as_configured(self) -> None:
        selector = _selector(
            AuthSelectorProvider(
                id="local-proxy",
                name="local-proxy",
                auth_type="api_key",
                status=_status("api_key", "key in models.json"),
            )
        )

        assert "✓ key in models.json" in _render(selector)

    def test_shows_models_json_command_auth_as_configured(self) -> None:
        selector = _selector(
            AuthSelectorProvider(
                id="op-proxy",
                name="op-proxy",
                auth_type="api_key",
                status=_status("api_key", "command in models.json"),
            )
        )

        assert "✓ command in models.json" in _render(selector)

    def test_marks_a_configured_oauth_provider_as_configured_not_as_a_mismatched_method(self) -> None:
        # `getLoginProviderOptions` maps the runtime's auth status onto an
        # `AuthCheck` whose `type` comes from `isUsingOAuth`. Passing the raw
        # status object through instead leaves `type` unset, which renders every
        # configured provider as if it were authenticated the *other* way.
        providers = [
            _provider(
                "anthropic",
                "Anthropic",
                ProviderAuth(api_key=ApiKeyAuth(name="Anthropic API key"), oauth=_oauth("Claude Pro/Max")),
            )
        ]
        runtime = _FakeModelRuntime(
            providers,
            auth_status=AuthCheck(configured=True, source="OAuth"),
            using_oauth=True,
        )
        context = _LoginOptionsContext(runtime)

        oauth_option = context.get_login_provider_options("oauth")[0]
        assert oauth_option.status == AuthCheck(configured=True, type="oauth", source="OAuth")
        assert "✓ configured" in _render(_selector(oauth_option))

        api_key_option = context.get_login_provider_options("api_key")[0]
        assert api_key_option.status == AuthCheck(configured=True, type="oauth", source="OAuth")
        assert "subscription configured" in _render(_selector(api_key_option))

    def test_carries_the_auth_status_source_through_to_the_selector(self) -> None:
        # TS asserts `authStatus.label ?? authStatus.source`, because its
        # `getProviderAuthStatus` returns both. This port performs that
        # collapse one level down, inside `ModelRuntime.get_provider_auth_status`
        # -- `test_model_registry.py::test_provider_auth_status_reports_interpolated_
        # environment_variables` pins that the models.json label, not the
        # `models_json_key` source, is what comes back. So the half this test
        # can pin is that whatever `source` the runtime reports reaches the
        # rendered selector unchanged.
        providers = [_provider("openai", "OpenAI", ProviderAuth(api_key=ApiKeyAuth(name="OpenAI API key")))]
        runtime = _FakeModelRuntime(
            providers,
            auth_status=AuthCheck(configured=True, source="key in models.json"),
        )
        context = _LoginOptionsContext(runtime)

        option = context.get_login_provider_options("api_key")[0]
        assert option.status == AuthCheck(configured=True, type="api_key", source="key in models.json")
        assert "✓ key in models.json" in _render(_selector(option))

    def test_the_oauth_login_method_it_hands_to_the_selector_is_awaitable(self) -> None:
        # The selector calls `method.login(interaction)` and awaits it; a stub
        # that is a plain function would make a missing `await` in production
        # look correct, so pin the real shape here.
        providers = [
            _provider(
                "anthropic",
                "Anthropic",
                ProviderAuth(api_key=ApiKeyAuth(name="Anthropic API key"), oauth=_oauth("Claude Pro/Max")),
            )
        ]
        option = _LoginOptionsContext(_FakeModelRuntime(providers)).get_login_provider_options("oauth")[0]

        assert isinstance(option.method, OAuthAuth)
        assert inspect.iscoroutinefunction(option.method.login)
        assert inspect.iscoroutinefunction(option.method.to_auth)
