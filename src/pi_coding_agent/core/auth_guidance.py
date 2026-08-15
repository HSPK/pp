"""Login/auth guidance messages shown when no model or API key is configured.

Port of `packages/coding-agent/src/core/auth-guidance.ts`.

The TypeScript version points users at `getDocsPath()/providers.md` and
`models.md`. `config.py`'s docstring documents that the packaging-path
helpers (`getPackageDir`, `getDocsPath`, ...) were deliberately not ported
(no `node_modules`/Bun-style install layout to introspect for a
`uv`/`pip`-installed package, and the `docs/` directory itself has not been
copied into this port). This module therefore drops the concrete doc paths
and just references "/login" and "/model" -- the guidance text stays useful
without depending on files that don't exist in this port.
"""

from __future__ import annotations

_UNKNOWN_PROVIDER = "unknown"


def get_provider_login_help() -> str:
    return "Use /login to log into a provider via OAuth or API key."


def format_no_models_available_message() -> str:
    return f"No models available. {get_provider_login_help()}"


def format_no_model_selected_message() -> str:
    return f"No model selected.\n\n{get_provider_login_help()}\n\nThen use /model to select a model."


def format_no_api_key_found_message(provider: str) -> str:
    provider_display = "the selected model" if provider == _UNKNOWN_PROVIDER else provider
    return f"No API key found for {provider_display}.\n\n{get_provider_login_help()}"
