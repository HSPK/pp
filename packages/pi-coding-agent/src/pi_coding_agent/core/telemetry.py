"""Install telemetry opt-in resolution.

Ported from ``packages/coding-agent/src/core/telemetry.ts``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings_manager import SettingsManager

_UNSET = object()


def _is_truthy_env_flag(value: str | None) -> bool:
    if not value:
        return False
    return value == "1" or value.lower() == "true" or value.lower() == "yes"


def is_install_telemetry_enabled(
    settings_manager: SettingsManager,
    telemetry_env: str | object | None = _UNSET,
) -> bool:
    """The ``PI_TELEMETRY`` env var, when set at all, overrides the setting."""
    if telemetry_env is _UNSET:
        telemetry_env = os.environ.get("PI_TELEMETRY")
    if telemetry_env is not None:
        return _is_truthy_env_flag(telemetry_env)  # type: ignore[arg-type]
    return settings_manager.get_enable_install_telemetry()
