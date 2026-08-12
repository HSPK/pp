"""Resolve configuration values that may be shell commands, environment variables, or literals.

Python port of `packages/coding-agent/src/core/resolve-config-value.ts`. Used
by `provider_composer.py` (mirroring `auth-storage.ts`/`model-registry.ts`'s
use of the TypeScript original).
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VAR_NAME_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")

_command_result_cache: dict[str, str | None] = {}


@dataclass
class _LiteralPart:
    value: str
    type: Literal["literal"] = "literal"


@dataclass
class _EnvPart:
    name: str
    type: Literal["env"] = "env"


_TemplatePart = _LiteralPart | _EnvPart


@dataclass
class _CommandReference:
    config: str
    type: Literal["command"] = "command"


@dataclass
class _TemplateReference:
    parts: list[_TemplatePart] = field(default_factory=list)
    type: Literal["template"] = "template"


_ConfigValueReference = _CommandReference | _TemplateReference


def _append_literal(parts: list[_TemplatePart], value: str) -> None:
    if not value:
        return
    if parts and isinstance(parts[-1], _LiteralPart):
        parts[-1].value += value
        return
    parts.append(_LiteralPart(value=value))


def _parse_config_value_template(config: str) -> list[_TemplatePart]:
    parts: list[_TemplatePart] = []
    index = 0

    while index < len(config):
        dollar_index = config.find("$", index)
        if dollar_index < 0:
            _append_literal(parts, config[index:])
            break

        _append_literal(parts, config[index:dollar_index])
        next_char = config[dollar_index + 1] if dollar_index + 1 < len(config) else ""

        if next_char in ("$", "!"):
            _append_literal(parts, next_char)
            index = dollar_index + 2
            continue

        if next_char == "{":
            end_index = config.find("}", dollar_index + 2)
            if end_index < 0:
                _append_literal(parts, "$")
                index = dollar_index + 1
                continue

            name = config[dollar_index + 2 : end_index]
            if _ENV_VAR_NAME_RE.match(name):
                parts.append(_EnvPart(name=name))
            else:
                _append_literal(parts, config[dollar_index : end_index + 1])
            index = end_index + 1
            continue

        match = _ENV_VAR_NAME_PREFIX_RE.match(config[dollar_index + 1 :])
        if match:
            parts.append(_EnvPart(name=match.group(0)))
            index = dollar_index + 1 + len(match.group(0))
            continue

        _append_literal(parts, "$")
        index = dollar_index + 1

    return parts


def _parse_config_value_reference(config: str) -> _ConfigValueReference:
    if config.startswith("!"):
        return _CommandReference(config=config)
    return _TemplateReference(parts=_parse_config_value_template(config))


def _resolve_env_config_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    if env is not None and env.get(name):
        return env[name]
    return os.environ.get(name) or None


def _get_template_env_var_names(parts: list[_TemplatePart]) -> list[str]:
    names: list[str] = []
    for part in parts:
        if isinstance(part, _EnvPart) and part.name not in names:
            names.append(part.name)
    return names


def _resolve_template(parts: list[_TemplatePart], env: Mapping[str, str] | None = None) -> str | None:
    resolved = ""
    for part in parts:
        if isinstance(part, _LiteralPart):
            resolved += part.value
            continue
        env_value = _resolve_env_config_value(part.name, env)
        if env_value is None:
            return None
        resolved += env_value
    return resolved


def get_config_value_env_var_name(config: str) -> str | None:
    reference = _parse_config_value_reference(config)
    if not isinstance(reference, _TemplateReference):
        return None
    return reference.parts[0].name if len(reference.parts) == 1 and isinstance(reference.parts[0], _EnvPart) else None


def get_config_value_env_var_names(config: str) -> list[str]:
    reference = _parse_config_value_reference(config)
    return _get_template_env_var_names(reference.parts) if isinstance(reference, _TemplateReference) else []


def get_missing_config_value_env_var_names(config: str, env: Mapping[str, str] | None = None) -> list[str]:
    return [name for name in get_config_value_env_var_names(config) if _resolve_env_config_value(name, env) is None]


def is_command_config_value(config: str) -> bool:
    return isinstance(_parse_config_value_reference(config), _CommandReference)


def is_config_value_configured(config: str, env: Mapping[str, str] | None = None) -> bool:
    return len(get_missing_config_value_env_var_names(config, env)) == 0


def _execute_command_uncached(command_config: str) -> str | None:
    command = command_config[1:]
    try:
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _execute_command(command_config: str) -> str | None:
    if command_config in _command_result_cache:
        return _command_result_cache[command_config]
    result = _execute_command_uncached(command_config)
    _command_result_cache[command_config] = result
    return result


def resolve_config_value(config: str, env: Mapping[str, str] | None = None) -> str | None:
    """Resolve a config value (API key, header value, ...).

    - Starting with ``!`` executes the rest as a shell command and uses stdout (cached).
    - Interpolates ``$ENV_VAR``/``${ENV_VAR}`` references.
    - ``$$`` escapes a literal ``$`` and ``$!`` escapes a literal ``!``.
    - Otherwise the value is a literal.
    """
    reference = _parse_config_value_reference(config)
    if isinstance(reference, _CommandReference):
        return _execute_command(reference.config)
    return _resolve_template(reference.parts, env)


def resolve_config_value_uncached(config: str, env: Mapping[str, str] | None = None) -> str | None:
    reference = _parse_config_value_reference(config)
    if isinstance(reference, _CommandReference):
        return _execute_command_uncached(reference.config)
    return _resolve_template(reference.parts, env)


def resolve_config_value_or_throw(config: str, description: str, env: Mapping[str, str] | None = None) -> str:
    resolved_value = resolve_config_value_uncached(config, env)
    if resolved_value is not None:
        return resolved_value

    reference = _parse_config_value_reference(config)
    if isinstance(reference, _CommandReference):
        raise ValueError(f"Failed to resolve {description} from shell command: {reference.config[1:]}")

    missing_env_vars = get_missing_config_value_env_var_names(config, env)
    if len(missing_env_vars) == 1:
        raise ValueError(f"Failed to resolve {description} from environment variable: {missing_env_vars[0]}")
    if len(missing_env_vars) > 1:
        raise ValueError(f"Failed to resolve {description} from environment variables: {', '.join(missing_env_vars)}")

    raise ValueError(f"Failed to resolve {description}")


def resolve_headers(headers: Mapping[str, str] | None, env: Mapping[str, str] | None = None) -> dict[str, str] | None:
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved_value = resolve_config_value(value, env)
        if resolved_value:
            resolved[key] = resolved_value
    return resolved or None


def resolve_headers_or_throw(
    headers: Mapping[str, str] | None, description: str, env: Mapping[str, str] | None = None
) -> dict[str, str] | None:
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved[key] = resolve_config_value_or_throw(value, f'{description} header "{key}"', env)
    return resolved or None


def clear_config_value_cache() -> None:
    """Clear the config value command cache. Exported for testing."""
    _command_result_cache.clear()


__all__ = [
    "clear_config_value_cache",
    "get_config_value_env_var_name",
    "get_config_value_env_var_names",
    "get_missing_config_value_env_var_names",
    "is_command_config_value",
    "is_config_value_configured",
    "resolve_config_value",
    "resolve_config_value_or_throw",
    "resolve_config_value_uncached",
    "resolve_headers",
    "resolve_headers_or_throw",
]
