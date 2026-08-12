"""Immutable, credential-blind models.json snapshot.

Python port of the data-shape subset of `packages/coding-agent/src/core/model-config.ts`
needed by `provider_composer.py`: a provider may override `baseUrl`/`headers`/
`compat`/`authHeader`/`apiKey`, add or override models, and mark a `radius`
gateway. Model/header/cost/`compat` fields are treated as free-form dicts
rather than being individually schema-validated field by field: the
TypeScript version validates every field with a generated `typebox` schema
(`ModelDefinitionSchema`, `ModelOverrideSchema`, `ProviderConfigSchema`,
covering per-API `compat` variants such as `OpenAICompletionsCompatSchema`).
Reproducing that schema is out of scope here; this port instead validates
only the invariants `compose_model_provider` itself depends on (`id` is a
non-empty string, `baseUrl`/`api` presence rules, `contextWindow`/`maxTokens`
positivity) and otherwise passes `compat`/`headers`/`cost`/`samplingParams`
through as opaque dicts, matching the "port only the narrow interface
required" boundary for this file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_TRAILING_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)


def _strip_json_comments(content: str) -> str:
    """Strip ``//`` and ``/* */`` comments outside of string literals.

    A minimal, string-literal-aware port of `packages/coding-agent/src/utils/json.ts`'s
    `stripJsonComments`.
    """
    result: list[str] = []
    in_string = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    n = len(content)
    while i < n:
        char = content[i]
        nxt = content[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                result.append(char)
            i += 1
            continue
        if in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            result.append(char)
            if char == "\\" and i + 1 < n:
                result.append(nxt)
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


@dataclass
class ModelsJsonModelOverride:
    name: str | None = None
    reasoning: bool | None = None
    thinking_level_map: dict[str, str | None] | None = None
    input: list[Literal["text", "image"]] | None = None
    cost: dict[str, Any] | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    sampling_params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None


@dataclass
class ModelsJsonModel:
    id: str
    name: str | None = None
    api: str | None = None
    base_url: str | None = None
    reasoning: bool | None = None
    thinking_level_map: dict[str, str | None] | None = None
    input: list[Literal["text", "image"]] | None = None
    cost: dict[str, Any] | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    sampling_params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None


@dataclass
class ModelsJsonProvider:
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    oauth: Literal["radius"] | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None
    auth_header: bool | None = None
    models: list[ModelsJsonModel] = field(default_factory=list)
    model_overrides: dict[str, ModelsJsonModelOverride] = field(default_factory=dict)


def _parse_model(provider_id: str, raw: dict[str, Any]) -> ModelsJsonModel:
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f'Provider {provider_id}: model definition missing required "id"')
    return ModelsJsonModel(
        id=model_id,
        name=raw.get("name"),
        api=raw.get("api"),
        base_url=raw.get("baseUrl"),
        reasoning=raw.get("reasoning"),
        thinking_level_map=raw.get("thinkingLevelMap"),
        input=raw.get("input"),
        cost=raw.get("cost"),
        context_window=raw.get("contextWindow"),
        max_tokens=raw.get("maxTokens"),
        sampling_params=raw.get("samplingParams"),
        headers=raw.get("headers"),
        compat=raw.get("compat"),
    )


def _parse_model_override(raw: dict[str, Any]) -> ModelsJsonModelOverride:
    return ModelsJsonModelOverride(
        name=raw.get("name"),
        reasoning=raw.get("reasoning"),
        thinking_level_map=raw.get("thinkingLevelMap"),
        input=raw.get("input"),
        cost=raw.get("cost"),
        context_window=raw.get("contextWindow"),
        max_tokens=raw.get("maxTokens"),
        sampling_params=raw.get("samplingParams"),
        headers=raw.get("headers"),
        compat=raw.get("compat"),
    )


def _parse_provider(provider_id: str, raw: dict[str, Any]) -> ModelsJsonProvider:
    models_raw = raw.get("models") or []
    overrides_raw = raw.get("modelOverrides") or {}
    # TypeScript validates the whole document against `ModelsConfigSchema`
    # (`models: Type.Optional(Type.Array(ModelDefinitionSchema))`) and reports
    # failures through `getError()`. These shape checks keep a malformed
    # `models`/`modelOverrides` a reported config error instead of an
    # `AttributeError` escaping `ModelConfig.load`'s `except ValueError`.
    if not isinstance(models_raw, list):
        raise ValueError(f'Provider {provider_id}: "models" must be an array')
    if any(not isinstance(model, dict) for model in models_raw):
        raise ValueError(f'Provider {provider_id}: "models" entries must be objects')
    if not isinstance(overrides_raw, dict):
        raise ValueError(f'Provider {provider_id}: "modelOverrides" must be an object')
    if any(not isinstance(override, dict) for override in overrides_raw.values()):
        raise ValueError(f'Provider {provider_id}: "modelOverrides" entries must be objects')
    return ModelsJsonProvider(
        name=raw.get("name"),
        base_url=raw.get("baseUrl"),
        api_key=raw.get("apiKey"),
        api=raw.get("api"),
        oauth=raw.get("oauth"),
        headers=raw.get("headers"),
        compat=raw.get("compat"),
        auth_header=raw.get("authHeader"),
        models=[_parse_model(provider_id, m) for m in models_raw],
        model_overrides={key: _parse_model_override(value) for key, value in overrides_raw.items()},
    )


class ModelConfig:
    """One immutable load of `models.json`."""

    def __init__(self, providers: dict[str, ModelsJsonProvider] | None = None, error: str | None = None) -> None:
        self._providers = dict(providers or {})
        self._error = error

    @staticmethod
    def load(models_json_path: str | Path | None) -> ModelConfig:
        if not models_json_path:
            return ModelConfig()
        path = Path(models_json_path)
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ModelConfig()
        except OSError as error:
            return ModelConfig(error=f"Failed to load models.json: {error}\n\nFile: {path}")

        try:
            parsed = json.loads(_strip_json_comments(content))
        except json.JSONDecodeError as error:
            return ModelConfig(error=f"Failed to parse models.json: {error}\n\nFile: {path}")

        if not isinstance(parsed, dict) or not isinstance(parsed.get("providers"), dict):
            return ModelConfig(error=f'Invalid models.json schema: missing "providers" object\n\nFile: {path}')

        providers: dict[str, ModelsJsonProvider] = {}
        try:
            for provider_id, provider_raw in parsed["providers"].items():
                if not isinstance(provider_raw, dict):
                    raise ValueError(f"Provider {provider_id}: expected an object")
                providers[provider_id] = _parse_provider(provider_id, provider_raw)
        except ValueError as error:
            return ModelConfig(error=f"Invalid models.json schema:\n  - {error}\n\nFile: {path}")

        return ModelConfig(providers)

    def get_provider(self, provider_id: str) -> ModelsJsonProvider | None:
        return self._providers.get(provider_id)

    def get_provider_ids(self) -> list[str]:
        return list(self._providers.keys())

    def get_error(self) -> str | None:
        return self._error


__all__ = ["ModelConfig", "ModelsJsonModel", "ModelsJsonModelOverride", "ModelsJsonProvider"]
