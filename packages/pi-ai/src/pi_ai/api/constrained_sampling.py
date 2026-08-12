"""Constrained sampling helpers: JSON-schema strict mode and OpenAI grammar tools.

Python port of `packages/ai/src/api/constrained-sampling.ts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..types import GrammarConstrainedSampling as GrammarConstrainedSamplingConfig
from ..types import Tool


@dataclass
class GrammarConstrainedSampling:
    format: Literal["lark", "regex"]
    definition: str
    input_property: str


@dataclass
class GrammarToolInputJsonBuffer:
    input: str
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: dict[str, Any], input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def _json_string_body(text: str) -> str:
    """`JSON.stringify(text).slice(1, -1)`: the escaped contents without quotes."""
    return json.dumps(text)[1:-1]


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')

    input_delta = next_input[len(buffer.input) :]
    if not close and len(input_delta) == 0:
        return None

    delta = ""
    if not buffer.started:
        delta += f'{{{json.dumps(input_property)}:"'
        buffer.started = True
    delta += _json_string_body(input_delta)
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")

    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")

    input_property = required[0]
    properties = schema.get("properties") or {}
    property_schema = properties.get(input_property)
    if not property_schema:
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if not isinstance(property_schema, dict) or property_schema.get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    config = tool.constrained_sampling
    if not config or config.type != "json_schema":
        return None

    if supports_strict_mode:
        return True
    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool, supports_openai_grammar_tools: bool
) -> GrammarConstrainedSampling | None:
    config = tool.constrained_sampling
    if not config or config.type != "grammar":
        return None

    if not supports_openai_grammar_tools:
        return None

    grammar_config = cast(GrammarConstrainedSamplingConfig, config)
    lark_definition = grammar_config.variants.get("openai_lark")
    regex_definition = grammar_config.variants.get("openai_regex")
    has_lark_definition = isinstance(lark_definition, str) and lark_definition.strip() != ""
    has_regex_definition = isinstance(regex_definition, str) and regex_definition.strip() != ""
    if not has_lark_definition and not has_regex_definition:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.'
        )

    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark_definition else "regex",
            definition=lark_definition if has_lark_definition else regex_definition,  # type: ignore[arg-type]
            input_property=_infer_grammar_input_property(tool),
        )
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error


def create_grammar_tool_input_properties(
    tools: list[Tool] | None, supports_openai_grammar_tools: bool
) -> dict[str, str]:
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties
