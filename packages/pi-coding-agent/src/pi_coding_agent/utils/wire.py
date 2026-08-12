"""Serialize Python objects to the TypeScript JSON wire shape.

The JSON stdout protocol (``pi --mode json``) is consumed by tools written
against the TypeScript agent, so the bytes have to match what
``JSON.stringify`` produces there. Two things differ from a naive
``dataclasses.asdict`` + ``json.dumps``:

1. **Key casing.** TS interface fields are camelCase (``stopReason``,
   ``assistantMessageEvent``); the Python port declares them snake_case.
2. **Absent fields.** A TS optional field holds ``undefined`` and
   ``JSON.stringify`` drops the key entirely. Python holds ``None`` and would
   emit ``"parentSession": null``, which is a different document.

Only *dataclass field names* are camelCased. Plain ``dict`` keys are left
alone, because those are data rather than field names — tool-call arguments
and HTTP headers must survive verbatim.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any


def snake_to_camel(name: str) -> str:
    """``assistant_message_event`` -> ``assistantMessageEvent``.

    A trailing underscore is stripped first: the port uses it to dodge Python
    keywords (``continue_`` for TS ``continue``).
    """
    name = name.rstrip("_")
    head, _, rest = name.partition("_")
    if not rest:
        return head
    return head + "".join(part[:1].upper() + part[1:] for part in rest.split("_"))


def to_wire(value: Any) -> Any:
    """Recursively convert `value` to its TypeScript JSON representation."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return to_wire(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name, None)
            if field_value is None:
                continue
            result[snake_to_camel(field.name)] = to_wire(field_value)
        return result
    if isinstance(value, dict):
        return {key: to_wire(item) for key, item in value.items() if item is not None}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_wire(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            snake_to_camel(key): to_wire(item)
            for key, item in vars(value).items()
            if item is not None and not key.startswith("_")
        }
    return repr(value)


__all__ = ["snake_to_camel", "to_wire"]
