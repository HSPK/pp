"""Python port of `packages/coding-agent/test/ansi-utils.test.ts`.

The compatibility case rebuilds chalk's `ansi-regex` pattern here from the same
source strings the TypeScript test uses, so it checks the JS-to-Python regex
translation rather than restating the implementation.
"""

from __future__ import annotations

import re

import pytest

from pi_coding_agent.utils.ansi import strip_ansi


def _reference_ansi_regex() -> re.Pattern[str]:
    st = "(?:\\u0007|\\u001B\\u005C|\\u009C)"
    osc = f"(?:\\u001B\\][\\s\\S]*?{st})"
    csi = "[\\u001B\\u009B][\\[\\]()#;?]*(?:\\d{1,4}(?:[;:]\\d{0,4})*)?[\\dA-PR-TZcf-nq-uy=><~]"
    return re.compile(f"{osc}|{csi}")


_REFERENCE_REGEX = _reference_ansi_regex()


def _reference_strip_ansi(value: str) -> str:
    if "\u001b" not in value and "\u009b" not in value:
        return value
    return _REFERENCE_REGEX.sub("", value)


def _compatibility_inputs() -> list[str]:
    inputs = [
        "plain",
        "a\x1b[31mred\x1b[0mz",
        "a\x1b]8;;https://example.com\x07link\x1b]8;;\x07z",
        "a\x1b]unterminated",
        "a\x1b]funterminated",
        "a\x1bPabc\x1b\\z",
        "a\x1b^abc\x07z",
        "a\x1b_abc\x9cz",
        "a\x90abc\x9cz",
        "a\x9dabc\x9cz",
        "a\x9b31mred",
        "a\x1b(0x",
        "a\x1b*0x",
        "a\x1b+c",
        "a\x1b/0x",
        "a\x1bcok",
        "a\x1b\\ok",
    ]
    chars = [
        "a",
        "f",
        "0",
        "1",
        ";",
        ":",
        "[",
        "]",
        "(",
        ")",
        "#",
        "?",
        "m",
        "P",
        "_",
        "\\",
        "\x07",
        "\x1b",
        "\x9b",
        "\x9c",
        "\x90",
        "\x9d",
    ]

    for char in chars:
        inputs.append(f"x\x1b{char}y")
        inputs.append(f"x\x9b{char}y")
        for index in range(0, len(chars), 3):
            inputs.append(f"x\x1b{char}{chars[index]}y")

    return inputs


def test_matches_chalk_strip_ansi_for_generated_compatibility_inputs() -> None:
    for value in _compatibility_inputs():
        assert strip_ansi(value) == _reference_strip_ansi(value)


def test_raises_type_error_for_non_string_values() -> None:
    # TypeScript asserts both the exception type and the exact message
    # ``Expected a `string`, got `${typeof value}` `` for every value. The port
    # substitutes Python's type name for JS's `typeof`, so the message is pinned
    # per value with that one substitution and nothing else relaxed.
    #
    # TS also passes `Object("x")` (a boxed String, `typeof` "object"). Python
    # has no boxed str: a `str` subclass passes `isinstance`, which is the
    # correct behaviour here, so that value has no counterpart.
    cases: list[tuple[object, str]] = [
        (None, "NoneType"),
        (123, "int"),
        ({}, "dict"),
        (object(), "object"),
        ([], "list"),
        (b"x", "bytes"),
    ]
    for value, type_name in cases:
        with pytest.raises(TypeError) as excinfo:
            strip_ansi(value)  # type: ignore[arg-type]
        assert str(excinfo.value) == f"Expected a `str`, got `{type_name}`"


def test_strips_ris_without_leaking_the_final_byte() -> None:
    assert strip_ansi("\x1bcdone") == "done"


def test_strips_single_byte_esc_sequences_without_leaking_final_bytes() -> None:
    for code in range(ord("g"), ord("m") + 1):
        assert strip_ansi(f"\x1b{chr(code)}ok") == "ok"
    for code in range(ord("r"), ord("t") + 1):
        assert strip_ansi(f"\x1b{chr(code)}ok") == "ok"


def test_strips_common_ansi_sequences_used_in_tool_output() -> None:
    value = "a\x1b[31mred\x1b[0m\x1b]8;;https://example.com\x07link\x1b]8;;\x07z"

    assert strip_ansi(value) == "aredlinkz"
