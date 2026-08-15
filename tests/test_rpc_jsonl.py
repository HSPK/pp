"""Python port of `packages/coding-agent/test/rpc-jsonl.test.ts`.

Pins strict LF-only JSONL framing.
"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.modes.rpc.jsonl import (
    JsonlLineReader,
    iter_json_lines,
    serialize_json_line,
)


def test_serialize_appends_exactly_one_newline():
    line = serialize_json_line({"a": 1})
    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert json.loads(line) == {"a": 1}


def test_serialize_keeps_non_ascii_readable():
    line = serialize_json_line({"text": "你好"})
    assert "你好" in line
    assert json.loads(line)["text"] == "你好"


def test_serialize_does_not_escape_line_separators_into_newlines():
    # U+2028/U+2029 are valid inside JSON strings and must not create a record
    # boundary; the framing is LF-only.
    line = serialize_json_line({"text": "a\u2028b\u2029c"})
    # TS asserts the raw line still contains the literal separators (not \u2028 escapes).
    assert "a\u2028b\u2029c" in line
    assert line.count("\n") == 1
    assert json.loads(line)["text"] == "a\u2028b\u2029c"


def test_reads_complete_lines():
    assert iter_json_lines(['{"a":1}\n{"b":2}\n']) == ['{"a":1}', '{"b":2}']


def test_does_not_split_on_unicode_separators():
    payload = serialize_json_line({"text": "a\u2028b\u2029c"})
    lines = iter_json_lines([payload])
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "a\u2028b\u2029c"


def test_does_not_split_on_a_bare_carriage_return():
    # A CR inside the record (escaped, as JSON requires) is not a boundary.
    lines = iter_json_lines(['{"text":"a\\rb"}\n'])
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "a\rb"


def test_strips_a_trailing_carriage_return_for_crlf_input():
    assert iter_json_lines(['{"a":1}\r\n']) == ['{"a":1}']


def test_handles_crlf_delimited_input_with_multiple_records():
    # Matches the TS "handles CRLF-delimited input" case exactly: two
    # CRLF-terminated records delivered in a single chunk.
    assert iter_json_lines(['{"a":1}\r\n{"b":2}\r\n']) == ['{"a":1}', '{"b":2}']


def test_reassembles_a_line_split_across_chunks():
    assert iter_json_lines(['{"a":', "1}", "\n"]) == ['{"a":1}']


def test_reassembles_a_line_split_byte_by_byte():
    payload = b'{"a":1}\n{"b":2}\n'
    lines: list[str] = []
    reader = JsonlLineReader(lines.append)
    for index in range(len(payload)):
        reader.feed(payload[index : index + 1])
    reader.close()
    assert lines == ['{"a":1}', '{"b":2}']


def test_handles_a_multibyte_character_split_across_chunks():
    payload = serialize_json_line({"text": "你好"}).encode()
    split = payload[:9], payload[9:]
    lines: list[str] = []
    reader = JsonlLineReader(lines.append)
    for chunk in split:
        reader.feed(chunk)
    reader.close()
    assert json.loads(lines[0])["text"] == "你好"


def test_close_flushes_a_trailing_unterminated_line():
    lines: list[str] = []
    reader = JsonlLineReader(lines.append)
    reader.feed('{"a":1}')
    assert lines == []
    reader.close()
    assert lines == ['{"a":1}']


def test_close_is_idempotent_and_emits_once():
    lines: list[str] = []
    reader = JsonlLineReader(lines.append)
    reader.feed('{"a":1}')
    reader.close()
    reader.close()
    assert lines == ['{"a":1}']


def test_feeding_after_close_raises():
    reader = JsonlLineReader(lambda _line: None)
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.feed("x")


def test_empty_lines_are_emitted():
    assert iter_json_lines(["\n\n"]) == ["", ""]


def test_no_input_produces_no_lines():
    assert iter_json_lines([]) == []
    assert iter_json_lines([""]) == []


def test_round_trip_through_serialize_and_read():
    records = [{"a": 1}, {"text": "line\u2028break"}, {"nested": {"x": [1, 2]}}]
    stream = "".join(serialize_json_line(record) for record in records)
    assert [json.loads(line) for line in iter_json_lines([stream])] == records
