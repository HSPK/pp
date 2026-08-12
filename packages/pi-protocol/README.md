# pi-protocol

Runtime-neutral schemas, CBOR encoding, and byte-stream framing for the experimental pi protocol.

Protocol version `1` uses binary messages with this wire layout:

1. A four-byte unsigned big-endian payload length.
2. One definite-length CBOR item containing the message.

The first client message is always `hello`, containing `PROTOCOL_VERSION`. Subsequent messages use correlated request/response envelopes and server event envelopes. Session and server snapshots are authoritative. Progress events are transient UI hints and must not be reduced into authoritative state. Transports complete authentication before protocol bytes are exchanged.

Session lists contain `SessionMetadata`, the normalized durable metadata available without acquiring a session runtime. Only `id` and `createdAt` are required; `updatedAt`, `parentSessionId`, `sessionName`, and `cwd` are included when supported by the backing store. Runtime state such as phase, model, thinking level, attachment, and locking appears only in an acquired `SessionSnapshot`.

This package does not implement JSONL session files. JSONL session persistence belongs to `pi_coding_agent`; `pi_protocol` frames CBOR messages for the socket protocol.

## Validated message API

`encode_client_message()` and `encode_server_message()` validate a message and return a complete framed `bytes` object. The incremental decoders accept arbitrary fragmentation or coalescing, so they work with streams, sockets, and custom byte transports.

```python
from pi_protocol import (
    PROTOCOL_VERSION,
    create_server_message_decoder,
    encode_client_message,
)

hello = {
    "type": "hello",
    "version": PROTOCOL_VERSION,
}

outbound = encode_client_message(hello)

decoder = create_server_message_decoder(max_frame_length=1024 * 1024)
for message in decoder.push(incoming_chunk):
    handle_server_message(message)
decoder.end()
```

`ClientMessageDecoder` and `ServerMessageDecoder` are also available directly. Schema violations, malformed CBOR, and invalid framing throw `ProtocolValidationError`. Validation errors do not retain rejected payloads.

`parse_client_message()` and `parse_server_message()` only validate already-decoded Python values. They do not parse JSON strings.

The TypeScript package builds TypeBox schemas. The Python port expresses those schemas as JSON Schema draft 2020-12 dictionaries in `src/pi_protocol/schemas.py` and validates them with `jsonschema.Draft202012Validator` in `src/pi_protocol/codec.py`. Field names are on-the-wire names and stay camelCase.

## Transport support

Every transport carries the same complete bytes: `[uint32-be CBOR length][CBOR payload]`. Transports may split or coalesce those bytes arbitrarily.

This package does not bundle a transport. Consumers provide a byte-stream transport that preserves byte order and reports stream closure. Custom transports must handle arbitrary frame fragmentation and coalescing.

All transports are untrusted. Configure matching frame limits and enforce access controls appropriate for the transport before exposing a connection to the protocol. Unix sockets can use filesystem permissions, while network transports can authenticate during connection establishment.

## Encoding and framing

`encode_cbor()` and `decode_cbor()` implement the protocol's strict RFC 8949 subset. `encode_frame()` and `FrameDecoder` handle framing independently of schemas and CBOR.

The CBOR subset supports:

- `None` and booleans
- finite numbers, with integers restricted to JavaScript's safe range and non-integers encoded as float64
- UTF-8 strings
- `bytes` byte strings
- definite-length arrays
- definite-length maps represented by `dict` objects with unique string keys

JSON-valued protocol fields reject CBOR byte strings and non-plain objects. Top-level non-values, non-finite or unsafe numbers, malformed UTF-8, tags, indefinite-length items, trailing data, excessive nesting, oversized values, cycles, and unknown object properties are rejected.

Default limits are 16 MiB per CBOR payload/frame, 1,000,000 array elements or map entries, and 64 nested item levels. Options can configure these limits. A frame decoder validates the declared length before buffering payload bytes.

All schemas reject unknown object properties. The protocol is experimental and has no compatibility guarantees.

## Development

From the repository root:

```bash
uv sync --all-packages
uv run pytest packages/pi-protocol
uv run ruff check packages/pi-protocol
```
