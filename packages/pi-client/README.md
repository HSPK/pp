# pi-client

Transport-neutral client for remote pi sessions. `PiClient` exchanges length-prefixed CBOR messages through a small `ByteTransport` protocol. The core package has no platform-specific imports; `pi_client.unix` provides the Unix-domain-socket transport.

```python
from pi_client import ByteTransportHandlers, PiClient, PiClientOptions


class MyTransport:
    async def send(self, chunk: bytes) -> None:
        # Deliver chunks in invocation order and honor backpressure.
        return None

    def close(self) -> None:
        return None


async def transport_factory(handlers: ByteTransportHandlers) -> MyTransport:
    # Connect using WebSocket, Unix socket, or another ordered byte transport.
    # Call handlers.on_data(), handlers.on_close(), or handlers.on_error() for inbound activity.
    return MyTransport()


async def main() -> None:
    client = PiClient(PiClientOptions(transport_factory=transport_factory))
    await client.connect()
    session = await client.create_session(cwd="/workspace")
    unsubscribe = session.subscribe(lambda snapshot: print(snapshot))
    await session.prompt("Inspect this project")
    unsubscribe()
    await session.dispose()
    await client.dispose()
```

Call `handlers.on_data(chunk)` for inbound bytes, `handlers.on_close()` for an orderly terminal close, and `handlers.on_error(error)` for transport failures. A factory must create a fresh transport for every connection attempt and complete any transport-specific authentication before resolving. For example, a WebSocket factory can provide credentials in its upgrade request.

`PiClient` does not reconnect automatically. Call `reconnect()` after disconnection. One connection can attach several sessions. Requests are correlated by ID. Server snapshots and successful response snapshots are authoritative, while progress events do not mutate snapshot state optimistically. Read cached session metadata from `client.snapshot["sessions"]` when a snapshot exists; call `list_sessions()` to request refreshed durable metadata from the server. Runtime state is available after acquiring a session.

`acquire_session()` returns an independent `SessionHandle`; handles cannot be constructed directly for active protocol use. Use `mode="exclusive"` for a lifecycle or mutation coordinator and `mode="shared"` when multiple low-level consumers intentionally share the session. Exclusive acquisition fails with `PiSessionOwnershipError` while any lease exists, and shared acquisition fails while an exclusive lease exists. `attach_session()` is a shared-acquisition convenience method. `create_session()` returns an exclusive handle for the newly created session.

Calling `dispose()` or `detach()` releases only that handle. A handle rejects commands as soon as release begins. The client sends the protocol detach request after the final handle is released. If explicit `detach()` fails, the handle becomes active again for retry. If cleanup-oriented `dispose()` fails, it reports the protocol error but relinquishes local ownership; `PiClient` reconciles the failed protocol cleanup before the next acquisition. A released handle becomes unavailable without affecting other shared handles. Server removal or disconnection invalidates every handle for the affected attachment, and disposing an invalidated handle is a no-op. Commands fail with `PiDisconnectedError` while the client is disconnected and `PiSessionDetachedError` when the client is connected but a handle is releasing, released, or invalidated. `SessionHandle` supports `async with`.

`subscribe()` observes authoritative snapshots. `on_event()` observes protocol events. Both return an unsubscribe function. Structured errors returned by the server are exposed as `PiServerError`.

## Limits and security

`PiClientOptions.max_frame_length` bounds inbound and outbound CBOR payloads. Configure matching limits on the client and server. Transports should separately bound queued outbound bytes and preserve send order.

Treat peers as untrusted. Use a secure transport with appropriate access controls and authenticate during transport establishment.

Subscriber exceptions are isolated from protocol state. Set `on_listener_error` in `PiClientOptions` to report them to application logging or diagnostics.

## Unix-domain sockets

Consumers can use the separately exported Unix-domain socket transport:

```python
from pi_client import PiClient, PiClientOptions, create_unix_transport_factory


async def main() -> None:
    client = PiClient(
        PiClientOptions(
            transport_factory=create_unix_transport_factory(
                ".scratch/pi.sock",
                max_pending_bytes=64 * 1024 * 1024,
            ),
        )
    )

    await client.connect()
```

`max_pending_bytes` bounds queued outbound data. It defaults to four times the protocol frame limit. The transport preserves send order and waits for socket backpressure before resolving each send.

The `pi_client` root remains transport-neutral except for re-exporting the Unix factory for convenience. The implementation also lives at the explicit `pi_client.unix` module path.

## Development

From the repository root:

```bash
uv sync --all-packages
uv run pytest packages/pi-client
uv run ruff check packages/pi-client
```
