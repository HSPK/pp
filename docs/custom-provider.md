# Custom Providers

The Python port does not support extension provider registration (`pi.registerProvider()`, `pi.registerNativeProvider()`, or `pi.unregisterProvider()`). Custom OpenAI-compatible providers are configured with `models.json` in `~/.pi/agent/models.json`.

This enables:

- **Proxies** - Route built-in provider requests through corporate proxies or API gateways.
- **Custom endpoints** - Use self-hosted or private OpenAI-compatible deployments.
- **Configured headers** - Add provider-level and model-level headers.
- **Built-in OAuth providers** - Use the ported built-in OAuth flows. Custom extension OAuth flows are not available.

## Example Extensions

The TypeScript custom-provider extension examples are not ported. In Python, use `models.json` instead.

## Table of Contents

- [Example Extensions](#example-extensions)
- [Quick Reference](#quick-reference)
- [Override Existing Provider](#override-existing-provider)
- [Register New Provider](#register-new-provider)
- [Unregister Provider](#unregister-provider)
- [OAuth Support](#oauth-support)
- [Custom Streaming API](#custom-streaming-api)
- [Context Overflow Errors](#context-overflow-errors)
- [Testing Your Implementation](#testing-your-implementation)
- [Config Reference](#config-reference)
- [Model Definition Reference](#model-definition-reference)

## Quick Reference

Create or edit `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "my-llm": {
      "name": "My LLM",
      "baseUrl": "https://my-llm.example.com/v1",
      "api": "openai-completions",
      "apiKey": "$MY_LLM_API_KEY",
      "headers": {
        "X-Org": "acme"
      },
      "models": [
        {
          "id": "my-model",
          "name": "My Model",
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 128000,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

`models` must be an array. Do not use an object keyed by model id; the Python loader validates `models` as an array and a keyed object is a startup configuration error.

After saving, run:

```bash
uv run pp --list-models my-llm
uv run pp --model my-llm/my-model "Say hello"
```

`ModelRuntime.refresh()` reloads `models.json` for already-running code paths that call it. For normal CLI startup, restart `pp` or reopen the model selector.

## Override Existing Provider

The simplest use case is redirecting an existing provider through a proxy:

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://proxy.example.com/anthropic"
    }
  }
}
```

Add custom headers to a built-in provider:

```json
{
  "providers": {
    "openai": {
      "headers": {
        "X-Custom-Header": "value"
      }
    }
  }
}
```

Use both `baseUrl` and headers:

```json
{
  "providers": {
    "google": {
      "baseUrl": "https://ai-gateway.corp.com/google",
      "headers": {
        "X-Corp-Auth": "$CORP_AUTH_TOKEN"
      }
    }
  }
}
```

When only `baseUrl`, `headers`, `compat`, `modelOverrides`, `apiKey`, `oauth`, or `authHeader` are provided, existing built-in models are preserved and overlaid.

## Register New Provider

To add a new provider, define it under `providers.<id>` and include `models` as an array.

```json
{
  "providers": {
    "local-openai": {
      "name": "Local OpenAI-compatible server",
      "baseUrl": "http://localhost:1234/v1",
      "apiKey": "$LOCAL_OPENAI_API_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "local-model",
          "name": "Local Model",
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 128000,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

A provider-level `api` applies to all models unless a model overrides it. A config-only provider must have enough information to build requests: `baseUrl`, an API implementation, and an authentication method.

`apiKey` and custom header values use the same config value syntax as the TypeScript project: `!command` executes a command for the whole value, `$ENV_VAR` and `${ENV_VAR}` interpolate environment variables, `$$` emits a literal `$`, and `$!` emits a literal `!`.

## Unregister Provider

Dynamic unregistering is not available in the Python port. Remove the provider entry from `~/.pi/agent/models.json` and restart `pp` or trigger a runtime refresh in code that owns `ModelRuntime`.

### API Types

For a new `models.json`-only provider, the Python composer has API modules for:

| API | Use for |
|-----|---------|
| `anthropic-messages` | Anthropic Messages API and compatibles |
| `openai-completions` | OpenAI Chat Completions API and compatibles |
| `openai-responses` | OpenAI Responses API |
| `google-generative-ai` | Google Generative AI API |

Built-in providers may expose additional APIs through their own provider definitions. `bedrock-converse-stream` and `openai-codex-responses` remain discoverable in the catalog, but streaming raises `NotImplementedError` in the Python port.

Most OpenAI-compatible providers should use `openai-completions`. Use model-level `thinkingLevelMap` for model-specific thinking levels, and `compat` for provider quirks:

```json
{
  "providers": {
    "custom-provider": {
      "baseUrl": "https://api.example.com/v1",
      "apiKey": "$CUSTOM_API_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "custom-model",
          "reasoning": true,
          "thinkingLevelMap": {
            "minimal": null,
            "low": null,
            "medium": null,
            "high": "default",
            "xhigh": null,
            "max": "max"
          },
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": true,
            "maxTokensField": "max_tokens",
            "requiresToolResultName": true,
            "thinkingFormat": "qwen",
            "cacheControlFormat": "anthropic"
          },
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 128000,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

Use `thinkingFormat: "openrouter"` for OpenRouter-style `reasoning: { effort }` controls. Use `thinkingFormat: "together"` for Together-style `reasoning: { enabled }` controls. Use `thinkingFormat: "qwen-chat-template"` for local Qwen-compatible servers that read `chat_template_kwargs.enable_thinking` and need `preserve_thinking`.

Use `cacheControlFormat: "anthropic"` for OpenAI-compatible providers that expose Anthropic-style prompt caching via `cache_control` on the system prompt, last tool definition, and last user, assistant, or tool-result text content.

For Anthropic-compatible providers using `api: "anthropic-messages"`, set `compat.forceAdaptiveThinking: true` for upstream models that require adaptive thinking (`thinking.type: "adaptive"` plus `output_config.effort`). Set `compat.allowEmptySignature: true` only for providers that emit empty thinking signatures and expect `signature: ""` on replay.

> Migration note: native Mistral `mistral-conversations` exists in `pi_ai`, but the `models.json` composer does not register that API module for new config-only providers. Use a built-in Mistral provider entry or an OpenAI-compatible endpoint with `openai-completions`.

### Auth Header

If your OpenAI-compatible provider expects an `Authorization` header generated from `apiKey`, set `authHeader: true`:

```json
{
  "providers": {
    "custom-api": {
      "baseUrl": "https://api.example.com/v1",
      "apiKey": "$MY_API_KEY",
      "authHeader": true,
      "api": "openai-completions",
      "models": [
        {
          "id": "custom-model",
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 128000,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

The key is resolved for each auth check/request. Configured headers are resolved when the provider is composed. A request-supplied explicit `Authorization` header takes precedence in request code paths that support request headers.

## OAuth Support

Custom extension OAuth providers are not available. `models.json` supports only the ported built-in OAuth methods and the special Radius gateway marker:

```json
{
  "providers": {
    "corp-radius": {
      "name": "Corporate Radius Gateway",
      "baseUrl": "https://radius.example.com/v1",
      "oauth": "radius"
    }
  }
}
```

OAuth-only providers do not receive a fabricated API-key login method. If a provider has only OAuth auth, `/login` shows OAuth, not an unusable "enter API key" option.

Credentials are persisted in `~/.pi/agent/auth.json`. OAuth credentials use fields equivalent to the Python `Credential` dataclass: `type: "oauth"`, `access`, `refresh`, `expires`, and provider-specific `data`.

## Custom Streaming API

Extension-supplied custom streaming APIs are not ported. The Python runtime streams through `pi_ai.registry.Provider` objects and API modules. Tests or SDK code can inject a native `Provider` into `ModelRuntime.create(providers=[...])`, but the CLI does not load custom stream implementations from extensions.

Reference implementations live in:

- `packages/pi-ai/src/pi_ai/api/anthropic_messages.py`
- `packages/pi-ai/src/pi_ai/api/openai_completions.py`
- `packages/pi-ai/src/pi_ai/api/openai_responses.py`
- `packages/pi-ai/src/pi_ai/api/google_generative_ai.py`
- `packages/pi-ai/src/pi_ai/api/mistral_conversations.py`

### Stream Pattern

Native Python providers return `AssistantMessageEventStream` and push dataclass events:

```python
from pi_ai import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Usage,
    create_assistant_message_event_stream,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


def stream_my_provider(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    stream = create_assistant_message_event_stream()
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        stream.push(StartEvent(partial=output))
        output.content.append(TextContent(text=""))
        content_index = len(output.content) - 1
        stream.push(TextStartEvent(content_index=content_index, partial=output))
        output.content[content_index].text += "hello"
        stream.push(TextDeltaEvent(content_index=content_index, delta="hello", partial=output))
        stream.push(TextEndEvent(content_index=content_index, content="hello", partial=output))
        output.stop_reason = "stop"
        stream.push(DoneEvent(reason="stop", message=output))
        stream.end(output)
    except Exception as error:
        output.stop_reason = "aborted" if options and options.signal and options.signal.aborted else "error"
        output.error_message = str(error)
        stream.push(ErrorEvent(reason=output.stop_reason, error=output))
        stream.end(output)
    return stream
```

### Event Types

Push events in this order:

1. `StartEvent(partial=output)`
2. Content events, repeated as needed:
   - `TextStartEvent(content_index, partial)`
   - `TextDeltaEvent(content_index, delta, partial)`
   - `TextEndEvent(content_index, content, partial)`
   - `ThinkingStartEvent(content_index, partial)`
   - `ThinkingDeltaEvent(content_index, delta, partial)`
   - `ThinkingEndEvent(content_index, content, partial)`
   - `ToolCallStartEvent(content_index, partial)`
   - `ToolCallDeltaEvent(content_index, delta, partial)`
   - `ToolCallEndEvent(content_index, tool_call, partial)`
3. `DoneEvent(reason, message)` or `ErrorEvent(reason, error)`

The `partial` field contains the current `AssistantMessage`. Update `output.content` and `output.usage`, then include `output` in the event.

### Content Blocks

Add content blocks to `output.content` as they arrive:

```python
from pi_ai import AssistantMessage, TextContent, TextDeltaEvent, TextEndEvent, TextStartEvent
from pi_ai.utils.event_stream import AssistantMessageEventStream


def push_text(stream: AssistantMessageEventStream, output: AssistantMessage, delta: str) -> None:
    output.content.append(TextContent(text=""))
    content_index = len(output.content) - 1
    stream.push(TextStartEvent(content_index=content_index, partial=output))
    output.content[content_index].text += delta
    stream.push(TextDeltaEvent(content_index=content_index, delta=delta, partial=output))
    stream.push(TextEndEvent(content_index=content_index, content=delta, partial=output))
```

### Tool Calls

Tool calls use the `ToolCall` content block and tool-call event dataclasses:

```python
import json

from pi_ai import AssistantMessage, ToolCall, ToolCallDeltaEvent, ToolCallEndEvent, ToolCallStartEvent
from pi_ai.utils.event_stream import AssistantMessageEventStream


def push_tool_call(stream: AssistantMessageEventStream, output: AssistantMessage, raw_arguments: str) -> None:
    block = ToolCall(id="call-1", name="read", arguments={})
    output.content.append(block)
    content_index = len(output.content) - 1
    stream.push(ToolCallStartEvent(content_index=content_index, partial=output))
    try:
        block.arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        block.arguments = {}
    stream.push(ToolCallDeltaEvent(content_index=content_index, delta=raw_arguments, partial=output))
    stream.push(ToolCallEndEvent(content_index=content_index, tool_call=block, partial=output))
```

### Usage and Cost

Update usage from the API response and call `calculate_cost()`:

```python
from pi_ai import Model, Usage, calculate_cost


def fill_usage(model: Model, usage: Usage) -> None:
    usage.input = 100
    usage.output = 25
    usage.cache_read = 0
    usage.cache_write = 0
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
    calculate_cost(model, usage)
```

### Context Overflow Errors

When a request exceeds the model context window, pp can recover by compacting and retrying if the finalized assistant message is recognized as an overflow.

Detection runs on the finalized assistant message:

- `message.stop_reason == "error"`
- `message.error_message` matches known overflow patterns in `packages/pi-ai/src/pi_ai/utils/overflow.py`

The generic fallback phrase `context_length_exceeded` is recognized. If a native provider returns a different phrase, normalize it before the message is checked:

```python
import re

from pi_ai import AssistantMessage

MY_PROVIDER_OVERFLOW_PATTERN = re.compile(r"your provider's overflow phrase", re.IGNORECASE)


def normalize_overflow(message: AssistantMessage) -> AssistantMessage:
    if message.stop_reason != "error":
        return message
    error_message = message.error_message or ""
    if "context_length_exceeded" in error_message:
        return message
    if message.provider != "my-provider":
        return message
    if not MY_PROVIDER_OVERFLOW_PATTERN.search(error_message):
        return message
    message.error_message = f"context_length_exceeded: {error_message}"
    return message
```

With this in place, pp can drop the failed assistant message from live context, run compaction, and retry once. Do not rewrite rate-limit or throttling errors as overflow errors.

### Registration

Registering a custom stream function from an extension is not available. For CLI use, configure a supported `api` in `models.json`. For tests or SDK code, construct a `pi_ai.registry.Provider` and pass it to `ModelRuntime.create(providers=[provider])`.

## Testing Your Implementation

For a `models.json` provider:

1. Validate the provider is listed:

   ```bash
   uv run pp --list-models my-llm
   ```

2. Run a small prompt against the model:

   ```bash
   uv run pp --model my-llm/my-model "Say exactly: ok"
   ```

3. Add targeted tests like the existing provider tests:

| Test file | Purpose |
|------|---------|
| `packages/pi-coding-agent/tests/test_agent_session_dynamic_provider.py` | Custom provider discovery from `models.json` |
| `packages/pi-coding-agent/tests/test_model_runtime_modify_models_compat.py` | Outbound request behavior for custom OpenAI-compatible providers |
| `packages/pi-coding-agent/tests/test_model_registry.py` | `models.json` overlay and config-value resolution |
| `packages/pi-coding-agent/tests/test_model_runtime_credential_sync.py` | Credential behavior for `models.json` providers |
| `packages/pi-ai/tests/` provider tests | Built-in API streaming, abort, tokens, overflow, images, and tool calls |

Run targeted tests only, for example:

```bash
uv run pytest packages/pi-coding-agent/tests/test_agent_session_dynamic_provider.py
```

## Config Reference

`models.json` shape:

```json
{
  "providers": {
    "provider-id": {
      "name": "Display name",
      "baseUrl": "https://api.example.com/v1",
      "apiKey": "$PROVIDER_API_KEY",
      "api": "openai-completions",
      "headers": { "X-Header": "value" },
      "authHeader": false,
      "oauth": "radius",
      "compat": {},
      "models": [],
      "modelOverrides": {}
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `name` | Display name for UI such as `/login`. |
| `baseUrl` | API endpoint URL. Required when defining custom models. |
| `apiKey` | API key literal, env interpolation, or `!command`. Required for config-only providers unless OAuth or stored login supplies auth. |
| `api` | API type for streaming. Required at provider or model level when defining custom models. |
| `headers` | Provider-level request headers. Values use the same resolution syntax as `apiKey`. |
| `authHeader` | If true, adds an `Authorization` header from the resolved API key. |
| `oauth` | Only `"radius"` is supported in `models.json`. |
| `compat` | Provider-level compatibility settings merged into models. |
| `models` | Array of model definitions. If provided, entries add or replace models by id. |
| `modelOverrides` | Object keyed by built-in model id for partial overrides. |

## Model Definition Reference

A model definition inside the `models` array supports:

```json
{
  "id": "model-id",
  "name": "Model display name",
  "api": "openai-completions",
  "baseUrl": "https://api.example.com/v1",
  "reasoning": false,
  "thinkingLevelMap": {
    "off": null,
    "minimal": null,
    "low": null,
    "medium": null,
    "high": "default",
    "xhigh": null,
    "max": "max"
  },
  "input": ["text", "image"],
  "cost": {
    "input": 0,
    "output": 0,
    "cacheRead": 0,
    "cacheWrite": 0
  },
  "contextWindow": 128000,
  "maxTokens": 4096,
  "samplingParams": {},
  "headers": {},
  "compat": {}
}
```

| Field | Description |
|-------|-------------|
| `id` | Model ID. Required. |
| `name` | Display name. Defaults to `id`. |
| `api` | API type override for this model. |
| `baseUrl` | Endpoint override for this model. |
| `reasoning` | Whether the model supports extended thinking. |
| `thinkingLevelMap` | Maps pp thinking levels to provider-specific values; `null` marks unsupported levels. |
| `input` | Supported input types: `"text"` and optionally `"image"`. Defaults to `["text"]`. |
| `cost` | Cost per million tokens. Missing values default to `0`. |
| `contextWindow` | Maximum context window in tokens. Defaults to `128000` for custom models. |
| `maxTokens` | Maximum output tokens. Defaults to `16384` for custom models. |
| `samplingParams` | Extra sampling parameters merged into request options. |
| `headers` | Model-specific headers. |
| `compat` | Compatibility flags consumed by the selected API module. |

Common `compat` keys include `supportsDeveloperRole`, `supportsReasoningEffort`, `maxTokensField`, `requiresToolResultName`, `thinkingFormat`, `chatTemplateKwargs`, `chatTemplateArgs`, `cacheControlFormat`, `sessionAffinityFormat`, `sendSessionAffinityHeaders`, `forceAdaptiveThinking`, and `allowEmptySignature`. The Python port treats `compat` as an opaque dictionary and passes keys through to API modules that implement them.
