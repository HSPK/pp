# Custom Models

Add custom providers and models (Ollama, vLLM, LM Studio, proxies) via `~/.pi/agent/models.json`.

## Table of Contents

- [Minimal Example](#minimal-example)
- [Full Example](#full-example)
- [Google AI Studio Example](#google-ai-studio-example)
- [Supported APIs](#supported-apis)
- [Provider Configuration](#provider-configuration)
- [Model Configuration](#model-configuration)
- [Overriding Built-in Providers](#overriding-built-in-providers)
- [Per-model Overrides](#per-model-overrides)
- [Anthropic Messages Compatibility](#anthropic-messages-compatibility)
- [OpenAI Compatibility](#openai-compatibility)
- [Programmatic Loading](#programmatic-loading)

## Minimal Example

For local models (Ollama, LM Studio, vLLM), `models` is an array and each model only needs `id`:

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        { "id": "llama3.1:8b" },
        { "id": "qwen2.5-coder:7b" }
      ]
    }
  }
}
```

The `apiKey` value is a placeholder because Ollama ignores it. The Python port still treats models as requiring configured auth before they appear as available, so keyless local servers should keep a dummy value, save a key for that provider with `/login`, or pass `--api-key` when selecting the model.

Some OpenAI-compatible servers do not understand the `developer` role used for reasoning-capable models. For those providers, set `compat.supportsDeveloperRole` to `false` so pi sends the system prompt as a `system` message instead. If the server also does not support `reasoning_effort`, set `compat.supportsReasoningEffort` to `false` too.

You can set `compat` at the provider level to apply to all models, or at the model level to override a specific model. This commonly applies to Ollama, vLLM, SGLang, and similar OpenAI-compatible servers.

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "gpt-oss:20b",
          "reasoning": true
        }
      ]
    }
  }
}
```

## Full Example

Override defaults when you need specific values:

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B (Local)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 128000,
          "maxTokens": 32000,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

The file reloads each time the runtime refreshes model configuration; interactive model selection calls this refresh when opening `/model`. Edit during a session; no restart is normally needed.

## Google AI Studio Example

Use `google-generative-ai` with a `baseUrl` to add models from Google AI Studio:

```json
{
  "providers": {
    "my-google": {
      "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
      "api": "google-generative-ai",
      "apiKey": "$GEMINI_API_KEY",
      "models": [
        {
          "id": "gemma-4-31b-it",
          "name": "Gemma 4 31B",
          "input": ["text", "image"],
          "contextWindow": 262144,
          "reasoning": true
        }
      ]
    }
  }
}
```

The `baseUrl` is required when adding custom models to the `google-generative-ai` API type.

## Supported APIs

`core/provider_composer.py` registers these custom-model API modules:

| API | Description |
|-----|-------------|
| `openai-completions` | OpenAI Chat Completions and compatible servers |
| `openai-responses` | OpenAI Responses API |
| `anthropic-messages` | Anthropic Messages API |
| `google-generative-ai` | Google Generative AI |

Set `api` at provider level as the default for all models, or model level to override a specific model. `azure-openai-responses`, `mistral-conversations`, `google-vertex`, `bedrock-converse-stream`, `openai-codex-responses`, and `pi-messages` may exist in built-in catalogs, but they are not registered for `models.json` custom providers in `provider_composer.py`.

## Provider Configuration

| Field | Description |
|-------|-------------|
| `name` | Optional human-readable provider name |
| `baseUrl` | API endpoint URL |
| `api` | API type (see above) |
| `apiKey` | Optional API key config (see value resolution below). Omit it when auth is provided by `/login`/`auth.json` or CLI `--api-key`. |
| `oauth` | Dynamic OAuth provider type. Currently supports `"radius"`; requires the gateway `baseUrl`. |
| `headers` | Custom headers (see value resolution below) |
| `authHeader` | Set `true` to add an `Authorization` header from the resolved API key |
| `compat` | Provider compatibility overrides. Merged into each model's `compat`. |
| `models` | Array of model configurations. This is an array, not an object keyed by model id. |
| `modelOverrides` | Object keyed by model id for overriding built-in or configured models on this provider |

For providers with `models`, non-built-in provider configs need `baseUrl` and an `api` value at either provider or model level. `apiKey` is not required to load the file: models become available when auth is configured through `/login`/`auth.json`, CLI `--api-key`, or provider `apiKey`. If no auth is configured, the models load but stay unavailable in `/model` and `--list-models`.

Extension provider registration is not ported. `modelOverrides` applies to built-in models and models composed from `models.json`, not extension-registered providers.

### Value Resolution

The `apiKey` and `headers` fields support command execution, environment interpolation, and literals:

- **Shell command:** `"!command"` at the start executes the whole value as a command and uses stdout.
  ```json
  "apiKey": "!security find-generic-password -ws 'anthropic'"
  ```
- **Environment interpolation:** `"$ENV_VAR"` or `"${ENV_VAR}"` uses the value of the named variable. Interpolation works inside larger literals.
  ```json
  "apiKey": "$MY_API_KEY"
  "apiKey": "${KEY_PREFIX}_${KEY_SUFFIX}"
  ```
  `$FOO_BAR` is the variable `FOO_BAR`; use `${FOO}_BAR` when `BAR` is literal text. Missing environment variables make the value unresolved.
- **Escapes:** `"$$"` emits a literal `"$"`; `"$!"` emits a literal `"!"` without triggering command execution.
  ```json
  "apiKey": "$$literal-dollar-prefix"
  "apiKey": "$!literal-bang-prefix"
  ```
- **Literal value:** Used directly. Plain uppercase strings such as `MY_API_KEY` are literals; use `$MY_API_KEY` for environment variables.
  ```json
  "apiKey": "sk-..."
  ```

For request auth, configured API keys and headers are resolved through `resolve_config_value_or_throw()`, so command-backed values are executed at request/auth resolution time rather than cached by `/model` availability checks. `configured_request_auth_status()` reports command-backed values as configured without executing them.

### Custom Headers

```json
{
  "providers": {
    "custom-proxy": {
      "baseUrl": "https://proxy.example.com/v1",
      "apiKey": "$MY_API_KEY",
      "api": "anthropic-messages",
      "headers": {
        "x-portkey-api-key": "$PORTKEY_API_KEY",
        "x-secret": "!op read 'op://vault/item/secret'"
      },
      "models": [
        { "id": "claude-proxy" }
      ]
    }
  }
}
```

Provider-level headers are composed onto the provider. Model-level and override-level headers can be resolved with `resolve_configured_model_headers()`, but the current Python request path does not re-resolve arbitrary per-model headers on every request.

## Model Configuration

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `id` | Yes | — | Model identifier passed to the API |
| `name` | No | `id` | Human-readable model label. Used for matching and secondary display text. |
| `api` | No | provider's `api` | Override provider's API for this model |
| `baseUrl` | No | provider/base model URL | Override endpoint URL for this model |
| `reasoning` | No | `false` | Supports extended thinking |
| `thinkingLevelMap` | No | `{}` | Maps pi thinking levels to provider values and marks unsupported levels |
| `input` | No | `["text"]` | Input types: `["text"]` or `["text", "image"]` |
| `contextWindow` | No | `128000` | Context window size in tokens |
| `maxTokens` | No | `16384` | Maximum output tokens |
| `samplingParams` | No | `{}` | Sampling parameters merged into request options |
| `headers` | No | omitted | Model-specific configured headers |
| `cost` | No | all zeros | Per-million-token rates with optional request-wide input pricing tiers |
| `compat` | No | provider `compat` | Provider compatibility overrides. Merged with provider-level `compat` when both are set. |

A cost tier supplies a complete alternate rate set and applies to the full request when total input usage (`input + cacheRead + cacheWrite`) exceeds `inputTokensAbove`. When multiple tiers match, the highest threshold wins.

```json
{
  "cost": {
    "input": 5,
    "output": 30,
    "cacheRead": 0.5,
    "cacheWrite": 6.25,
    "tiers": [
      {
        "inputTokensAbove": 272000,
        "input": 10,
        "output": 45,
        "cacheRead": 1,
        "cacheWrite": 12.5
      }
    ]
  }
}
```

Current behavior:

- `/model`, `--list-models`, and the interactive footer display entries by model `id`.
- The configured `name` is used for model matching and secondary model detail text. It does not replace the footer/status-bar model id.

### Sampling Parameters

`samplingParams` is a free-form object merged into request options. Model-level parameters are merged with call-level parameters by `pi_ai.api.simple_options.prepare_simple_options()`; call-level parameters win there. The OpenAI-compatible modules also merge request `sampling_params` into the final request body.

```json
{
  "id": "deepseek-v4-flash",
  "samplingParams": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 0,
    "min_p": 0.0
  }
}
```

OpenAI-compatible APIs apply it (`openai-completions`, `openai-responses`, and built-in `azure-openai-responses`). Other APIs ignore it. In `modelOverrides`, `samplingParams` merges per key with the base model's value.

### Thinking Level Map

Use `thinkingLevelMap` on a model to describe model-specific thinking controls. Keys are pi thinking levels: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Maps may contain holes; for example, a model can expose `high` and `max` without exposing `xhigh`.

Values are tristate:

| Value | Meaning |
|-------|---------|
| omitted | Provider default mapping may be used by the API module |
| string | Level is supported and this value is sent to the provider |
| `null` | Level is unsupported and hidden/skipped/clamped away |

Example for a model that only supports off, high, and max reasoning:

```json
{
  "id": "deepseek-v4-pro",
  "reasoning": true,
  "thinkingLevelMap": {
    "minimal": null,
    "low": null,
    "medium": null,
    "high": "high",
    "xhigh": null,
    "max": "max"
  }
}
```

Example for a model where thinking cannot be disabled:

```json
{
  "id": "always-thinking-model",
  "reasoning": true,
  "thinkingLevelMap": {
    "off": null
  }
}
```

## Overriding Built-in Providers

Route a built-in provider through a proxy without redefining models:

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://my-proxy.example.com/v1"
    }
  }
}
```

All built-in Anthropic models remain available. Existing OAuth or API-key auth continues to work if the provider has a base auth method.

To merge custom models into a built-in provider, include the `models` array:

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://my-proxy.example.com/v1",
      "apiKey": "$ANTHROPIC_API_KEY",
      "api": "anthropic-messages",
      "models": [
        { "id": "claude-proxy" }
      ]
    }
  }
}
```

Merge semantics:

- Built-in models are kept.
- Custom models are upserted by `id` within the provider.
- If a custom model `id` matches a built-in model `id`, the custom model replaces that built-in model.
- If a custom model `id` is new, it is added alongside built-in models.

## Per-model Overrides

Use `modelOverrides` to customize built-in models and matching models on the same provider without replacing the provider's full model list.

```json
{
  "providers": {
    "openrouter": {
      "modelOverrides": {
        "anthropic/claude-sonnet-4": {
          "name": "Claude Sonnet 4 (Bedrock Route)",
          "compat": {
            "openRouterRouting": {
              "only": ["amazon-bedrock"]
            }
          }
        }
      }
    }
  }
}
```

`modelOverrides` supports these fields per model: `name`, `reasoning`, `thinkingLevelMap`, `input`, `cost` (partial), `contextWindow`, `maxTokens`, `samplingParams` (merged per key), `headers`, and `compat`.

Behavior notes:

- `modelOverrides` entries are keyed by model id.
- Unknown model IDs are ignored.
- You can combine provider-level `baseUrl`/`headers` with `modelOverrides`.
- Overriding `name` changes model matching and secondary detail text only; the footer and primary model lists continue to show the model `id`.
- If `models` is also defined for a provider, custom models are merged before overrides are applied, so overrides can also match configured models.

## Anthropic Messages Compatibility

For providers or proxies using `api: "anthropic-messages"`, use `compat` to control Anthropic-specific request compatibility.

By default pi sends per-tool `eager_input_streaming: true`. If a proxy or Anthropic-compatible backend rejects that field, set `supportsEagerToolInputStreaming` to `false`. Some Anthropic models require adaptive thinking (`thinking.type: "adaptive"` plus `output_config.effort`) instead of the legacy budget-based thinking payload. Built-in models set this automatically. For custom providers or aliases that route to those models, set `forceAdaptiveThinking` to `true`.

```json
{
  "providers": {
    "anthropic-proxy": {
      "baseUrl": "https://proxy.example.com",
      "api": "anthropic-messages",
      "apiKey": "$ANTHROPIC_PROXY_KEY",
      "compat": {
        "supportsEagerToolInputStreaming": false,
        "supportsLongCacheRetention": true,
        "forceAdaptiveThinking": true,
        "allowEmptySignature": true
      },
      "models": [
        {
          "id": "claude-opus-4-7",
          "reasoning": true,
          "input": ["text", "image"]
        }
      ]
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `supportsEagerToolInputStreaming` | Whether the provider accepts per-tool `eager_input_streaming`. Default: `true`. |
| `supportsLongCacheRetention` | Whether the provider accepts Anthropic long cache retention (`cache_control.ttl: "1h"`) when cache retention is `long`. Default: `true`. |
| `sendSessionAffinityHeaders` | Whether to send session-affinity headers from the session id when caching is enabled. Default: `false`. |
| `supportsCacheControlOnTools` | Whether the provider accepts Anthropic-style `cache_control` markers on tool definitions. Default: `true`. |
| `supportsTemperature` | Whether to send a temperature field. Default: `true`. |
| `forceAdaptiveThinking` | Whether to send adaptive thinking for this model. Default: absent/false. |
| `allowEmptySignature` | Whether to replay empty thinking signatures as `signature: ""` instead of converting thinking to text. Default: `false`. |
| `supportsStrictTools` | Whether the provider accepts strict JSON-schema tool definitions. Default: `false`. |
| `supportsToolReferences` | Whether the provider accepts Anthropic tool-reference blocks. Default: detected for first-party Anthropic models. |

CamelCase and snake_case compat keys are accepted by the Python API modules, but `models.json` examples should use the TypeScript-compatible camelCase keys shown here.

## OpenAI Compatibility

For providers with partial OpenAI compatibility, use the `compat` field.

- Provider-level `compat` applies defaults to all models under that provider.
- Model-level `compat` overrides provider-level values for that model.

```json
{
  "providers": {
    "local-llm": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "compat": {
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        { "id": "local-model" }
      ]
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `supportsStore` | Provider supports the OpenAI `store` field |
| `supportsDeveloperRole` | Use `developer` vs `system` role |
| `supportsReasoningEffort` | Support for `reasoning_effort` parameter |
| `supportsUsageInStreaming` | Supports `stream_options: { include_usage: true }` |
| `supportsFinishReason` | Whether streamed responses include `finish_reason` |
| `maxTokensField` | Use `max_completion_tokens` or `max_tokens` |
| `requiresToolResultName` | Include `name` on tool result messages |
| `requiresAssistantAfterToolResult` | Insert an assistant message before a user message after tool results |
| `requiresThinkingAsText` | Convert thinking blocks to plain text |
| `requiresReasoningContentOnAssistantMessages` | Include empty `reasoning_content` on replayed assistant messages when reasoning is enabled |
| `thinkingFormat` | Use `reasoning_effort`, `openrouter`, `deepseek`, `together`, `baseten`, `zai`, `qwen`, `chat-template`, or `qwen-chat-template` thinking parameters |
| `chatTemplateKwargs` | `chat_template_kwargs` values for `thinkingFormat: "chat-template"` |
| `chatTemplateArgs` | `chat_template_args` values for `thinkingFormat: "baseten"` |
| `zaiToolStream` | Enable z.ai-style tool stream handling |
| `supportsThinkingTokenBudget` | Whether the provider accepts a thinking token budget |
| `cacheControlFormat` | Use Anthropic-style `cache_control` markers; currently only `anthropic` is supported |
| `sendSessionAffinityHeaders` | Send session-affinity headers from the session id when caching is enabled |
| `sessionAffinityFormat` | Session-affinity header format: `openai`, `openai-nosession`, or `openrouter` |
| `supportsStrictMode` | Whether the provider accepts strict JSON-schema function tool definitions |
| `supportsOpenAIGrammarTools` | Whether OpenAI-compatible APIs emit custom Lark/regex grammar tools |
| `deferredToolsMode` | Provider-specific deferred tool serialization; currently `"kimi"` |
| `supportsLongCacheRetention` | Whether the provider accepts long cache retention |
| `openRouterRouting` | OpenRouter provider routing preferences; sent as the request `provider` object |
| `vercelGatewayRouting` | Vercel AI Gateway routing config for provider selection (`only`, `order`) |

`openrouter` uses `reasoning: { effort }`. `together` uses `reasoning: { enabled }` and also `reasoning_effort` when `supportsReasoningEffort` is enabled. `qwen` uses top-level `enable_thinking`. Use `qwen-chat-template` for local Qwen-compatible servers that require `chat_template_kwargs.enable_thinking` and `preserve_thinking`. Use `chat-template` for vLLM/Hugging Face chat templates that need configurable `chat_template_kwargs`. Use `thinkingFormat: "baseten"` with `chatTemplateArgs` for providers that expose toggle controls through `chat_template_args`.

Example:

```json
{
  "providers": {
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "apiKey": "$OPENROUTER_API_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "openrouter/anthropic/claude-3.5-sonnet",
          "name": "OpenRouter Claude 3.5 Sonnet",
          "compat": {
            "openRouterRouting": {
              "allow_fallbacks": true,
              "require_parameters": false,
              "data_collection": "deny",
              "only": ["anthropic", "amazon-bedrock"],
              "ignore": ["gmicloud", "friendli"],
              "sort": { "by": "price", "partition": "model" }
            }
          }
        }
      ]
    }
  }
}
```

Vercel AI Gateway example:

```json
{
  "providers": {
    "vercel-ai-gateway": {
      "baseUrl": "https://ai-gateway.vercel.sh/v1",
      "apiKey": "$AI_GATEWAY_API_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "moonshotai/kimi-k2.5",
          "name": "Kimi K2.5 (Fireworks via Vercel)",
          "reasoning": true,
          "input": ["text", "image"],
          "cost": { "input": 0.6, "output": 3, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 262144,
          "maxTokens": 262144,
          "compat": {
            "vercelGatewayRouting": {
              "only": ["fireworks", "novita"],
              "order": ["fireworks", "novita"]
            }
          }
        }
      ]
    }
  }
}
```

## Programmatic Loading

`ModelConfig` parses `models.json`. `ModelRuntime.create()` composes built-in providers, the `models.json` overlay, and credentials.

```python
import asyncio

from pi_coding_agent.core.model_runtime import ModelRuntime


async def main() -> None:
    runtime = await ModelRuntime.create(models_path="/path/to/models.json")
    runtime.refresh()
    for model in runtime.get_models("ollama"):
        print(model.provider, model.id, model.context_window)


asyncio.run(main())
```

Use the CLI entry point `pp`, not `pi`:

```bash
uv run pp --list-models
uv run pp --model ollama/llama3.1:8b -p "Say exactly: ok"
```
