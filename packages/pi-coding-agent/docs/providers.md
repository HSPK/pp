# Providers

Pi supports subscription-based providers via OAuth and API key providers via environment variables or `auth.json`. Built-in catalogs ship with the Python port; the current catalog contains 40 providers and 1219 discoverable models.

## Table of Contents

- [Subscriptions](#subscriptions)
- [API Keys](#api-keys)
- [Auth File](#auth-file)
- [Cloud Providers](#cloud-providers)
- [llama.cpp](#llamacpp)
- [Custom Providers](#custom-providers)
- [Resolution Order](#resolution-order)

## Subscriptions

Use `/login` in interactive mode, then select a provider:

- Claude Pro/Max
- GitHub Copilot
- Kimi For Coding
- Mistral
- OpenRouter (OAuth-minted API key billed from OpenRouter credits)
- Radius
- xAI (Grok/X subscription)

Use `/logout` to clear credentials. Tokens are stored in `~/.pi/agent/auth.json` and auto-refresh when expired when the provider supports refresh. OpenRouter instead mints a user-controlled API key that does not expire automatically.

OpenAI Codex models are discoverable, but Codex OAuth and streaming are not ported; using them raises `NotImplementedError`.

### Claude Pro/Max

Anthropic subscription auth is active for Claude Pro/Max accounts. Third-party harness usage draws from [extra usage](https://claude.ai/settings/usage) and is billed per token, not against Claude plan limits.

### GitHub Copilot

- Press Enter for github.com, or enter your GitHub Enterprise Server domain
- If you get "model not supported", enable it in VS Code: Copilot Chat → model selector → select model → "Enable"

### xAI (Grok/X subscription)

- Run `/login xai`, then select **Use a subscription**
- `XAI_API_KEY` remains available through **Use an API key**

### OpenRouter

- Run `/login openrouter`, then select **Sign in with OpenRouter** to open the OpenRouter PKCE authorization flow
- The authorization creates a user-controlled OpenRouter API key billed from your OpenRouter credits
- On remote/headless machines (e.g. over SSH) the browser cannot reach the loopback callback; paste the final redirect URL (or the authorization code) into the login prompt instead
- `OPENROUTER_API_KEY` remains available through **Use an API key**

### Radius

Radius is a dynamic `pi-messages` gateway. `/login radius` stores OAuth tokens in `auth.json`. Custom Radius gateways can be declared in `models.json` with `"oauth": "radius"` and a gateway `baseUrl`.

## API Keys

### Environment Variables or Auth File

Use `/login` in interactive mode and select a provider to store an API key in `auth.json`, or set credentials via environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pp
```

| Provider | Environment Variable | `auth.json` key |
|----------|----------------------|------------------|
| Anthropic | `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_OAUTH_TOKEN`, or `ANTHROPIC_API_KEY` | `anthropic` |
| Ant Ling | `ANT_LING_API_KEY` | `ant-ling` |
| Azure OpenAI Responses | `AZURE_OPENAI_API_KEY` | `azure-openai-responses` |
| OpenAI | `OPENAI_API_KEY` | `openai` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` |
| NVIDIA NIM | `NVIDIA_API_KEY` | `nvidia` |
| Google Gemini | `GEMINI_API_KEY` | `google` |
| Google Vertex AI | `GOOGLE_CLOUD_API_KEY` or Application Default Credentials | `google-vertex` |
| Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` or ambient AWS credentials | `amazon-bedrock` |
| Mistral | `MISTRAL_API_KEY` | `mistral` |
| Groq | `GROQ_API_KEY` | `groq` |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras` |
| Cloudflare AI Gateway | `CLOUDFLARE_API_KEY` (+ `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_GATEWAY_ID`) | `cloudflare-ai-gateway` |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` (+ `CLOUDFLARE_ACCOUNT_ID`) | `cloudflare-workers-ai` |
| xAI | `XAI_API_KEY` | `xai` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` | `vercel-ai-gateway` |
| ZAI Coding Plan (Global) | `ZAI_API_KEY` | `zai` |
| ZAI Coding Plan (China) | `ZAI_CODING_CN_API_KEY` | `zai-coding-cn` |
| OpenCode Zen | `OPENCODE_API_KEY` | `opencode` |
| OpenCode Go | `OPENCODE_API_KEY` | `opencode-go` |
| Radius | `RADIUS_API_KEY` | `radius` |
| Hugging Face | `HF_TOKEN` | `huggingface` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks` |
| Together AI | `TOGETHER_API_KEY` | `together` |
| Baseten | `BASETEN_API_KEY` | `baseten` |
| Kimi For Coding | `KIMI_API_KEY` | `kimi-coding` |
| MiniMax | `MINIMAX_API_KEY` | `minimax` |
| MiniMax (China) | `MINIMAX_CN_API_KEY` | `minimax-cn` |
| Moonshot AI | `MOONSHOT_API_KEY` | `moonshotai` |
| Moonshot AI (China) | `MOONSHOT_API_KEY` | `moonshotai-cn` |
| Qwen Token Plan (existing catalog) | `QWEN_TOKEN_PLAN_API_KEY` | `qwen-token-plan` |
| Qwen Token Plan (Individual) | `QWEN_TOKEN_PLAN_API_KEY` | `qwen-token-plan-individual` |
| Qwen Token Plan (China) | `QWEN_TOKEN_PLAN_CN_API_KEY` | `qwen-token-plan-cn` |
| Xiaomi MiMo | `XIAOMI_API_KEY` | `xiaomi` |
| Xiaomi MiMo Token Plan (China) | `XIAOMI_TOKEN_PLAN_CN_API_KEY` | `xiaomi-token-plan-cn` |
| Xiaomi MiMo Token Plan (Amsterdam) | `XIAOMI_TOKEN_PLAN_AMS_API_KEY` | `xiaomi-token-plan-ams` |
| Xiaomi MiMo Token Plan (Singapore) | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` | `xiaomi-token-plan-sgp` |
| GitHub Copilot token | `COPILOT_GITHUB_TOKEN` | `github-copilot` |

Reference for environment variables: `packages/pi-ai/src/pi_ai/env_api_keys.py`.

#### Auth File

Store credentials in `~/.pi/agent/auth.json`:

```json
{
  "anthropic": { "type": "api_key", "key": "sk-ant-..." },
  "ant-ling": { "type": "api_key", "key": "..." },
  "openai": { "type": "api_key", "key": "sk-..." },
  "deepseek": { "type": "api_key", "key": "sk-..." },
  "nvidia": { "type": "api_key", "key": "nvapi-..." },
  "google": { "type": "api_key", "key": "..." },
  "opencode": { "type": "api_key", "key": "..." },
  "opencode-go": { "type": "api_key", "key": "..." },
  "together": { "type": "api_key", "key": "..." },
  "moonshotai": { "type": "api_key", "key": "..." },
  "qwen-token-plan":  { "type": "api_key", "key": "sk-sp-..." },
  "qwen-token-plan-individual": { "type": "api_key", "key": "sk-sp-..." },
  "qwen-token-plan-cn": { "type": "api_key", "key": "sk-sp-..." },
  "xiaomi": { "type": "api_key", "key": "..." },
  "xiaomi-token-plan-cn":  { "type": "api_key", "key": "..." },
  "xiaomi-token-plan-ams": { "type": "api_key", "key": "..." },
  "xiaomi-token-plan-sgp": { "type": "api_key", "key": "..." }
}
```

`qwen-token-plan-individual` uses the same international endpoint and `QWEN_TOKEN_PLAN_API_KEY` as `qwen-token-plan`, but limits the picker to the models documented for Individual subscriptions. When using `auth.json`, store the credential under the provider you select; an environment variable is shared by both international providers.

The file is created with `0600` permissions (user read/write only). Auth file credentials take priority over environment variables.

API key credentials can also include provider-scoped environment values. These values are used before process environment variables when resolving the credential key, provider/model headers, and provider configuration such as Cloudflare account IDs, Azure OpenAI settings, Vertex project/location, Bedrock settings, `PI_CACHE_RETENTION`, and `HTTP_PROXY`/`HTTPS_PROXY`.

```json
{
  "cloudflare-ai-gateway": {
    "type": "api_key",
    "key": "$CLOUDFLARE_API_KEY",
    "env": {
      "CLOUDFLARE_API_KEY": "...",
      "CLOUDFLARE_ACCOUNT_ID": "account-id",
      "CLOUDFLARE_GATEWAY_ID": "gateway-id"
    }
  }
}
```

Use this when pi should use different provider settings than the project shell environment.

### Key Resolution

The `key` field supports command execution, environment interpolation, and literals:

- **Shell command:** `"!command"` at the start executes the whole value as a command and uses stdout (cached for process lifetime)
  ```json
  { "type": "api_key", "key": "!security find-generic-password -ws 'anthropic'" }
  { "type": "api_key", "key": "!op read 'op://vault/item/credential'" }
  ```
- **Environment interpolation:** `"$ENV_VAR"` or `"${ENV_VAR}"` uses the value of the named variable. Interpolation works inside larger literals.
  ```json
  { "type": "api_key", "key": "$MY_ANTHROPIC_KEY" }
  { "type": "api_key", "key": "${KEY_PREFIX}_${KEY_SUFFIX}" }
  ```
  `$FOO_BAR` is the variable `FOO_BAR`; use `${FOO}_BAR` when `BAR` is literal text. Missing environment variables make the value unresolved.
- **Escapes:** `"$$"` emits a literal `"$"`; `"$!"` emits a literal `"!"` without triggering command execution.
  ```json
  { "type": "api_key", "key": "$$literal-dollar-prefix" }
  { "type": "api_key", "key": "$!literal-bang-prefix" }
  ```
- **Literal value:** Used directly. Plain uppercase strings such as `MY_API_KEY` are literals; use `$MY_API_KEY` for environment variables.
  ```json
  { "type": "api_key", "key": "sk-ant-..." }
  { "type": "api_key", "key": "public" }
  ```

OAuth credentials are also stored here after `/login` and managed automatically.

## Cloud Providers

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_BASE_URL=https://your-resource.ai.azure.com
# also supported: https://your-resource.cognitiveservices.azure.com
# also supported: https://your-resource.openai.azure.com
# root endpoints are auto-normalized to /openai/v1
# or use resource name instead of base URL
export AZURE_OPENAI_RESOURCE_NAME=your-resource

# Optional
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_OPENAI_DEPLOYMENT_NAME_MAP=gpt-4=my-gpt4,gpt-4o=my-gpt4o
```

### Amazon Bedrock

Bedrock models are discoverable, but `bedrock-converse-stream` is not ported; streaming raises `NotImplementedError`.

Use `/login amazon-bedrock` to store a Bedrock API key, or configure one of the ambient AWS credential sources below:

```bash
# Option 1: AWS Profile
export AWS_PROFILE=your-profile

# Option 2: IAM Keys
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...

# Option 3: Bedrock bearer token
export AWS_BEARER_TOKEN_BEDROCK=...

# Optional region
export AWS_REGION=us-west-2
```

Also supports ECS task roles (`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` or `AWS_CONTAINER_CREDENTIALS_FULL_URI`) and IRSA (`AWS_WEB_IDENTITY_TOKEN_FILE`).

```bash
pp --provider amazon-bedrock --model us.anthropic.claude-sonnet-4-20250514-v1:0
```

### Cloudflare AI Gateway

`CLOUDFLARE_API_KEY` can be set via `/login`. The account ID and gateway slug can be set as environment variables or in the API key credential's `env` object in `auth.json`.

```bash
export CLOUDFLARE_API_KEY=...           # or use /login
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_GATEWAY_ID=...        # create at dash.cloudflare.com → AI → AI Gateway
pp --provider cloudflare-ai-gateway --model "claude-sonnet-4-5"
```

Routes to OpenAI, Anthropic, and Workers AI through Cloudflare AI Gateway. Workers AI uses the Unified API (`/compat`) and prefixed model IDs (`workers-ai/@cf/...`). OpenAI uses the OpenAI passthrough route (`/openai`) with native OpenAI model IDs such as `gpt-5.1`. Anthropic uses the Anthropic passthrough route (`/anthropic`) with native Anthropic model IDs such as `claude-sonnet-4-5`.

### Cloudflare Workers AI

`CLOUDFLARE_API_KEY` can be set via `/login`. `CLOUDFLARE_ACCOUNT_ID` can be set as an environment variable or in the API key credential's `env` object in `auth.json`.

```bash
export CLOUDFLARE_API_KEY=...           # or use /login
export CLOUDFLARE_ACCOUNT_ID=...
pp --provider cloudflare-workers-ai --model "@cf/moonshotai/kimi-k2.6"
```

Pi automatically sets `x-session-affinity` for [prefix caching](https://developers.cloudflare.com/workers-ai/features/prompt-caching/) discounts.

### Google Vertex AI

Uses Application Default Credentials:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project  # GCLOUD_PROJECT is also accepted
export GOOGLE_CLOUD_LOCATION=us-central1
```

Or set `GOOGLE_APPLICATION_CREDENTIALS` to a service account key file. `GOOGLE_CLOUD_API_KEY` is also recognized as an API-key credential.

## llama.cpp

The llama.cpp local-model extension is not ported in the Python coding-agent package, so `/llama` is unavailable.

## Custom Providers

**Via models.json:** Add Ollama, LM Studio, vLLM, or any provider that speaks a supported API. The Python port supports these API values in `models.json`: `openai-completions`, `openai-responses`, `anthropic-messages`, and `google-generative-ai`. See [models.md](models.md).

**Via extensions:** For providers that need custom API implementations or OAuth flows, create a Python extension. See [custom-provider.md](custom-provider.md).

## Resolution Order

When resolving credentials for a provider:

1. CLI `--api-key` flag
2. `models.json` provider `apiKey`
3. `auth.json` entry (API key or OAuth token)
4. Environment variable or ambient cloud credentials
5. Custom provider fallback keys from `models.json`
