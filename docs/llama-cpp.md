# llama.cpp

The TypeScript project includes a llama.cpp router extension and `/llama` command. That extension is not ported to Python.

The Python repository's tests assert that `src/extensions/llama/*` is absent. The Python examples directory also has no llama.cpp extension.

## What is missing

The Python port does not currently provide:

- `/llama`
- `LLAMA_BASE_URL` / `LLAMA_API_KEY` login handling
- Hugging Face search and download from the TUI
- router model load/unload management
- the native `llama.cpp` provider registered by the TypeScript extension

## Using llama.cpp anyway

You can still run `llama-server` yourself and connect through a separately configured OpenAI-compatible provider if your local model configuration supports it. That is not the same as the TypeScript `/llama` workflow and is not managed by pi.

Start a llama.cpp server according to the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Keep `--host 127.0.0.1` for local-only access unless you intentionally expose it.

## Troubleshooting

If `/llama` is missing in the Python port, that is expected. It is a not-yet-ported extension, not a configuration problem.
