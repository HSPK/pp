# Containerization

Pi runs with all permissions of the user account that starts it. In some cases, you will want more control over what directories Pi can write to and which credentials it can access.

There are two general options. You can either
1. run the whole `pp` process inside an isolated environment, or
2. run `pp` on the host and route tool execution into an isolated environment.

The Python port currently documents the first pattern. The TypeScript Gondolin tool-routing extension is not ported.

## Choose a pattern

| Pattern | What is isolated | Best for | Notes |
| --- | --- | --- | --- |
| Plain Docker | Whole `pp` process in a local container | Simple local isolation | Provider API keys enter the container. |
| OpenShell | Whole `pp` process in a policy-controlled sandbox | Local or remote managed sandbox | Requires an OpenShell gateway. |
| Gondolin extension | Built-in tools and `!` commands | Local micro-VM isolation while keeping auth on host | Not ported to Python. |

Extensions run wherever the `pp` process runs. If you run host `pp` with a future tool-routing extension, other custom extension tools still run on the host unless they also delegate their operations.

## Gondolin

The TypeScript project has a Gondolin example extension. The Python port does not include `examples/extensions/gondolin/`, and the TypeScript extension cannot be loaded by this Python extension host.

## Plain Docker

Run the whole `pp` process in Docker when you want the simplest local container boundary.

`Dockerfile.pi` at the Python workspace root:

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash ca-certificates git curl \
  && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/pp
COPY . /opt/pp
RUN uv sync --all-packages --locked

WORKDIR /workspace
ENTRYPOINT ["uv", "run", "--project", "/opt/pp", "pp"]
```

Build from the Python workspace root and run from the project you want mounted:

```bash
cd /path/to/pp
docker build -t pi-python-sandbox -f Dockerfile.pi .

cd /path/to/project
docker run --rm -it \
  -e ANTHROPIC_API_KEY \
  -v "$PWD:/workspace" \
  -v pi-agent-home:/root/.pi/agent \
  pi-python-sandbox
```

The `-v "$PWD:/workspace"` mount makes reads and writes in `/workspace` inside Docker directly affect your host files.

Use a named volume for `/root/.pi/agent` if you want container-local settings and sessions. Mounting your host `~/.pi/agent` exposes host auth and session files to the container.

## OpenShell

Use [NVIDIA OpenShell](https://docs.nvidia.com/openshell/about/overview) when you want a policy-controlled sandbox with filesystem, process, network, credential, and inference controls.
OpenShell can run sandboxes through a local gateway backed by Docker, Podman, or a VM runtime, or through a remote Kubernetes gateway.

Every sandbox requires an active gateway.
Register and select one before creating a sandbox:

```bash
openshell gateway add <gateway-url> --name <name>
openshell gateway select <name>
```

Launch `pp` inside an OpenShell sandbox after the Python workspace is available in that sandbox:

```bash
cd /path/to/pp
uv sync --all-packages
openshell sandbox create --name pi-sandbox --from python -- uv run --project /path/to/pp pp
```

In this pattern, the whole `pp` process runs inside the sandbox. Built-in tools, `!` commands, and extension tools execute inside the OpenShell boundary.

If the gateway is remote, project files are not bind-mounted from the host, meaning writes in the sandbox are not reflected on your machine. Clone the repository inside the sandbox or use OpenShell file transfer commands:

```bash
openshell sandbox upload pi-sandbox ./repo /workspace
openshell sandbox download pi-sandbox /workspace/repo ./repo-out
```

OpenShell providers can keep raw model API keys outside the sandbox. When inference routing is configured, code inside the sandbox can call `https://inference.local`, and the gateway injects the configured provider credentials upstream. Configure Pi to use the corresponding OpenAI-compatible or Anthropic-compatible endpoint if you want model traffic to use this route.
