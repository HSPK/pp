# Windows Setup

The Python port's local bash runner currently spawns `/bin/bash -c`. The TypeScript Git Bash discovery helper is not ported.

For Windows, run pi from WSL or another environment where `/bin/bash` exists:

```bash
cd /path/to/pp
uv sync --all-packages
uv run pp
```

`settings.json` still accepts `shellPath`, but the current Python local runner does not apply it to built-in bash execution. Prefer WSL until Windows shell discovery is ported.
