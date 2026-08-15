# Shell Aliases

Pi runs bash in non-interactive mode (`/bin/bash -c`), which does not expand aliases by default.

To enable your shell aliases, add to `~/.pi/agent/settings.json`:

```json
{
  "shellCommandPrefix": "shopt -s expand_aliases\neval \"$(grep '^alias ' ~/.bashrc)\""
}
```

Adjust the path (`~/.bashrc`, `~/.zshrc`, etc.) to match your shell config.
