> pi can help you create pi packages. Ask it to bundle your extensions, skills, prompt templates, or themes.

# Pi Packages

Pi packages bundle extensions, skills, prompt templates, and themes so you can share them through git or local paths. A package can declare resources in `pi.json`, or use conventional directories. The Python package manager can install, update, list, and configure these resources, but package-managed resources are not yet wired into normal agent session startup; use default resource directories or `-e` for runtime extension loading.

## Table of Contents

- [Install and Manage](#install-and-manage)
- [Package Sources](#package-sources)
- [Creating a Pi Package](#creating-a-pi-package)
- [Package Structure](#package-structure)
- [Dependencies](#dependencies)
- [Package Filtering](#package-filtering)
- [Enable and Disable Resources](#enable-and-disable-resources)
- [Scope and Deduplication](#scope-and-deduplication)

## Install and Manage

> **Security:** Pi packages run with full system access. Extensions execute arbitrary Python code, and skills can instruct the model to perform any action including running executables. Review source code before installing third-party packages.

```bash
pp install git:github.com/user/repo@v1
pp install https://github.com/user/repo  # raw URLs work too
pp install /absolute/path/to/package
pp install ./relative/path/to/package

pp remove git:github.com/user/repo
pp list                     # show installed packages from settings
pp update                   # update all configured git packages
pp update git:github.com/user/repo  # update one git package
```

These commands manage pi packages. `pp update` does not update the `pp` CLI itself, refresh model catalogs, or support `--all`, `--extensions`, `--models`, `--self`, `--force`, or `--extension`.

By default, `install` and `remove` write to user settings (`~/.pi/agent/settings.json`). Use `-l` or `--local` to write to project settings (`.pi/settings.json`) instead. Project settings can be shared with your team. Missing project git packages are reconciled by explicit package-manager operations; normal session startup does not install package sources in this port.

To try a local extension without installing it, use `--extension` or `-e`. This loads a Python extension file or directory for the current run only:

```bash
pp -e ./extensions/my_extension.py
pp --no-extensions -e ./extensions
```

`-e` does not install git packages or npm packages in the Python port.

## Package Sources

Pi accepts git and local path sources in settings and `pp install`.

### npm

`npm:` package sources are not supported by this Python port. Use a git or local path source instead.

### git

```
git:github.com/user/repo@v1
git:git@github.com:user/repo@v1
https://github.com/user/repo@v1
ssh://git@github.com/user/repo@v1
```

- Without `git:` prefix, only protocol URLs are accepted (`https://`, `http://`, `ssh://`, `git://`).
- With `git:` prefix, shorthand formats are accepted, including `github.com/user/repo` and `git@github.com:user/repo`.
- HTTPS and SSH URLs are both supported.
- SSH URLs use your configured SSH keys automatically (respects `~/.ssh/config`).
- For non-interactive runs (for example CI), set `GIT_TERMINAL_PROMPT=0` to disable credential prompts and set `GIT_SSH_COMMAND` (for example `ssh -o BatchMode=yes -o ConnectTimeout=5`) to fail fast.
- Refs are pinned tags or commits. `pp update` does not move them to newer refs, but it does reconcile an existing clone to the configured ref.
- Use `pp install git:host/user/repo@new-ref` to update settings and move an existing package to a new pinned ref.
- Cloned to `~/.pi/agent/git/<host>/<path>` (global) or `.pi/git/<host>/<path>` (project).
- When reconciliation changes the checkout, pi resets and cleans the clone. It does not install dependencies inside the clone.

**SSH examples:**
```bash
# git@host:path shorthand (requires git: prefix)
pp install git:git@github.com:user/repo

# ssh:// protocol format
pp install ssh://git@github.com/user/repo

# With version ref
pp install git:git@github.com:user/repo@v1.0.0
```

### Local Paths

```
/absolute/path/to/package
./relative/path/to/package
```

Local paths point to files or directories on disk and are added to settings without copying. Relative paths are resolved against the settings file they appear in. If the path is a file, it loads as a single Python extension. If it is a directory, pi loads resources using package rules.

## Creating a Pi Package

Add a `pi.json` manifest to the package root or use conventional directories.

```json
{
  "extensions": ["./extensions"],
  "skills": ["./skills"],
  "prompts": ["./prompts"],
  "themes": ["./themes"]
}
```

Paths are relative to the package root. Arrays support glob patterns and `!` exclusions.

### Gallery Metadata

The TypeScript package gallery reads npm package metadata. There is no Python package-gallery ingestion for local/git `pi.json` packages.

## Package Structure

### Convention Directories

If no `pi.json` manifest is present, pi auto-discovers resources from these directories:

- `extensions/` loads `.py` files and package directories with `__init__.py`
- `skills/` recursively finds `SKILL.md` folders and loads top-level `.md` files as skills
- `prompts/` loads `.md` files
- `themes/` loads `.json` files

Extension directories can also include their own `pi.json` with an `extensions` array, or expose `__init__.py` as the entry point.

## Dependencies

Third-party runtime dependencies must already be available in the Python environment that runs `pp`, or be vendored in the package. The Python package manager does not run `pip install`, `uv sync`, or any dependency installation step for git packages.

A package can still include helper scripts and instructions for the model to run, but installing a pi package only clones or records the source and resolves resources.

## Package Filtering

Filter what a package loads using the object form in settings:

```json
{
  "packages": [
    "git:github.com/user/simple-pkg",
    {
      "source": "git:github.com/user/my-package",
      "extensions": ["extensions/*.py", "!extensions/legacy.py"],
      "skills": [],
      "prompts": ["prompts/review.md"],
      "themes": ["+themes/legacy.json"]
    }
  ]
}
```

`+path` and `-path` are exact paths relative to the package root.

- Omit a key to load all of that type.
- Use `[]` to load none of that type.
- `!pattern` excludes matches.
- `+path` force-includes an exact path.
- `-path` force-excludes an exact path.
- Filters layer on top of the manifest. They narrow down what is already allowed.

## Enable and Disable Resources

Use `pp config` to enable or disable extensions, skills, prompt templates, and themes from installed packages and local directories. `pp config` starts in global settings (`~/.pi/agent/settings.json`); press Tab to switch between global and project-local modes. Use `pp config -l` to start in project overrides (`.pi/settings.json`) with inherited global resources dimmed. These settings are not yet applied to normal agent session startup in this port.

## Scope and Deduplication

Packages can appear in both global and project settings. If the same package appears in both, the project entry wins unless the project entry has `autoload: false`, in which case it is applied as a delta over the global entry. Identity is determined by:

- git: repository URL without ref
- local: resolved absolute path

npm identity is unavailable because npm package sources are not supported.
