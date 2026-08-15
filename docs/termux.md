# Termux (Android) Setup

Pi can run on Android through [Termux](https://termux.dev/) when the Python port's runtime dependencies are available. This port has not been release-smoke-tested on Termux in this repository.

## Prerequisites

1. Install [Termux](https://github.com/termux/termux-app#installation) from GitHub or F-Droid (not Google Play, that version is deprecated)
2. Install [Termux:API](https://github.com/termux/termux-api#installation) from GitHub or F-Droid for clipboard and other device integrations
3. Use Python >=3.11 and uv

## Installation

```bash
# Update packages
pkg update && pkg upgrade

# Install runtime dependencies
pkg install python git termux-api

# Install uv if your Termux repository provides it
pkg install uv

# Install the Python workspace
cd /path/to/pp
uv sync --all-packages

# Run pi
uv run pp
```

If your Termux build does not provide `uv`, install uv by a method supported by that build before running `uv sync`. This port has no npm-based Termux install path.

## Clipboard Support

Clipboard text operations use `termux-clipboard-set` and `termux-clipboard-get` when `TERMUX_VERSION` is set. The Termux:API app and `termux-api` package must be installed for these to work.

Image clipboard is not supported on Termux; the Ctrl+V image paste feature returns no image in the Python port when running under Termux.

## Example AGENTS.md for Termux

Create `~/.pi/agent/AGENTS.md` to help the agent understand the Termux environment:

````markdown
# Agent Environment: Termux on Android

## Location
- **OS**: Android (Termux terminal emulator)
- **Home**: `/data/data/com.termux/files/home`
- **Prefix**: `/data/data/com.termux/files/usr`
- **Shared storage**: `/storage/emulated/0` (Downloads, Documents, etc.)

## Opening URLs
```bash
termux-open-url "https://example.com"
```

## Opening Files
```bash
termux-open file.pdf          # Opens with default app
termux-open --chooser image.jpg      # Choose app
```

## Clipboard
```bash
termux-clipboard-set "text"   # Copy
termux-clipboard-get          # Paste
```

## Notifications
```bash
termux-notification -t "Title" -c "Content"
```

## Device Info
```bash
termux-battery-status         # Battery info
termux-wifi-connectioninfo    # WiFi info
termux-telephony-deviceinfo   # Device info
```

## Sharing
```bash
termux-share -a send file.txt # Share file
```

## Other Useful Commands
```bash
termux-toast "message"        # Quick toast popup
termux-vibrate                # Vibrate device
termux-tts-speak "hello"      # Text to speech
termux-camera-photo out.jpg   # Take photo
```

## Notes
- Termux:API app must be installed for `termux-*` commands
- Use `pkg install termux-api` for the command-line tools
- Storage permission needed for `/storage/emulated/0` access
````

## Limitations

- **No image clipboard**: Termux clipboard API only supports text, and the Python port explicitly skips image reads under Termux.
- **No npm install path**: Install from the Python uv workspace instead.
- **Storage access**: To access files in `/storage/emulated/0` (Downloads, etc.), run `termux-setup-storage` once to grant permissions.

## Troubleshooting

### Clipboard not working

Ensure both apps are installed:
1. Termux (from GitHub or F-Droid)
2. Termux:API (from GitHub or F-Droid)

Then install the CLI tools:
```bash
pkg install termux-api
```

### Permission denied for shared storage

Run once to grant storage permissions:
```bash
termux-setup-storage
```

### uv is unavailable

The Python port requires uv for the documented workspace workflow. If `pkg install uv` is unavailable, use a Termux-supported uv installation method, then rerun `uv sync --all-packages` from the workspace root.
