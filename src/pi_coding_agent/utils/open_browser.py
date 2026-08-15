"""Open URLs in the platform browser.

Ported from ``packages/coding-agent/src/utils/open-browser.ts``.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys


def open_browser(target: str) -> None:
    """Open a URL or file in the platform browser/default handler.

    This intentionally never invokes a shell. On Windows, do not use
    ``cmd /c start``: cmd.exe re-parses metacharacters (&, |, ^, ...) before
    ``start`` runs, which would make attacker-controlled URLs injectable.
    """
    if sys.platform == "darwin":
        cmd = ["open", target]
    elif sys.platform == "win32":
        cmd = ["rundll32", "url.dll,FileProtocolHandler", target]
    else:
        cmd = ["xdg-open", target]

    # Launching the browser is best-effort: callers still present the target to
    # the user, so a missing launcher must not become a process crash.
    with contextlib.suppress(OSError):
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
