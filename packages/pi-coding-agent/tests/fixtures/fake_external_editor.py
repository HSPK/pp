"""Fake `$EDITOR` used by `test_external_editor.py`.

Python port of `packages/coding-agent/test/fixtures/fake-external-editor.mjs`.
Records what the editor saw (path, contents, directory listing and mode) into
a capture file, then either fails or rewrites the prompt.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2:
        return 1

    capture_path = argv[0]
    file_path = argv[-1]
    directory = Path(file_path).parent

    Path(capture_path).write_text(
        json.dumps(
            {
                "filePath": file_path,
                "content": Path(file_path).read_text(encoding="utf-8"),
                "entries": sorted(os.listdir(directory)),
                "directoryMode": directory.stat().st_mode & 0o777,
            }
        ),
        encoding="utf-8",
    )

    if "--fail" in argv:
        return 1

    Path(file_path).write_text("" if "--empty" in argv else "edited\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
