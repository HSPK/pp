"""Turn ``@file`` CLI arguments into prompt text and image attachments.

Ported from ``packages/coding-agent/src/cli/file-processor.ts``.

``pi @screenshot.png @notes.md "what changed?"`` inlines each file into the
first prompt: text files become ``<file name="...">...</file>`` blocks, images
become real image attachments plus an empty ``<file>`` marker so the model
knows where in the prompt they sit. Empty files are skipped and a missing
file is a hard error, matching the TS ``process.exit(1)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi_ai import ImageContent

from pi_coding_agent.tools.path_utils import resolve_read_path
from pi_coding_agent.utils.image_process import process_image
from pi_coding_agent.utils.mime import detect_supported_image_mime_type_from_file


class FileProcessingError(Exception):
    """A missing or unreadable ``@file`` argument; the CLI prints it and exits 1."""


@dataclass
class ProcessedFiles:
    text: str = ""
    images: list[Any] = field(default_factory=list)


def process_file_arguments(
    file_args: list[str], *, cwd: str | None = None, auto_resize_images: bool = True
) -> ProcessedFiles:
    result = ProcessedFiles()
    base_dir = cwd or os.getcwd()

    for file_arg in file_args:
        absolute_path = os.path.abspath(resolve_read_path(file_arg, base_dir))
        path = Path(absolute_path)

        if not path.exists():
            raise FileProcessingError(f"Error: File not found: {absolute_path}")

        if path.stat().st_size == 0:
            continue

        mime_type = detect_supported_image_mime_type_from_file(absolute_path)

        if mime_type:
            content = path.read_bytes()
            processed = process_image(content, mime_type, auto_resize_images=auto_resize_images)

            if not processed.ok:
                result.text += f'<file name="{absolute_path}">{processed.message}</file>\n'
                continue

            result.images.append(ImageContent(mime_type=processed.mime_type, data=processed.data))

            if processed.hints:
                hints = "\n".join(processed.hints)
                result.text += f'<file name="{absolute_path}">{hints}</file>\n'
            else:
                result.text += f'<file name="{absolute_path}"></file>\n'
        else:
            try:
                content_text = path.read_text(encoding="utf-8")
            except Exception as error:
                raise FileProcessingError(f"Error: Could not read file {absolute_path}: {error}") from error
            result.text += f'<file name="{absolute_path}">\n{content_text}\n</file>\n'

    return result


__all__ = ["FileProcessingError", "ProcessedFiles", "process_file_arguments"]
