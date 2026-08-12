#!/usr/bin/env python3
"""Serve a fake OpenAI-compatible endpoint for manual CLI smoke tests.

Replies with a scripted tool call followed by a final answer, so the CLI can be
exercised end to end without a provider API key:

    python scripts/fake_openai_server.py 8123 &
    PI_API_KEY=x uv run pp --base-url http://127.0.0.1:8123/v1 -m fake "list files"
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TURNS = [
    {
        "id": "r1",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "ls", "arguments": '{"path": "."}'},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    },
    {
        "id": "r2",
        "choices": [{"delta": {"content": "I listed the directory. Done."}, "finish_reason": "stop"}],
    },
]


class Handler(BaseHTTPRequestHandler):
    turn = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)

        chunk = TURNS[min(Handler.turn, len(TURNS) - 1)]
        Handler.turn += 1
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
