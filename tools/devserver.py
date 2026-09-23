#!/usr/bin/env python3
"""
Static dev server for web/ that never caches.

http.server sends Last-Modified with no Cache-Control, so browsers heuristically
cache JS modules and CSS — which means edits silently don't show up. Production
hosting sets its own headers (see vercel.json), so this only affects local work.

    python tools/devserver.py [port]
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        # getUserMedia needs a secure context; localhost already counts as one.
        self.send_header("Permissions-Policy", "camera=(self)")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        if "304" not in fmt % args:
            super().log_message(fmt, *args)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4173
    handler = partial(NoCacheHandler, directory=str(WEB_ROOT))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {WEB_ROOT} at http://localhost:{port} (no-store)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
