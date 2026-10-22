"""Serve the demo locally: the static page from demo/ and the live function at /api/docrooter_demo.

Run from the repository root:  PYTHONPATH=src python demo/serve.py [port]
then open http://localhost:8765. Live uploads need ANTHROPIC_API_KEY in the environment;
the samples work without it.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("docrooter_demo", DEMO / "api" / "docrooter_demo.py")
assert _spec is not None and _spec.loader is not None
_live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_live)
Live = _live.handler


class Demo(SimpleHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/api/docrooter_demo":
            Live.do_POST(self)  # the function's handler, on this request
        else:
            self.send_error(404)

    _reply = Live._reply


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Demo, directory=str(DEMO)))
    print(f"docrooter demo on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
