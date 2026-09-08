#!/usr/bin/env python
"""Start the local RaceShift API with a clear message when the port is already taken.

    API_PORT=8010 npm run dev:api        # default 8000
    API_HOST=127.0.0.1                   # loopback only by default; RaceShift is local-first

The web dev server reads the same API_PORT so `API_PORT=8010 WEB_PORT=5180 npm run dev`
runs a second checkout next to the first one.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> None:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex((host, port)) == 0:
            sys.stderr.write(
                f"RaceShift API: {host}:{port} is already in use.\n"
                f"Pick another port for both servers, e.g.  API_PORT={port + 10} WEB_PORT=5183 npm run dev\n"
            )
            sys.exit(3)
    import uvicorn

    uvicorn.run("apps.api.main:app", host=host, port=port, reload=os.environ.get("API_RELOAD", "1") == "1")


if __name__ == "__main__":
    main()
