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


def _free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


def _next_free(host: str, start: int) -> int:
    port = start
    while not _free(host, port):
        port += 1
    return port


def main() -> None:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    if not _free(host, port):
        web_port = int(os.environ.get("WEB_PORT", "5173"))
        api_free = _next_free(host, port + 10)
        web_free = _next_free(host, web_port + 10)
        sys.stderr.write(
            f"RaceShift API: {host}:{port} is already in use.\n"
            f"These two ports are free right now:  API_PORT={api_free} WEB_PORT={web_free} npm run dev\n"
        )
        sys.exit(3)
    import uvicorn

    uvicorn.run("apps.api.main:app", host=host, port=port, reload=os.environ.get("API_RELOAD", "1") == "1")


if __name__ == "__main__":
    main()
