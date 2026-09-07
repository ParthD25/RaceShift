from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.openf1.org/v1"


def _get(endpoint: str, params: dict[str, Any] | None = None, retries: int = 4) -> list[dict[str, Any]]:
    query = urlencode(params or {}, doseq=True)
    url = f"{BASE_URL}/{endpoint}" + (f"?{query}" if query else "")
    request = Request(url, headers={"User-Agent": "RaceShift/0.1 educational-project"})

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_session_bundle(session_key: int, output_dir: str | Path) -> dict[str, int]:
    """Download the OpenF1 endpoints needed for a single historical session."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    endpoints = {
        "laps": {"session_key": session_key},
        "stints": {"session_key": session_key},
        "weather": {"session_key": session_key},
        "race_control": {"session_key": session_key},
        "intervals": {"session_key": session_key},
        "position": {"session_key": session_key},
        "drivers": {"session_key": session_key},
    }

    counts: dict[str, int] = {}
    for name, params in endpoints.items():
        rows = _get(name, params)
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        counts[name] = len(rows)
    return counts


def find_race_sessions(year: int) -> list[dict[str, Any]]:
    return _get("sessions", {"year": year, "session_name": "Race"})
