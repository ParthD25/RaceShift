#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.openf1 import fetch_session_bundle, find_race_sessions


def main():
    parser = argparse.ArgumentParser(description="Fetch historical OpenF1 race data without an API key.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--event", type=str, help="Substring of event/country/location to select.")
    parser.add_argument("--session-key", type=int)
    parser.add_argument("--output", default=str(ROOT / "data" / "raw" / "openf1"))
    args = parser.parse_args()

    session_key = args.session_key
    label = str(session_key) if session_key else None
    if session_key is None:
        sessions = find_race_sessions(args.year)
        if not sessions:
            raise SystemExit(f"No race sessions found for {args.year}")
        if args.event:
            needle = args.event.lower()
            sessions = [s for s in sessions if needle in " ".join(str(s.get(k, "")) for k in ["country_name", "location", "session_name"]).lower()]
        if not sessions:
            raise SystemExit(f"No matching race session for event={args.event!r}")
        selected = sessions[0]
        session_key = int(selected["session_key"])
        label = f"{args.year}_{selected.get('location','race')}_{session_key}".replace(" ", "_")

    out = Path(args.output) / str(label)
    counts = fetch_session_bundle(session_key, out)
    print(f"Saved OpenF1 session {session_key} to {out}")
    for name, count in counts.items():
        print(f"  {name}: {count:,} rows")


if __name__ == "__main__":
    main()
