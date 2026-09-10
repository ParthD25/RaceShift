#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.fastf1_loader import export_session


def main():
    parser = argparse.ArgumentParser(description="Export a FastF1 session to lap-level Parquet.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True, help="Event name or round number")
    parser.add_argument("--session", default="R", help="R, Q, FP1, FP2, FP3, S, SQ")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "raw" / "fastf1"),
        help="Output directory (file named <year>_<event>_<session>.parquet) or an explicit .parquet file path, "
        "e.g. data/imports/abu_dhabi_2025.parquet so the UI lists it",
    )
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache" / "fastf1"))
    args = parser.parse_args()

    event = int(args.event) if args.event.isdigit() else args.event
    target = Path(args.output)
    try:
        if target.suffix.lower() == ".parquet":
            target.parent.mkdir(parents=True, exist_ok=True)
            produced = export_session(args.year, event, args.session, target.parent, args.cache)
            if produced.resolve() != target.resolve():
                produced.replace(target)
            path = target
        else:
            path = export_session(args.year, event, args.session, target, args.cache)
    except Exception as exc:  # FastF1 raises several exception types for unknown or unrun sessions
        raise SystemExit(
            f"Could not load {args.year} {args.event!r} session {args.session}: {type(exc).__name__}: {exc}\n"
            "Check the event name or round number, and that the session has already been run "
            "(future rounds have no timing data yet)."
        ) from exc
    print(path)


if __name__ == "__main__":
    main()
