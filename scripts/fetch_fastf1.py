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
    parser.add_argument("--output", default=str(ROOT / "data" / "raw" / "fastf1"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache" / "fastf1"))
    args = parser.parse_args()

    event = int(args.event) if args.event.isdigit() else args.event
    path = export_session(args.year, event, args.session, args.output, args.cache)
    print(path)


if __name__ == "__main__":
    main()
