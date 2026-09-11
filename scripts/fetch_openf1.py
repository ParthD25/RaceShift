#!/usr/bin/env python
"""Fetch race and sprint lap timing from OpenF1 (2023 onward), one parquet per session.

Resumable: sessions whose parquet already exists are skipped, and every API response is
cached under ``--cache`` so a re-run after a failure downloads nothing twice.

Examples:
    python scripts/fetch_openf1.py --years 2025-2026 --sessions R --output data/raw/openf1
    python scripts/fetch_openf1.py --years 2023-2026 --sessions S --output data/raw/openf1 \
        --combine data/imports/f1_sprints_openf1.parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from raceshift.data.openf1_loader import OpenF1Client, export_session, meeting_sessions, season_meetings  # noqa: E402
from fetch_fastf1_seasons import parse_years  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch F1 lap timing from OpenF1 into RaceShift lap tables.")
    p.add_argument("--years", default="2025")
    p.add_argument("--sessions", nargs="+", default=["R"], help="Session codes: R, S (sprint), Q, SQ, SS, FP1-3")
    p.add_argument("--output", default=str(ROOT / "data" / "raw" / "openf1"))
    p.add_argument("--cache", default=str(ROOT / "data" / "cache" / "openf1"))
    p.add_argument("--combine", help="Also write every fetched session of this run into one parquet")
    p.add_argument("--min-interval", type=float, default=0.4, help="Seconds between requests")
    args = p.parse_args()

    client = OpenF1Client(cache_dir=args.cache, min_interval_s=args.min_interval)
    wanted = set(args.sessions)
    output = Path(args.output)
    produced: list[Path] = []
    failures: list[tuple[int, str, str]] = []
    skipped = 0
    for year in parse_years(args.years):
        for meeting in season_meetings(client, year):
            if str(meeting.get("date_start", ""))[:10] > date.today().isoformat():
                continue
            for session in meeting_sessions(client, int(meeting["meeting_key"]), wanted):
                if str(session.get("date_end", ""))[:10] >= date.today().isoformat():
                    continue
                expected = output / f"{year}_{str(meeting['meeting_name']).replace(' ', '_')}_{session['session_code']}.parquet"
                if expected.exists():
                    skipped += 1
                    produced.append(expected)
                    continue
                try:
                    path = export_session(client, meeting, session, output)
                except Exception as exc:  # noqa: BLE001 - keep collecting, report at the end
                    print(f"FAILED {year} {meeting['meeting_name']} {session['session_code']}: {exc!r}", flush=True)
                    failures.append((year, str(meeting["meeting_name"]), repr(exc)))
                    continue
                if path is None:
                    print(f"NO LAPS {year} {meeting['meeting_name']} {session['session_code']}", flush=True)
                    continue
                produced.append(path)
                print(f"OK {year} R{meeting['round_number']} {meeting['meeting_name']} {session['session_code']}: {path.name} ({client.requests_made} requests)", flush=True)
    print(f"Sessions written or present: {len(produced)}; skipped existing: {skipped}; failures: {len(failures)}")
    if args.combine and produced:
        frame = pd.concat([pd.read_parquet(p) for p in sorted(produced)], ignore_index=True)
        target = Path(args.combine)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        print(f"Combined {len(frame)} laps into {target}")
    for item in failures:
        print("  ", item)
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
