#!/usr/bin/env python
"""Fetch lap tables from the TracingInsights public archive (github.com/TracingInsights/<year>).

The season repository is listed with a blob-less partial clone (no telemetry downloaded),
then only ``session_laptimes.json`` of the requested sessions is fetched over HTTPS and
converted to the RaceShift schema. Resumable: existing per-session parquet files are kept.

    python scripts/fetch_tracinginsights.py --years 2025-2026 --sessions R S \
        --output data/raw/tracinginsights --combine data/imports/f1_tracinginsights.parquet
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from raceshift.data.tracinginsights_loader import SESSION_FOLDERS, assign_rounds, raw_url, session_frame  # noqa: E402
from fetch_fastf1_seasons import parse_years  # noqa: E402


def list_session_files(year: int, cache: Path) -> list[tuple[str, str]]:
    """(event, session folder) pairs that have a session_laptimes.json, from a partial clone."""
    repo = cache / f"index-{year}"
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", f"https://github.com/TracingInsights/{year}", str(repo)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "origin"], check=False)
        subprocess.run(["git", "-C", str(repo), "reset", "-q", "--soft", "origin/HEAD"], check=False)
    listing = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"], check=True, capture_output=True, text=True).stdout
    out = []
    for line in listing.splitlines():
        parts = line.split("/")
        if len(parts) == 3 and parts[2] == "session_laptimes.json" and parts[1] in SESSION_FOLDERS:
            out.append((parts[0], parts[1]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch TracingInsights lap tables into RaceShift parquet files.")
    p.add_argument("--years", default="2025")
    p.add_argument("--sessions", nargs="+", default=["R"], help="Session codes: R, S, Q, SQ, SS, FP1-3")
    p.add_argument("--output", default=str(ROOT / "data" / "raw" / "tracinginsights"))
    p.add_argument("--cache", default=str(ROOT / "data" / "cache" / "tracinginsights"))
    p.add_argument("--combine", help="Also write every session of this run into one parquet")
    p.add_argument("--rounds-from", help="Lap table (parquet/csv with season, event, round_number) giving official round numbers; otherwise events are numbered by date, which is wrong when the archive lacks an event")
    args = p.parse_args()

    wanted = set(args.sessions)
    output, cache = Path(args.output), Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "RaceShift research collector (github.com/parthd25/raceshift)"
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for year in parse_years(args.years):
        for event, folder in list_session_files(year, cache):
            code = SESSION_FOLDERS[folder]
            if code not in wanted:
                continue
            target = output / f"{year}_{event.replace(' ', '_')}_{code}.parquet"
            if target.exists():
                frames.append(pd.read_parquet(target))
                continue
            json_cache = cache / str(year) / event / folder / "session_laptimes.json"
            if not json_cache.exists():
                for attempt in range(4):
                    response = session.get(raw_url(year, event, folder), timeout=120)
                    if response.status_code == 200:
                        json_cache.parent.mkdir(parents=True, exist_ok=True)
                        json_cache.write_bytes(response.content)
                        break
                    if response.status_code == 404:
                        break
                    time.sleep(2 ** attempt)
                else:
                    failures.append(f"{year} {event} {folder}: HTTP {response.status_code}")
                    continue
            if not json_cache.exists():
                failures.append(f"{year} {event} {folder}: not found")
                continue
            frame = session_frame(json.loads(json_cache.read_text()), year, event, code)
            if frame.empty:
                continue
            frames.append(frame)
            print(f"OK {year} {event} {code}: {len(frame)} laps", flush=True)
    calendar = None
    if args.rounds_from:
        calendar = pd.read_parquet(args.rounds_from) if args.rounds_from.endswith(".parquet") else pd.read_csv(args.rounds_from)
    frames = assign_rounds(frames, calendar)
    for frame in frames:
        target = output / f"{int(frame['season'].iloc[0])}_{str(frame['event'].iloc[0]).replace(' ', '_')}_{frame['session'].iloc[0]}.parquet"
        frame.to_parquet(target, index=False)
    print(f"Sessions: {len(frames)}; failures: {len(failures)}")
    for item in failures:
        print("  ", item)
    if args.combine and frames:
        combined = pd.concat(frames, ignore_index=True)
        Path(args.combine).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(args.combine, index=False)
        print(f"Combined {len(combined)} laps into {args.combine}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
