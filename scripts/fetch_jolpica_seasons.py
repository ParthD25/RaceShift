#!/usr/bin/env python
"""Fetch legacy (pre-2018) race lap timing from Jolpica/Ergast, one parquet per race.

Resumable: races whose parquet already exists are skipped, so the script can be re-run
after a network failure or a rate-limit pause and continues where it stopped. The client
keeps under the public 500 requests/hour budget and backs off on HTTP 429.

Example:
    python scripts/fetch_jolpica_seasons.py --years 2000-2017 --output data/raw/jolpica
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.jolpica_loader import JolpicaClient, export_race, season_schedule  # noqa: E402
from fetch_fastf1_seasons import parse_years  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch legacy F1 lap timing (1996-2017) from Jolpica/Ergast.")
    p.add_argument("--years", default="2000-2017")
    p.add_argument("--output", default=str(ROOT / "data" / "raw" / "jolpica"))
    p.add_argument("--per-hour", type=int, default=480, help="request budget per rolling hour")
    args = p.parse_args()

    client = JolpicaClient(per_hour=args.per_hour)
    output = Path(args.output)
    completed, skipped, failures = 0, 0, []
    for year in parse_years(args.years):
        for race in season_schedule(client, year):
            round_number = int(race["round"])
            if date.fromisoformat(race["date"]) > date.today():
                continue
            expected = output / f"{year}_{race['raceName'].replace(' ', '_')}_R.parquet"
            if expected.exists():
                skipped += 1
                continue
            try:
                path = export_race(client, year, round_number, output)
            except Exception as exc:  # noqa: BLE001 - keep collecting, report at the end
                print(f"FAILED {year} R{round_number} {race['raceName']}: {exc!r}", flush=True)
                failures.append((year, round_number, repr(exc)))
                continue
            if path is None:
                print(f"NO LAP DATA {year} R{round_number} {race['raceName']}", flush=True)
                continue
            completed += 1
            print(f"OK {year} R{round_number} {race['raceName']}: {path.name} ({client.requests_made} requests so far)", flush=True)
    print(f"Completed {completed} races; skipped {skipped} existing; failures {len(failures)}")
    for item in failures:
        print("  ", item)
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
