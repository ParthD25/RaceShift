#!/usr/bin/env python
"""Fetch FastF1 race sessions for RaceShift, one parquet per session.

Resumable and rate-limit aware: sessions whose parquet already exists are skipped
(``--skip-existing``), events that have not happened yet are skipped, and when FastF1
reports that the public API budget (500 calls/hour) is exhausted the script sleeps and
retries the same session instead of failing.

Example:
    python scripts/fetch_fastf1_seasons.py --years 2018-2026 --session R --skip-existing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.fastf1_loader import export_session  # noqa: E402


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            years.extend(range(a, b + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


def main():
    parser = argparse.ArgumentParser(description="Fetch multiple FastF1 race seasons for RaceShift.")
    parser.add_argument("--years", default="2022-2025", help="Example: 2022-2025 or 2022,2024,2025")
    parser.add_argument("--session", default="R")
    parser.add_argument("--max-events", type=int, default=0, help="0 means every event in each season")
    parser.add_argument("--events", nargs="*", help="Optional substrings matched against event name, location and country (e.g. Monza, Silverstone, Bahrain)")
    parser.add_argument("--output", default=str(ROOT / "data" / "raw" / "fastf1"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache" / "fastf1"))
    parser.add_argument("--skip-existing", action="store_true", help="Skip sessions whose parquet already exists")
    parser.add_argument("--rate-limit-wait", type=int, default=900, help="Seconds to sleep when the FastF1 API budget is exhausted (0 = fail instead)")
    args = parser.parse_args()

    import fastf1
    from fastf1.exceptions import RateLimitExceededError

    failures = []
    completed = []
    skipped = 0
    now = pd.Timestamp.utcnow().tz_localize(None)
    for year in parse_years(args.years):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        schedule = schedule[schedule["RoundNumber"] > 0]
        if "EventDate" in schedule:
            schedule = schedule[pd.to_datetime(schedule["EventDate"]) <= now]
        if args.events:
            # Match circuit names ("Monza", "Silverstone") as well as event names and countries,
            # because FastF1 names rounds "Italian Grand Prix", "British Grand Prix" and so on.
            needles = [e.lower() for e in args.events]
            haystack = pd.Series("", index=schedule.index)
            for column in ("EventName", "Location", "Country"):
                if column in schedule:
                    haystack = haystack + " " + schedule[column].astype(str).str.lower()
            schedule = schedule[haystack.apply(lambda x: any(n in x for n in needles))]
        if args.max_events > 0:
            schedule = schedule.head(args.max_events)

        for _, row in schedule.iterrows():
            event = str(row["EventName"])
            expected = Path(args.output) / f"{year}_{event.replace(' ', '_')}_{args.session}.parquet"
            if args.skip_existing and expected.exists():
                skipped += 1
                continue
            while True:
                try:
                    path = export_session(year, event, args.session, args.output, args.cache)
                    print(f"OK {year} {event}: {path.name}", flush=True)
                    completed.append((year, event))
                except RateLimitExceededError as exc:
                    if args.rate_limit_wait <= 0:
                        raise
                    print(f"RATE LIMIT at {year} {event}: {exc}. Sleeping {args.rate_limit_wait}s", flush=True)
                    time.sleep(args.rate_limit_wait)
                    continue
                except Exception as exc:  # noqa: BLE001 - keep collecting, report at the end
                    print(f"FAILED {year} {event}: {exc!r}", flush=True)
                    failures.append((year, event, repr(exc)))
                break

    print(f"Completed {len(completed)} sessions; skipped {skipped} existing; failures {len(failures)}")
    if failures:
        for item in failures:
            print("  ", item)


if __name__ == "__main__":
    main()
