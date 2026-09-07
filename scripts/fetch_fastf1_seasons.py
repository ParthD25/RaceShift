#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.fastf1_loader import export_session


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
    parser.add_argument("--events", nargs="*", help="Optional event-name substrings to keep")
    parser.add_argument("--output", default=str(ROOT / "data" / "raw" / "fastf1"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache" / "fastf1"))
    args = parser.parse_args()

    import fastf1

    failures = []
    completed = []
    for year in parse_years(args.years):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        schedule = schedule[schedule["RoundNumber"] > 0]
        if args.events:
            needles = [e.lower() for e in args.events]
            schedule = schedule[schedule["EventName"].astype(str).str.lower().apply(lambda x: any(n in x for n in needles))]
        if args.max_events > 0:
            schedule = schedule.head(args.max_events)

        for _, row in schedule.iterrows():
            event = str(row["EventName"])
            try:
                path = export_session(year, event, args.session, args.output, args.cache)
                print(f"OK {year} {event}: {path.name}")
                completed.append((year, event))
            except Exception as exc:
                print(f"FAILED {year} {event}: {exc}")
                failures.append((year, event, repr(exc)))

    print(f"Completed {len(completed)} sessions; failures {len(failures)}")
    if failures:
        for item in failures:
            print("  ", item)


if __name__ == "__main__":
    main()
