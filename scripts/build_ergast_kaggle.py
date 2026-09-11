#!/usr/bin/env python
"""Build RaceShift legacy-tier lap tables from the Kaggle Ergast CSV dump.

    python scripts/build_ergast_kaggle.py --csv-dir data/raw/ergast_kaggle --years 1996-2017 \
        --output data/processed/f1_laps_ergast.parquet

Get the CSVs first, for example with kagglehub (needs a Kaggle API token):

    python -c "import kagglehub; print(kagglehub.dataset_download('jtrotman/formula-1-race-data'))"

Rows are tagged ``data_tier = "legacy_timing"`` and match what ``fetch_jolpica_seasons.py``
produces from the API, so either can feed the legacy-extended training table. Seasons from
2018 on are the FastF1 tier's; they are exported only with ``--include-timing-era`` and
only for the lap-for-lap checks (``scripts/cross_provider_check.py``), never concatenated
with FastF1 rows of the same seasons.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from raceshift.data.ergast_csv_loader import export_seasons  # noqa: E402
from fetch_fastf1_seasons import parse_years  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Convert the Kaggle Ergast CSVs into a RaceShift legacy-tier lap table.")
    p.add_argument("--csv-dir", required=True, help="Folder holding lap_times.csv, races.csv, drivers.csv, results.csv, constructors.csv, circuits.csv (pit_stops.csv optional)")
    p.add_argument("--years", default="1996-2017")
    p.add_argument("--output", default=str(ROOT / "data" / "processed" / "f1_laps_ergast.parquet"))
    p.add_argument("--include-timing-era", action="store_true", help="Also export seasons from 2018 on (FastF1 tier years) for cross-provider checks; never train on them together with FastF1 rows")
    args = p.parse_args()
    path = export_seasons(args.csv_dir, parse_years(args.years), args.output, include_timing_era=args.include_timing_era)
    print(path)


if __name__ == "__main__":
    main()
