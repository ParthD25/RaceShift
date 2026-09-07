#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.features.lap_features import build_next_lap_table


def main():
    parser = argparse.ArgumentParser(description="Build leakage-safe next-lap training table from FastF1 Parquet files.")
    parser.add_argument("--input", default=str(ROOT / "data" / "raw" / "fastf1"))
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "next_lap.parquet"))
    args = parser.parse_args()

    root = Path(args.input)
    files = sorted(root.glob("*.parquet")) + sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"No parquet/csv files in {args.input}. Run fetch_fastf1.py first.")
    frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    table = build_next_lap_table(raw)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        table.to_csv(out, index=False)
    else:
        table.to_parquet(out, index=False)
    print(f"Wrote {len(table):,} forecast rows to {out}")


if __name__ == "__main__":
    main()
