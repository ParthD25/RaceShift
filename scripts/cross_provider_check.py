#!/usr/bin/env python
"""Compare two lap tables of the same races from different providers, lap for lap.

Shared laps are matched on season, round number, session, driver and lap number when both
tables carry official round numbers (providers spell event names differently: Ergast's
"Barcelona-Catalunya Grand Prix" is FastF1's "Barcelona Grand Prix"), and on the event
name otherwise. Rows with a missing key and every row of a duplicated key (two drivers
sharing a three-letter code in one race) are dropped and counted in the report, so the
merge is one lap to one lap. The report gives
coverage (laps only one side has), agreement per column, and where the lap times
disagree. Used to verify that OpenF1 and FastF1 describe the same races identically,
that the Kaggle Ergast dump equals the Jolpica API rows, and that the legacy tier's lap
times match the timing tier where the years overlap.

    python scripts/cross_provider_check.py --left data/processed/f1_laps_fastf1.parquet \
        --right data/imports/f1_races_openf1.parquet --names fastf1 openf1 \
        --output reports/data_sources/fastf1_vs_openf1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["season", "event", "session", "driver", "lap_number"]
ROUND_KEYS = ["season", "round_number", "session", "driver", "lap_number"]
NUMERIC = {"lap_time_s": 0.002, "sector1_s": 0.002, "sector2_s": 0.002, "sector3_s": 0.002, "position": 0, "tyre_life": 0, "stint": 0,
           "air_temp_c": 0.6, "track_temp_c": 0.6, "humidity_pct": 3.0, "wind_speed_ms": 1.5}
EXACT = ["compound", "team", "circuit", "round_number", "event_date", "track_status", "pit_in", "pit_out", "is_accurate", "deleted", "fresh_tyre", "rainfall"]


def load(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)
    for col in KEYS:
        if col not in frame.columns:
            raise SystemExit(f"{path} lacks key column {col}")
    frame = frame.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["lap_number"] = pd.to_numeric(frame["lap_number"], errors="coerce")
    # Normalise the code without turning a missing session into the string "nan".
    frame["session"] = frame["session"].where(frame["session"].isna(), frame["session"].astype(str))
    if "round_number" in frame.columns:
        frame["round_number"] = pd.to_numeric(frame["round_number"], errors="coerce")
    return frame


def match_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Official round numbers when both tables carry them (rows without one are then
    dropped as missing keys and counted), else event names."""
    if all("round_number" in f.columns and f["round_number"].notna().any() for f in (left, right)):
        return ROUND_KEYS
    return KEYS


def _unique_rows(frame: pd.DataFrame, keys: list[str], exclude_first_lap: bool) -> tuple[pd.DataFrame, int, int]:
    """Rows with a complete key that are unambiguous: every row of a duplicated key (two
    drivers sharing a three-letter code in one race) is excluded, since keeping either
    would compare an arbitrary driver's lap. Returns (rows, dropped for a missing key,
    dropped as duplicates); the opening lap, when excluded, is removed after the missing
    keys are counted."""
    complete = frame.dropna(subset=keys)
    missing = int(len(frame) - len(complete))
    if exclude_first_lap:
        complete = complete[complete["lap_number"] > 1]
    unique = complete[~complete.duplicated(subset=keys, keep=False)]
    return unique, missing, int(len(complete) - len(unique))


def _event_key(frame: pd.DataFrame, keys: list[str]) -> set[str]:
    """Event identifiers matching the lap keys: official rounds (with the name for
    reading) or the event name."""
    if keys is ROUND_KEYS:
        return {f"{int(s)} R{int(r)} {e}" for s, r, e in zip(frame["season"], frame["round_number"], frame["event"])}
    return {f"{int(s)} {e}" for s, e in zip(frame["season"], frame["event"])}


def _event_diff(a: set[str], b: set[str], keys: list[str]) -> list[str]:
    """Events in ``a`` and not in ``b``, compared on the round when the laps are."""
    if keys is ROUND_KEYS:
        b_rounds = {" ".join(x.split(" ", 2)[:2]) for x in b}
        return sorted(x for x in a if " ".join(x.split(" ", 2)[:2]) not in b_rounds)
    return sorted(a - b)


def compare(left: pd.DataFrame, right: pd.DataFrame, names: tuple[str, str], exclude_first_lap: bool = True) -> dict:
    seasons = sorted(set(left["season"].dropna().astype(int)) & set(right["season"].dropna().astype(int)))
    left = left[left["season"].isin(seasons)]
    right = right[right["season"].isin(seasons)]
    keys = match_keys(left, right)
    # Both sides lose the opening lap (and their ambiguous rows) before anything is counted,
    # so totals, one-sided laps and shared laps describe the same rows.
    left, left_missing, left_dupes = _unique_rows(left, keys, exclude_first_lap)
    right, right_missing, right_dupes = _unique_rows(right, keys, exclude_first_lap)
    merged = left.merge(right, on=keys, how="outer", suffixes=("_l", "_r"), indicator=True, validate="one_to_one")
    both = merged[merged["_merge"] == "both"]
    left_events, right_events = _event_key(left, keys), _event_key(right, keys)
    report: dict = {
        "left": names[0], "right": names[1], "seasons": seasons, "matched_on": keys,
        "laps": {"left": int(len(left)), "right": int(len(right)), "shared": int(len(both)),
                 "left_only": int((merged["_merge"] == "left_only").sum()), "right_only": int((merged["_merge"] == "right_only").sum()),
                 "dropped_missing_key": {"left": left_missing, "right": right_missing},
                 "dropped_duplicate_key": {"left": left_dupes, "right": right_dupes}},
        "events": {"left": len(left_events), "right": len(right_events),
                   "left_only": _event_diff(left_events, right_events, keys),
                   "right_only": _event_diff(right_events, left_events, keys)},
        "first_lap_excluded": exclude_first_lap,
        "agreement": {},
    }
    for col, tol in NUMERIC.items():
        if f"{col}_l" in both.columns and f"{col}_r" in both.columns:
            a, b = pd.to_numeric(both[f"{col}_l"], errors="coerce"), pd.to_numeric(both[f"{col}_r"], errors="coerce")
            present = a.notna() & b.notna()
            ok = ((a - b).abs() <= tol) & present
            entry = {"tolerance": tol, "both_present": int(present.sum()), "agree_share": round(float(ok.sum() / present.sum()), 4) if present.any() else None}
            if col == "lap_time_s" and present.any():
                diff = (a - b).abs()[present]
                entry["abs_diff_median_s"] = round(float(diff.median()), 4)
                entry["abs_diff_p99_s"] = round(float(diff.quantile(0.99)), 4)
                entry["disagree_over_1s"] = int((diff > 1.0).sum())
            report["agreement"][col] = entry
    for col in EXACT + (["event"] if keys is ROUND_KEYS else []):
        if f"{col}_l" in both.columns and f"{col}_r" in both.columns:
            a, b = both[f"{col}_l"], both[f"{col}_r"]
            present = a.notna() & b.notna()
            ok = (a.astype(str) == b.astype(str)) & present
            report["agreement"][col] = {"both_present": int(present.sum()), "agree_share": round(float(ok.sum() / present.sum()), 4) if present.any() else None}
    if "lap_time_s_l" in both.columns:
        diff = (pd.to_numeric(both["lap_time_s_l"], errors="coerce") - pd.to_numeric(both["lap_time_s_r"], errors="coerce")).abs()
        worst = both.loc[diff.nlargest(10).index, keys + ["lap_time_s_l", "lap_time_s_r"]]
        report["largest_lap_time_differences"] = [
            {k: (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating, float)) else str(v))) for k, v in row.items()}
            for row in worst.to_dict("records")
        ]
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Lap-for-lap comparison of two providers' tables.")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--names", nargs=2, default=("left", "right"))
    p.add_argument("--output", help="JSON report path")
    p.add_argument("--include-first-lap", action="store_true", help="Providers time the opening lap differently; it is excluded from agreement by default")
    args = p.parse_args()
    report = compare(load(args.left), load(args.right), tuple(args.names), exclude_first_lap=not args.include_first_lap)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
