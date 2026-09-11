"""Legacy lap timing from the Ergast database dump published on Kaggle.

``jtrotman/formula-1-race-data`` (https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)
is the Ergast/Jolpica database as CSV files: ``lap_times.csv`` (every lap since 1996),
``pit_stops.csv`` (since 2011), ``races.csv``, ``drivers.csv``, ``results.csv``,
``constructors.csv`` and ``circuits.csv``. It is the same data the Jolpica API serves, so
this loader is an offline mirror: rows carry ``data_tier = "legacy_timing"`` and are built
by the same rules (:func:`raceshift.data.jolpica_loader.finish_legacy_frame`), and a race
built from the CSVs matches the API row for row. It reaches 1996-1999, which the API
collector skips by default, and it is updated a few days after each race, so it also
covers the current season without touching the API.

Download the CSVs with ``kagglehub.dataset_download("jtrotman/formula-1-race-data")`` (needs
a Kaggle token) or from the dataset page, then point ``--csv-dir`` at the folder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .jolpica_loader import CIRCUIT_TO_LOCATION, DATA_TIER, _ascii_code, _lap_time_seconds, finish_legacy_frame, team_name

REQUIRED_FILES = ("lap_times.csv", "races.csv", "drivers.csv", "results.csv", "constructors.csv", "circuits.csv")

# From this season the FastF1 timing tier covers every race; Ergast rows for these seasons
# are for integrity checks only, never for a training table that also holds FastF1 rows.
FIRST_TIMING_SEASON = 2018


def _read(csv_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(csv_dir / name, na_values=["\\N"], keep_default_na=True)


def _driver_code(code: object, surname: object) -> str:
    if isinstance(code, str) and code.strip():
        return _ascii_code(code.strip())
    return _ascii_code(str(surname or "UNK"))


def load_tables(csv_dir: str | Path) -> dict[str, pd.DataFrame]:
    csv_dir = Path(csv_dir)
    missing = [name for name in REQUIRED_FILES if not (csv_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Ergast CSV directory {csv_dir} is missing {missing}")
    tables = {name.split(".")[0]: _read(csv_dir, name) for name in REQUIRED_FILES}
    pit_file = csv_dir / "pit_stops.csv"
    tables["pit_stops"] = _read(csv_dir, "pit_stops.csv") if pit_file.is_file() else pd.DataFrame(columns=["raceId", "driverId", "lap"])
    return tables


def season_frame(tables: dict[str, pd.DataFrame], year: int) -> pd.DataFrame:
    """Every race lap of one season in the RaceShift legacy schema (empty when none)."""
    races = tables["races"]
    races = races[races["year"] == year].sort_values("round")
    if races.empty:
        return pd.DataFrame()
    laps = tables["lap_times"].merge(races[["raceId", "round", "name", "date", "circuitId"]], on="raceId", how="inner")
    if laps.empty:
        return pd.DataFrame()
    drivers = tables["drivers"][["driverId", "driverRef", "code", "surname"]]
    laps = laps.merge(drivers, on="driverId", how="left")
    # Team of each driver in each race, from the results table (a driver can change team).
    results = tables["results"][["raceId", "driverId", "constructorId"]].drop_duplicates(["raceId", "driverId"])
    constructors = tables["constructors"][["constructorId", "constructorRef", "name"]].rename(columns={"name": "constructor_name"})
    results = results.merge(constructors, on="constructorId", how="left")
    laps = laps.merge(results[["raceId", "driverId", "constructorRef", "constructor_name"]], on=["raceId", "driverId"], how="left")
    circuits = tables["circuits"][["circuitId", "circuitRef", "location"]]
    laps = laps.merge(circuits, on="circuitId", how="left")
    pits = tables["pit_stops"]
    pit_laps = set(zip(pits["raceId"].astype(int), pits["driverId"].astype(int), pits["lap"].astype(int))) if len(pits) else set()

    key = list(zip(laps["raceId"].astype(int), laps["driverId"].astype(int), laps["lap"].astype(int)))
    prev = list(zip(laps["raceId"].astype(int), laps["driverId"].astype(int), (laps["lap"].astype(int) - 1)))
    frame = pd.DataFrame({
        "season": int(year),
        "series": "F1",
        "round_number": laps["round"].astype(int),
        "event": laps["name"].astype(str),
        "circuit": [CIRCUIT_TO_LOCATION.get(ref, loc) for ref, loc in zip(laps["circuitRef"], laps["location"])],
        "event_date": laps["date"].astype(str),
        "session": "R",
        "driver": [_driver_code(c, s) for c, s in zip(laps["code"], laps["surname"])],
        "driver_id": laps["driverRef"].astype(str),
        "team": [team_name(ref, int(year), name) if isinstance(ref, str) else None for ref, name in zip(laps["constructorRef"], laps["constructor_name"])],
        "lap_number": laps["lap"].astype(int),
        "lap_time_s": [ms / 1000.0 if pd.notna(ms) else _lap_time_seconds(t) for ms, t in zip(laps["milliseconds"], laps["time"])],
        "position": pd.to_numeric(laps["position"], errors="coerce"),
        "pit_in": [k in pit_laps for k in key],
        "pit_out": [k in pit_laps for k in prev],
        "data_tier": DATA_TIER,
    })
    frame = frame.sort_values(["round_number", "driver", "lap_number"]).reset_index(drop=True)
    # The heuristic is per race, as in the API loader: apply it race by race.
    parts = [finish_legacy_frame(g, year) for _, g in frame.groupby("round_number", sort=True)]
    out = pd.concat(parts, ignore_index=True)
    out["position"] = out["position"].astype(float)
    out["lap_number"] = out["lap_number"].astype(int)
    return out


def export_seasons(csv_dir: str | Path, years: list[int], output: str | Path, include_timing_era: bool = False) -> Path:
    """Write the selected seasons to one parquet in the RaceShift legacy schema.

    Seasons from :data:`FIRST_TIMING_SEASON` on are refused unless ``include_timing_era``
    is set: those races already exist in the FastF1 tier with sectors, tyres and weather,
    and a training table holding both copies would duplicate every lap. The timing-era
    rows are only for lap-for-lap checks against the timing providers."""
    timing_era = [y for y in years if y >= FIRST_TIMING_SEASON]
    if timing_era and not include_timing_era:
        raise ValueError(f"Seasons {timing_era} belong to the FastF1 timing tier; pass include_timing_era=True (--include-timing-era) to export them for cross-provider checks only")
    tables = load_tables(csv_dir)
    frames = [season_frame(tables, year) for year in years]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError(f"No lap timing found for seasons {years}")
    out = pd.concat(frames, ignore_index=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path
