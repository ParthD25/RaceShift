"""Lap timing from the TracingInsights public archive (https://tracinginsights.com/data).

TracingInsights publishes one GitHub repository per season (``TracingInsights/2018`` ...
``TracingInsights/2026``) with a folder per event and session. ``session_laptimes.json``
in each session folder is FastF1's lap table for that session, column by column, with the
per-lap weather sample already joined; for races the lap times and positions are
overwritten with the official Ergast values. It is therefore not an independent measurement
of the laps (FastF1 is the upstream), but it is a complete, fast, rate-limit-free mirror of
the FastF1 tier for every session type since 2018, and a third table to hold FastF1 and
OpenF1 against. Rows are tagged ``data_tier = "tracinginsights_timing"``.

The file is columnar: each key maps to one list with one entry per lap. Missing values are
the string ``"None"``. The circuit is not stored; :func:`apply_calendar` takes it, with the
official round number and event date, from any RaceShift lap table of the same seasons
(the FastF1 table), so the rows join the FastF1 tier exactly. Without a calendar the
circuit falls back to a season-aware map of FastF1's ``Location`` strings, which FastF1
itself renames between seasons (Miami/Miami Gardens, Monaco/Monte Carlo).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_TIER = "tracinginsights_timing"
RAW_BASE = "https://raw.githubusercontent.com/TracingInsights"

# TracingInsights folder names -> RaceShift session codes.
SESSION_FOLDERS = {
    "Race": "R",
    "Sprint": "S",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}

# FastF1 event name -> FastF1 ``Location`` (the ``circuit`` value of the FastF1 tier).
# FastF1 renames a few locations between seasons; RENAMED_CIRCUITS lists them as
# (first season, name) steps so the fallback reproduces the value of the same season.
RENAMED_CIRCUITS = {
    "Abu Dhabi Grand Prix": [(2018, "Yas Marina"), (2019, "Yas Island")],
    "Miami Grand Prix": [(2022, "Miami"), (2025, "Miami Gardens")],
    "Monaco Grand Prix": [(2018, "Monte Carlo"), (2022, "Monaco"), (2026, "Monte Carlo")],
    "Singapore Grand Prix": [(2018, "Singapore"), (2022, "Marina Bay")],
}
EVENT_TO_CIRCUIT = {
    "70th Anniversary Grand Prix": "Silverstone",
    "Abu Dhabi Grand Prix": "Yas Island",
    "Australian Grand Prix": "Melbourne",
    "Austrian Grand Prix": "Spielberg",
    "Azerbaijan Grand Prix": "Baku",
    "Bahrain Grand Prix": "Sakhir",
    "Barcelona Grand Prix": "Barcelona",
    "Belgian Grand Prix": "Spa-Francorchamps",
    "Brazilian Grand Prix": "São Paulo",
    "British Grand Prix": "Silverstone",
    "Canadian Grand Prix": "Montréal",
    "Chinese Grand Prix": "Shanghai",
    "Dutch Grand Prix": "Zandvoort",
    "Eifel Grand Prix": "Nürburgring",
    "Emilia Romagna Grand Prix": "Imola",
    "French Grand Prix": "Le Castellet",
    "German Grand Prix": "Hockenheim",
    "Hungarian Grand Prix": "Budapest",
    "Italian Grand Prix": "Monza",
    "Japanese Grand Prix": "Suzuka",
    "Las Vegas Grand Prix": "Las Vegas",
    "Mexican Grand Prix": "Mexico City",
    "Mexico City Grand Prix": "Mexico City",
    "Monaco Grand Prix": "Monaco",
    "Portuguese Grand Prix": "Portimão",
    "Qatar Grand Prix": "Lusail",
    "Russian Grand Prix": "Sochi",
    "Sakhir Grand Prix": "Sakhir",
    "Saudi Arabian Grand Prix": "Jeddah",
    "Singapore Grand Prix": "Marina Bay",
    "Spanish Grand Prix": "Barcelona",
    "Styrian Grand Prix": "Spielberg",
    "São Paulo Grand Prix": "São Paulo",
    "Turkish Grand Prix": "Istanbul",
    "Tuscan Grand Prix": "Mugello",
    "United States Grand Prix": "Austin",
    "Malaysian Grand Prix": "Kuala Lumpur",
}


def circuit_for(event: str, year: int) -> str:
    """FastF1's ``Location`` for an event in a given season."""
    steps = RENAMED_CIRCUITS.get(str(event))
    if steps:
        name = steps[0][1]
        for first_season, step_name in steps:
            if int(year) >= first_season:
                name = step_name
        return name
    return EVENT_TO_CIRCUIT.get(str(event), str(event).replace(" Grand Prix", ""))


def raw_url(year: int, event: str, session_folder: str, file: str = "session_laptimes.json") -> str:
    from urllib.parse import quote

    return f"{RAW_BASE}/{year}/main/{quote(event)}/{quote(session_folder)}/{file}"


def _column(payload: dict, key: str, n: int) -> pd.Series:
    values = payload.get(key)
    if values is None or len(values) != n:
        return pd.Series([np.nan] * n, dtype="object")
    return pd.Series([np.nan if v == "None" or v is None else v for v in values], dtype="object")


def _num(payload: dict, key: str, n: int) -> pd.Series:
    return pd.to_numeric(_column(payload, key, n), errors="coerce")


def _flag(payload: dict, key: str, n: int) -> pd.Series:
    col = _column(payload, key, n)
    return col.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else (str(v).lower() == "true")).astype(bool)


def session_frame(payload: dict, year: int, event: str, session_code: str, round_number: int | None = None, circuit: str | None = None) -> pd.DataFrame:
    """One session's laps in the RaceShift schema from a ``session_laptimes.json`` payload."""
    n = len(payload.get("lap", []))
    if n == 0:
        return pd.DataFrame()
    lap_dates = pd.to_datetime(_column(payload, "lSD", n), errors="coerce", format="ISO8601")
    event_date = str(lap_dates.dropna().min().date()) if lap_dates.notna().any() else None
    pit_in_time = _column(payload, "pin", n)
    pit_out_time = _column(payload, "pout", n)
    frame = pd.DataFrame({
        "season": int(year),
        "series": "F1",
        "round_number": round_number,
        "event": str(event),
        "circuit": circuit or circuit_for(event, year),
        "event_date": event_date,
        "session": session_code,
        "driver": _column(payload, "drv", n).astype(str),
        "team": _column(payload, "team", n),
        "lap_number": _num(payload, "lap", n).astype(float),
        "lap_time_s": _num(payload, "time", n).astype(float),
        "sector1_s": _num(payload, "s1", n).astype(float),
        "sector2_s": _num(payload, "s2", n).astype(float),
        "sector3_s": _num(payload, "s3", n).astype(float),
        "compound": _column(payload, "compound", n),
        "tyre_life": _num(payload, "life", n).astype(float),
        "fresh_tyre": _flag(payload, "fresh", n),
        "tyre_manufacturer": "Pirelli",
        "stint": _num(payload, "stint", n).astype(float),
        "position": _num(payload, "pos", n).astype(float),
        "track_status": _column(payload, "status", n).map(lambda v: "1" if (isinstance(v, float) and np.isnan(v)) else str(v)),
        "pit_in": pit_in_time.notna(),
        "pit_out": pit_out_time.notna(),
        "is_accurate": _flag(payload, "iacc", n),
        "deleted": _flag(payload, "del", n),
        "data_tier": DATA_TIER,
        "air_temp_c": _num(payload, "wAT", n).astype(float),
        "track_temp_c": _num(payload, "wTT", n).astype(float),
        "humidity_pct": _num(payload, "wH", n).astype(float),
        "pressure_mbar": _num(payload, "wP", n).astype(float),
        "rainfall": _flag(payload, "wR", n),
        "wind_speed_ms": _num(payload, "wWS", n).astype(float),
        "wind_direction_deg": _num(payload, "wWD", n).fillna(0).astype(int),
        "lap_start_time": lap_dates,
    })
    frame = frame.sort_values(["driver", "lap_number"]).reset_index(drop=True)
    return frame


def load_session_file(path: str | Path, year: int, event: str, session_code: str, round_number: int | None = None) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    return session_frame(payload, year, event, session_code, round_number)


def apply_calendar(frames: list[pd.DataFrame], calendar: pd.DataFrame | None = None) -> list[pd.DataFrame]:
    """Take ``round_number``, ``circuit`` and ``event_date`` per (season, event) from
    ``calendar``, any RaceShift lap table holding those columns (the FastF1 table), so the
    archive's rows carry exactly the identifiers the FastF1 tier uses for the same race.

    Without a calendar ``round_number`` is left unset: numbering the events the archive
    happens to hold would shift every round after a missing event, and ``event_date``
    already orders events chronologically. Events absent from the calendar keep the
    fallback circuit and their own event date and stay without a round."""
    known: dict[tuple[int, str], dict] = {}
    if calendar is not None and {"season", "event"} <= set(calendar.columns):
        cal = calendar.drop_duplicates(["season", "event"])
        for _, row in cal.iterrows():
            entry = {}
            if "round_number" in cal.columns and pd.notna(row["round_number"]):
                entry["round_number"] = int(row["round_number"])
            if "circuit" in cal.columns and pd.notna(row["circuit"]):
                entry["circuit"] = str(row["circuit"])
            if "event_date" in cal.columns and pd.notna(row["event_date"]):
                entry["event_date"] = str(row["event_date"])
            known[(int(row["season"]), str(row["event"]))] = entry
    out = []
    for f in frames:
        if not f.empty:
            f = f.copy()
            entry = known.get((int(f["season"].iloc[0]), str(f["event"].iloc[0])), {})
            f["round_number"] = entry.get("round_number")
            for col in ("circuit", "event_date"):
                if col in entry:
                    f[col] = entry[col]
        out.append(f)
    return out
