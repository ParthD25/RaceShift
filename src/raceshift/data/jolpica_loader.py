"""Legacy lap timing (1996-2017) from the Jolpica/Ergast API.

FastF1's live-timing archive starts in 2018. Earlier seasons are available only through
the Ergast schema (served today by Jolpica): per-lap times and positions from 1996, pit
stops from 2011, and no sectors, tyre compounds, track status or weather. Rows exported
here are tagged ``data_tier = "legacy_timing"`` so models and reports can tell the two
tiers apart, and the columns that do not exist for this tier are left missing rather than
invented.

Lap validity for this tier is a documented heuristic: without track status or pit
markers, a lap is treated as an accurate racing lap only when it is not the opening lap,
is not a recorded pit-stop lap or the lap after one (2011+), and is no slower than
``LEGACY_SLOW_LAP_RATIO`` times the driver's median lap of that race. Safety-car periods
and pre-2011 pit stops are caught by the ratio rule, which is intentionally conservative.
"""
from __future__ import annotations

import time
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100  # Jolpica caps ``limit`` at 100 rows per request.
LEGACY_SLOW_LAP_RATIO = 1.12
DATA_TIER = "legacy_timing"

# Ergast constructorId -> the team name FastF1 uses from 2018 on, so priors link across tiers.
CONSTRUCTOR_TO_TEAM = {
    "red_bull": "Red Bull Racing",
    "mercedes": "Mercedes",
    "ferrari": "Ferrari",
    "mclaren": "McLaren",
    "williams": "Williams",
    "renault": "Renault",
    "alpine": "Alpine",
    "haas": "Haas F1 Team",
    "toro_rosso": "Toro Rosso",
    "alphatauri": "AlphaTauri",
    "rb": "RB",
    "force_india": "Force India",
    "racing_point": "Racing Point",
    "aston_martin": "Aston Martin",
    "sauber": "Sauber",
    "alfa": "Alfa Romeo",
}

# Ergast circuitId -> the FastF1 ``Location`` string used as RaceShift's ``circuit``.
CIRCUIT_TO_LOCATION = {
    "bahrain": "Sakhir",
    "jeddah": "Jeddah",
    "albert_park": "Melbourne",
    "baku": "Baku",
    "miami": "Miami",
    "imola": "Imola",
    "monaco": "Monaco",
    "catalunya": "Barcelona",
    "villeneuve": "Montréal",
    "red_bull_ring": "Spielberg",
    "silverstone": "Silverstone",
    "hungaroring": "Budapest",
    "spa": "Spa-Francorchamps",
    "zandvoort": "Zandvoort",
    "monza": "Monza",
    "marina_bay": "Marina Bay",
    "suzuka": "Suzuka",
    "losail": "Lusail",
    "americas": "Austin",
    "rodriguez": "Mexico City",
    "interlagos": "São Paulo",
    "vegas": "Las Vegas",
    "yas_marina": "Yas Island",
    "shanghai": "Shanghai",
    "sochi": "Sochi",
    "ricard": "Le Castellet",
    "hockenheimring": "Hockenheim",
    "istanbul": "Istanbul",
    "nurburgring": "Nürburgring",
    "portimao": "Portimão",
    "mugello": "Mugello",
    "sepang": "Kuala Lumpur",
}


class JolpicaClient:
    """Small polite client: bounded requests per hour, minimum spacing, 429 back-off."""

    def __init__(self, per_hour: int = 480, min_interval_s: float = 0.6, session: requests.Session | None = None):
        self.per_hour = per_hour
        self.min_interval_s = min_interval_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "RaceShift research collector (github.com/parthd25/raceshift)"
        self._calls: deque[float] = deque()
        self._last = 0.0
        self.requests_made = 0

    def _throttle(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()
        if len(self._calls) >= self.per_hour:
            wait = 3600 - (now - self._calls[0]) + 1
            print(f"[jolpica] hourly budget reached, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
        gap = self.min_interval_s - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)

    def get(self, path: str, **params: Any) -> dict:
        url = f"{BASE_URL}/{path}.json"
        backoff = 30.0
        for _attempt in range(12):
            self._throttle()
            self._last = time.monotonic()
            self._calls.append(self._last)
            self.requests_made += 1
            response = self.session.get(url, params=params, timeout=60)
            if response.status_code == 429:
                print(f"[jolpica] 429 on {path}; sleeping {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 900)
                continue
            if response.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 900)
                continue
            response.raise_for_status()
            return response.json()["MRData"]
        raise RuntimeError(f"Jolpica request failed repeatedly: {url}")

    def paged(self, path: str, table: str, item: str) -> tuple[list[dict], list[dict]]:
        """Collect every page of a race-table sub-resource (laps, pit stops, results)."""
        offset = 0
        races: list[dict] = []
        while True:
            data = self.get(path, limit=PAGE_LIMIT, offset=offset)
            page_races = data.get(table, {}).get("Races", [])
            races.extend(page_races)
            total = int(data.get("total", 0))
            offset += PAGE_LIMIT
            if offset >= total or not page_races:
                break
        # Sub-resources arrive nested per race; flatten while keeping race metadata once.
        items: list[dict] = []
        for race in races:
            items.extend(race.get(item, []))
        return items, races


def _lap_time_seconds(text: str) -> float:
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(text)
    except ValueError:
        return float("nan")


def _driver_code(driver: dict) -> str:
    code = driver.get("code")
    if code:
        return str(code).upper()
    family = str(driver.get("familyName", driver.get("driverId", "UNK")))
    ascii_family = unicodedata.normalize("NFKD", family).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_family.upper() if ch.isalpha())[:3] or "UNK"


def season_schedule(client: JolpicaClient, year: int) -> list[dict]:
    data = client.get(str(year), limit=100)
    return data.get("RaceTable", {}).get("Races", [])


def export_race(client: JolpicaClient, year: int, round_number: int, output_dir: str | Path) -> Path | None:
    """Fetch one race's laps (+ results, + pit stops when available) and write a parquet.

    Returns None when the race has no lap timing (Ergast lap data starts in 1996 and a
    few early races are incomplete).
    """
    laps, races = client.paged(f"{year}/{round_number}/laps", "RaceTable", "Laps")
    if not laps or not races:
        return None
    race = races[0]
    results, _ = client.paged(f"{year}/{round_number}/results", "RaceTable", "Results")
    pit_stops: list[dict] = []
    if year >= 2011:
        pit_stops, _ = client.paged(f"{year}/{round_number}/pitstops", "RaceTable", "PitStops")

    driver_meta = {
        r["Driver"]["driverId"]: (
            _driver_code(r["Driver"]),
            CONSTRUCTOR_TO_TEAM.get(r["Constructor"]["constructorId"], r["Constructor"]["name"]),
        )
        for r in results
    }
    pit_laps = {(p["driverId"], int(p["lap"])) for p in pit_stops}

    rows: list[dict] = []
    circuit = race["Circuit"]
    location = CIRCUIT_TO_LOCATION.get(circuit["circuitId"], circuit["Location"]["locality"])
    for lap in laps:
        lap_number = int(lap["number"])
        for timing in lap.get("Timings", []):
            driver_id = timing["driverId"]
            code, team = driver_meta.get(driver_id, (driver_id[:3].upper(), None))
            rows.append(
                {
                    "season": year,
                    "series": "F1",
                    "round_number": round_number,
                    "event": race["raceName"],
                    "circuit": location,
                    "event_date": race["date"],
                    "session": "R",
                    "driver": code,
                    "driver_id": driver_id,
                    "team": team,
                    "lap_number": lap_number,
                    "lap_time_s": _lap_time_seconds(timing["time"]),
                    "position": int(timing["position"]) if timing.get("position") else None,
                    "pit_in": (driver_id, lap_number) in pit_laps,
                    "pit_out": (driver_id, lap_number - 1) in pit_laps,
                    "data_tier": DATA_TIER,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame["tyre_manufacturer"] = "Bridgestone" if year <= 2010 else "Pirelli"
    frame["track_status"] = None
    frame["deleted"] = False
    median = frame.groupby("driver")["lap_time_s"].transform("median")
    frame["is_accurate"] = (
        (frame["lap_number"] > 1)
        & ~frame["pit_in"]
        & ~frame["pit_out"]
        & (frame["lap_time_s"] <= LEGACY_SLOW_LAP_RATIO * median)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{year}_{race['raceName'].replace(' ', '_')}_R.parquet"
    frame.to_parquet(path, index=False)
    return path
