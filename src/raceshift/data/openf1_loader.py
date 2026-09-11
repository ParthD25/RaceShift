"""Lap timing from the OpenF1 API (https://openf1.org), 2023 onward.

OpenF1 is an independent provider of the same live-timing feed FastF1 reads, so it serves
two purposes here: a second source for every race and sprint since 2023 (cross-provider
checks: the two must agree lap for lap) and the only source RaceShift reads for sprint
sessions. Rows are tagged ``data_tier = "openf1_timing"`` and carry the same columns as the
FastF1 tier, so one artifact scores both without a separate feature contract.

Column mapping (verified against FastF1 on 2025 races: lap times agree to the millisecond
from lap 2 on, pit-in/pit-out laps and stint boundaries coincide):

* ``laps``: ``lap_duration`` -> ``lap_time_s``; ``duration_sector_*`` -> ``sector*_s``;
  ``is_pit_out_lap`` -> ``pit_out``. Lap 1 differs between providers by a few tenths
  (they time the opening lap from different reference points); lap 1 is never a valid
  forecasting context anyway.
* ``pit``: the lap on which a car entered the pit lane -> ``pit_in``.
* ``stints``: ``compound``, ``stint_number`` -> ``stint``; ``tyre_life`` counts from 1 on
  the first lap of a stint plus ``tyre_age_at_start`` (FastF1 semantics); ``fresh_tyre``
  is ``tyre_age_at_start == 0``.
* ``position``: the last position update at or before the end of the lap.
* ``weather``: the nearest sample to the start of the lap (within 10 minutes).
* ``race_control``: rebuilt into FastF1's ``track_status`` codes for the interval a lap
  was on track (safety car from DEPLOYED to the green after IN THIS LAP or a ROLLING START,
  races started behind the safety car, VSC DEPLOYED/ENDING, red flags and session aborts
  until the restart): ``1`` clear, ``2`` yellow (any sector), ``4`` safety car, ``5`` red flag,
  ``6`` virtual safety car, ``7`` VSC ending. Several codes concatenate as in FastF1
  (``"12"``, ``"45"``). Time-deletion messages set ``deleted``.
* ``is_accurate`` mirrors FastF1's definition: a timed lap with all three sectors that sum
  to the lap time, no pit entry or exit, and a clear track throughout.

The public API has no authentication and asks for polite use; the client spaces requests
and backs off on HTTP 429. Every response is cached as JSON so a season is fetched once.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
DATA_TIER = "openf1_timing"

# OpenF1 session names -> RaceShift session codes (the FastF1 abbreviations).
SESSION_CODES = {
    "Race": "R",
    "Sprint": "S",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}
# OpenF1 ``location`` is the FastF1 ``Location`` string RaceShift uses as ``circuit``,
# with the exceptions below.
LOCATION_ALIASES = {"Yas Marina": "Yas Island"}

WEATHER_COLUMNS = {
    "air_temperature": "air_temp_c",
    "track_temperature": "track_temp_c",
    "humidity": "humidity_pct",
    "pressure": "pressure_mbar",
    "rainfall": "rainfall",
    "wind_speed": "wind_speed_ms",
    "wind_direction": "wind_direction_deg",
}

# A clear-track lap this much slower than the driver's median clean lap is not an accurate
# racing lap (an in-lap the pit feed missed, or a damaged car).
SLOW_LAP_RATIO = 1.20
# FastF1 status codes, in the order FastF1 concatenates them.
STATUS_ORDER = ["1", "2", "4", "5", "6", "7"]
_DELETED_TIME = re.compile(r"CAR (\d+) \((\w{3})\) TIME (\d+):(\d+\.\d+) DELETED")
_DELETED_LAP = re.compile(r"CAR (\d+) \((\w{3})\) LAP DELETED.*?LAP (\d+)")


class OpenF1Client:
    """Polite cached client: minimum spacing between requests, back-off on 429/5xx, and a
    JSON cache keyed by endpoint and query so nothing is downloaded twice."""

    def __init__(self, cache_dir: str | Path | None = None, min_interval_s: float = 0.4, session: requests.Session | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval_s = min_interval_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "RaceShift research collector (github.com/parthd25/raceshift)"
        self._last = 0.0
        self.requests_made = 0

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = "&".join(f"{k}={params[k]}" for k in sorted(params))
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        return self.cache_dir / endpoint / f"{key.replace('&', '_').replace('=', '-')[:80]}_{digest}.json"

    def get(self, endpoint: str, **params: Any) -> list[dict]:
        path = self._cache_path(endpoint, params)
        if path is not None and path.exists():
            return json.loads(path.read_text())
        backoff = 5.0
        for _attempt in range(10):
            gap = self.min_interval_s - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()
            self.requests_made += 1
            response = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=120)
            if response.status_code == 404:
                # OpenF1 answers 404 when an endpoint has nothing for a session (no pit
                # stops recorded for some 2023 sprints, for example): that is "no rows".
                data: list[dict] = []
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("[]")
                return data
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            response.raise_for_status()
            data = response.json()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            return data
        raise RuntimeError(f"OpenF1 request failed repeatedly: {endpoint} {params}")


def season_meetings(client: OpenF1Client, year: int) -> list[dict]:
    """Race weekends of a season in calendar order with ``round_number`` assigned; testing
    and cancelled meetings are excluded so rounds line up with the official calendar."""
    meetings = [
        m for m in client.get("meetings", year=year)
        if "testing" not in str(m.get("meeting_name", "")).lower() and not m.get("is_cancelled")
    ]
    meetings.sort(key=lambda m: m["date_start"])
    for i, meeting in enumerate(meetings, start=1):
        meeting["round_number"] = i
    return meetings


def meeting_sessions(client: OpenF1Client, meeting_key: int, codes: set[str] | None = None) -> list[dict]:
    """Sessions of one weekend, optionally filtered to RaceShift session codes such as {"R", "S"}."""
    sessions = client.get("sessions", meeting_key=meeting_key)
    out = []
    for s in sessions:
        code = SESSION_CODES.get(str(s.get("session_name")))
        if code is None or (codes is not None and code not in codes):
            continue
        s["session_code"] = code
        out.append(s)
    return out


def _ts(values) -> pd.Series:
    """OpenF1 timestamps are ISO 8601 with or without fractional seconds, sometimes mixed
    within one payload; parse them individually so none is silently dropped."""
    return pd.to_datetime(pd.Series(values), utc=True, errors="coerce", format="ISO8601")


def track_status_intervals(race_control: list[dict], session_end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Turn race-control messages into (start, end, code) intervals in FastF1's coding.

    Yellow flags are tracked per sector and end with that sector's CLEAR or any
    track-wide CLEAR/GREEN. A safety car runs from DEPLOYED until the first track-wide
    GREEN/CLEAR after "IN THIS LAP"; a virtual safety car runs from DEPLOYED to ENDING
    (code 6) and from ENDING to the next GREEN/CLEAR (code 7); a red flag runs until the
    next GREEN.
    """
    events = sorted(race_control, key=lambda e: str(e.get("date")))
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    open_yellow: dict[str, pd.Timestamp] = {}
    sc_start: pd.Timestamp | None = None
    sc_in_this_lap = False
    vsc_start: pd.Timestamp | None = None
    vsc_ending: pd.Timestamp | None = None
    red_start: pd.Timestamp | None = None
    race_starts_behind_sc = False

    def close_yellows(at: pd.Timestamp) -> None:
        for start in list(open_yellow.values()):
            intervals.append((start, at, "2"))
        open_yellow.clear()

    for e in events:
        at = pd.Timestamp(e["date"])
        if at.tzinfo is None:
            at = at.tz_localize("UTC")
        category = str(e.get("category") or "")
        flag = str(e.get("flag") or "").upper()
        scope = str(e.get("scope") or "")
        message = str(e.get("message") or "").upper()
        if category == "Flag":
            key = f"sector-{e.get('sector')}" if scope == "Sector" else "track"
            if flag in ("YELLOW", "DOUBLE YELLOW"):
                open_yellow.setdefault(key, at)
            elif flag == "CLEAR" and scope == "Sector":
                start = open_yellow.pop(key, None)
                if start is not None:
                    intervals.append((start, at, "2"))
            elif flag in ("CLEAR", "GREEN"):
                close_yellows(at)
                if sc_start is not None and sc_in_this_lap:
                    intervals.append((sc_start, at, "4"))
                    sc_start, sc_in_this_lap = None, False
                if vsc_ending is not None:
                    intervals.append((vsc_ending, at, "7"))
                    vsc_ending = None
                if red_start is not None:
                    intervals.append((red_start, at, "5"))
                    red_start = None
            elif flag == "RED":
                red_start = red_start or at
                # A red flag ends any neutralisation in progress; the race restarts later
                # under whatever procedure race control announces.
                if sc_start is not None:
                    intervals.append((sc_start, at, "4"))
                    sc_start, sc_in_this_lap = None, False
                if vsc_start is not None:
                    intervals.append((vsc_start, at, "6"))
                    vsc_start = None
                if vsc_ending is not None:
                    intervals.append((vsc_ending, at, "7"))
                    vsc_ending = None
        if category == "SessionStatus" and ("ABORTED" in message or "SUSPENDED" in message):
            # The feed reports a stoppage as an aborted session, with or without a RED flag
            # message; either way the race is red-flagged until it restarts.
            red_start = red_start or at
            if sc_start is not None:
                intervals.append((sc_start, at, "4"))
                sc_start, sc_in_this_lap = None, False
            if vsc_start is not None:
                intervals.append((vsc_start, at, "6"))
                vsc_start = None
            if vsc_ending is not None:
                intervals.append((vsc_ending, at, "7"))
                vsc_ending = None
        if category == "SessionStatus" and ("RESUMED" in message or "STARTED" in message) and red_start is not None:
            intervals.append((red_start, at, "5"))
            red_start = None
        elif category == "SafetyCar":
            if "VIRTUAL" in message or message.startswith("VSC"):
                if "DEPLOYED" in message:
                    vsc_start = vsc_start or at
                elif "ENDING" in message and vsc_start is not None:
                    intervals.append((vsc_start, at, "6"))
                    vsc_start, vsc_ending = None, at
            elif "DEPLOYED" in message:
                sc_start = sc_start or at
                sc_in_this_lap = False
            elif "IN THIS LAP" in message:
                sc_in_this_lap = True
        elif category == "SessionStatus" and "STARTED" in message and race_starts_behind_sc:
            # A race that starts (or resumes) behind the safety car: the feed announces it
            # before the start and never sends "SAFETY CAR DEPLOYED".
            sc_start = sc_start or at
            sc_in_this_lap = False
            race_starts_behind_sc = False
        elif category == "Other" and "BEHIND THE SAFETY CAR" in message:
            race_starts_behind_sc = True
        elif category == "Other" and "ROLLING START" in message and sc_start is not None:
            # The safety car peels in and the race starts rolling: the period ends here.
            intervals.append((sc_start, at, "4"))
            sc_start, sc_in_this_lap = None, False
    for start in open_yellow.values():
        intervals.append((start, session_end, "2"))
    if sc_start is not None:
        intervals.append((sc_start, session_end, "4"))
    if vsc_start is not None:
        intervals.append((vsc_start, session_end, "6"))
    if vsc_ending is not None:
        intervals.append((vsc_ending, session_end, "7"))
    if red_start is not None:
        intervals.append((red_start, session_end, "5"))
    # FastF1's status feed reports the neutralisation, not the sector yellows shown under
    # it: a lap under a safety car is "4", never "24". Drop yellow spans inside SC/VSC/red.
    blocking = [(s, e) for s, e, code in intervals if code in ("4", "5", "6", "7")]
    kept: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for start, end, code in intervals:
        if code != "2":
            kept.append((start, end, code))
            continue
        pieces = [(start, end)]
        for b_start, b_end in blocking:
            pieces = [seg for a, b in pieces for seg in ((a, min(b, b_start)), (max(a, b_end), b)) if seg[0] < seg[1]]
        kept.extend((a, b, "2") for a, b in pieces)
    return kept


def lap_track_status(lap_start: pd.Series, lap_end: pd.Series, intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]]) -> pd.Series:
    """FastF1-style status string per lap: every code active at any moment of the lap, with
    ``1`` included when some part of the lap ran under a clear track (FastF1 writes ``"12"``
    for a lap that started green and met a yellow)."""
    codes = [set() for _ in range(len(lap_start))]
    spans: list[list[tuple[np.datetime64, np.datetime64]]] = [[] for _ in range(len(lap_start))]
    start = lap_start.to_numpy(dtype="datetime64[ns]")
    end = lap_end.to_numpy(dtype="datetime64[ns]")
    length = (end - start) / np.timedelta64(1, "s")
    for s, e, code in intervals:
        s64, e64 = np.datetime64(s.tz_convert("UTC").tz_localize(None)), np.datetime64(e.tz_convert("UTC").tz_localize(None))
        lo, hi = np.maximum(start, s64), np.minimum(end, e64)
        # An interval that ends exactly when the lap starts (or starts when it ends) did not
        # run during the lap: only a strictly positive overlap counts.
        for i in np.flatnonzero(lo < hi):
            codes[i].add(code)
            spans[i].append((lo[i], hi[i]))
    out = []
    for i, c in enumerate(codes):
        if not c or not np.isfinite(length[i]) or _union_seconds(spans[i]) < length[i] - 0.5:
            c.add("1")
        out.append("".join(code for code in STATUS_ORDER if code in c) or "1")
    return pd.Series(out, index=lap_start.index)


def _union_seconds(spans: list[tuple[np.datetime64, np.datetime64]]) -> float:
    """Length of the union of (start, end) spans in seconds; back-to-back neutralisations
    (a safety car handed over to a virtual one) cover a lap together."""
    total = 0.0
    current_lo = current_hi = None
    for lo, hi in sorted(spans, key=lambda span: span[0]):
        if current_hi is None or lo > current_hi:
            if current_hi is not None:
                total += float((current_hi - current_lo) / np.timedelta64(1, "s"))
            current_lo, current_hi = lo, hi
        elif hi > current_hi:
            current_hi = hi
    if current_hi is not None:
        total += float((current_hi - current_lo) / np.timedelta64(1, "s"))
    return total


_DELETED_ANY = re.compile(r"CAR (\d+) \((\w{3})\) (?:LAP|TIME [\d:.]+) DELETED")


def deleted_laps(race_control: list[dict], laps: pd.DataFrame, lap_end: pd.Series | None = None) -> pd.Series:
    """Mark laps whose time was deleted by the stewards (track limits). A message that quotes
    the lap time identifies the lap exactly; one that quotes a lap number uses it; otherwise
    the driver's lap in progress when the message was issued is taken."""
    deleted = pd.Series(False, index=laps.index)
    for e in race_control:
        message = str(e.get("message") or "")
        m = _DELETED_TIME.search(message)
        if m:
            seconds = int(m.group(3)) * 60 + float(m.group(4))
            deleted |= (laps["driver_number"] == int(m.group(1))) & (np.abs(laps["lap_time_s"] - seconds) < 0.002)
            continue
        m = _DELETED_LAP.search(message)
        if m:
            deleted |= (laps["driver_number"] == int(m.group(1))) & (laps["lap_number"] == int(m.group(3)))
            continue
        m = _DELETED_ANY.search(message)
        if m and e.get("date") and lap_end is not None:
            at = pd.Timestamp(e["date"])
            at = at.tz_localize("UTC") if at.tzinfo is None else at
            mine = (laps["driver_number"] == int(m.group(1))) & (laps["date_start"] <= at)
            if mine.any():
                deleted[laps[mine].index[-1]] = True
    return deleted


def session_frame(client: OpenF1Client, meeting: dict, session: dict) -> pd.DataFrame:
    """Every lap of one session in the RaceShift schema."""
    key = int(session["session_key"])
    year = int(session.get("year") or meeting.get("year"))
    laps_raw = client.get("laps", session_key=key)
    if not laps_raw:
        return pd.DataFrame()
    drivers = {int(d["driver_number"]): d for d in client.get("drivers", session_key=key)}
    stints = client.get("stints", session_key=key)
    pits = client.get("pit", session_key=key)
    weather = client.get("weather", session_key=key)
    race_control = client.get("race_control", session_key=key)
    positions = client.get("position", session_key=key)

    laps = pd.DataFrame(laps_raw)
    laps["driver_number"] = laps["driver_number"].astype(int)
    laps["lap_number"] = pd.to_numeric(laps["lap_number"], errors="coerce")
    laps = laps.dropna(subset=["lap_number"]).sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    laps["lap_time_s"] = pd.to_numeric(laps.get("lap_duration"), errors="coerce")
    laps["date_start"] = _ts(laps["date_start"])
    # A lap without a duration (the lap the car retired on, or a red-flag lap) ends when the
    # next lap of the same driver starts; the last lap of a driver ends at the session end.
    next_start = laps.groupby("driver_number")["date_start"].shift(-1)
    session_end = pd.Timestamp(session.get("date_end")) if session.get("date_end") else laps["date_start"].max() + pd.Timedelta(minutes=5)
    if session_end.tzinfo is None:
        session_end = session_end.tz_localize("UTC")
    lap_end = laps["date_start"] + pd.to_timedelta(laps["lap_time_s"], unit="s")
    lap_end = lap_end.where(lap_end.notna(), next_start).fillna(session_end)

    pit_laps = {(int(p["driver_number"]), int(p["lap_number"])) for p in pits if p.get("lap_number") is not None}
    laps["pit_in"] = [(d, int(n)) in pit_laps for d, n in zip(laps["driver_number"], laps["lap_number"])]
    laps["pit_out"] = laps.get("is_pit_out_lap", pd.Series(False, index=laps.index)).fillna(False).astype(bool)

    stint_rows = pd.DataFrame(stints) if stints else pd.DataFrame(columns=["driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"])
    compound = pd.Series(None, index=laps.index, dtype="object")
    stint = pd.Series(np.nan, index=laps.index)
    tyre_life = pd.Series(np.nan, index=laps.index)
    fresh = pd.Series(np.nan, index=laps.index)
    for _, s in stint_rows.iterrows():
        if pd.isna(s.get("lap_start")) or pd.isna(s.get("lap_end")):
            continue
        mask = (laps["driver_number"] == int(s["driver_number"])) & (laps["lap_number"] >= int(s["lap_start"])) & (laps["lap_number"] <= int(s["lap_end"]))
        compound[mask] = s.get("compound")
        stint[mask] = float(s["stint_number"])
        age0 = float(s["tyre_age_at_start"]) if pd.notna(s.get("tyre_age_at_start")) else 0.0
        tyre_life[mask] = age0 + (laps.loc[mask, "lap_number"] - int(s["lap_start"])) + 1
        fresh[mask] = 1.0 if age0 == 0 else 0.0

    # Position at the end of the lap: the last update at or before the lap's end time.
    if positions:
        pos = pd.DataFrame(positions)
        pos["date"] = _ts(pos["date"])
        pos["driver_number"] = pos["driver_number"].astype(int)
        pos = pos.dropna(subset=["date"]).sort_values("date")
        probe = pd.DataFrame({"driver_number": laps["driver_number"], "at": lap_end}).reset_index()
        probe = probe.dropna(subset=["at"]).sort_values("at")
        merged = pd.merge_asof(probe, pos[["date", "driver_number", "position"]].rename(columns={"date": "at"}), on="at", by="driver_number", direction="backward")
        position = merged.set_index("index")["position"].reindex(laps.index)
    else:
        position = pd.Series(np.nan, index=laps.index)

    # Weather nearest the start of the lap.
    wx = pd.DataFrame(weather) if weather else pd.DataFrame()
    weather_cols = {}
    if not wx.empty:
        wx["date"] = _ts(wx["date"])
        wx = wx.dropna(subset=["date"]).sort_values("date")
        probe = pd.DataFrame({"at": laps["date_start"]}).reset_index().dropna(subset=["at"]).sort_values("at")
        merged = pd.merge_asof(probe, wx.rename(columns={"date": "at"}), on="at", direction="nearest", tolerance=pd.Timedelta(minutes=10))
        merged = merged.set_index("index").reindex(laps.index)
        for src, dst in WEATHER_COLUMNS.items():
            if src in merged.columns:
                weather_cols[dst] = merged[src]
    for dst in WEATHER_COLUMNS.values():
        weather_cols.setdefault(dst, pd.Series(np.nan, index=laps.index, dtype=float))

    intervals = track_status_intervals(race_control, session_end)
    status = lap_track_status(laps["date_start"], lap_end, intervals)
    sectors = [pd.to_numeric(laps.get(f"duration_sector_{i}"), errors="coerce") for i in (1, 2, 3)]
    sector_sum = sectors[0] + sectors[1] + sectors[2]
    clear = ~status.str.contains(r"[4567]", regex=True)
    # OpenF1's pit feed occasionally lacks a stop (typically a driver's final in-lap). A lap
    # under a clear track with no pit flags that is far slower than the driver's own clean
    # laps is an in-lap or a damaged lap, not an accurate racing lap; FastF1 marks the same
    # laps inaccurate through its pit markers.
    clean = clear & ~laps["pit_in"] & ~laps["pit_out"] & laps["lap_time_s"].notna()
    reference = laps["lap_time_s"].where(clean).groupby(laps["driver_number"]).transform("median")
    too_slow = clean & (laps["lap_time_s"] > SLOW_LAP_RATIO * reference)
    is_accurate = (
        (laps["lap_number"] > 1)
        & laps["lap_time_s"].notna()
        & sector_sum.notna()
        & ((sector_sum - laps["lap_time_s"]).abs() <= 0.05)
        & ~laps["pit_in"]
        & ~laps["pit_out"]
        & clear
        & ~too_slow
    )

    event_date = str(pd.Timestamp(meeting.get("date_end") or session.get("date_start")).date())
    location = str(meeting.get("location") or session.get("location") or "")
    out = pd.DataFrame({
        "season": year,
        "series": "F1",
        "round_number": int(meeting["round_number"]),
        "event": str(meeting.get("meeting_name")),
        "circuit": LOCATION_ALIASES.get(location, location),
        "event_date": event_date,
        "session": session["session_code"],
        "driver": [str(drivers.get(d, {}).get("name_acronym") or f"#{d}") for d in laps["driver_number"]],
        "team": [drivers.get(d, {}).get("team_name") for d in laps["driver_number"]],
        "lap_number": laps["lap_number"].astype(float),
        "lap_time_s": laps["lap_time_s"].astype(float),
        "sector1_s": sectors[0].astype(float),
        "sector2_s": sectors[1].astype(float),
        "sector3_s": sectors[2].astype(float),
        "compound": compound,
        "tyre_life": tyre_life.astype(float),
        "fresh_tyre": fresh.fillna(0).astype(bool),
        "tyre_manufacturer": "Pirelli",
        "stint": stint.astype(float),
        "position": pd.to_numeric(position, errors="coerce").astype(float),
        "track_status": status.astype(str),
        "pit_in": laps["pit_in"].astype(bool),
        "pit_out": laps["pit_out"].astype(bool),
        "is_accurate": is_accurate.astype(bool),
        "deleted": deleted_laps(race_control, laps, lap_end).astype(bool),
        "data_tier": DATA_TIER,
        "air_temp_c": pd.to_numeric(weather_cols["air_temp_c"], errors="coerce"),
        "track_temp_c": pd.to_numeric(weather_cols["track_temp_c"], errors="coerce"),
        "humidity_pct": pd.to_numeric(weather_cols["humidity_pct"], errors="coerce"),
        "pressure_mbar": pd.to_numeric(weather_cols["pressure_mbar"], errors="coerce"),
        "rainfall": pd.to_numeric(weather_cols["rainfall"], errors="coerce").fillna(0).astype(float) > 0,
        "wind_speed_ms": pd.to_numeric(weather_cols["wind_speed_ms"], errors="coerce"),
        "wind_direction_deg": pd.to_numeric(weather_cols["wind_direction_deg"], errors="coerce").fillna(0).astype(int),
        "driver_number": laps["driver_number"].astype(int),
        "session_key": key,
    })
    return out


def export_session(client: OpenF1Client, meeting: dict, session: dict, output_dir: str | Path) -> Path | None:
    frame = session_frame(client, meeting, session)
    if frame.empty:
        return None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{frame['season'].iloc[0]}_{str(meeting['meeting_name']).replace(' ', '_')}_{session['session_code']}.parquet"
    frame.to_parquet(path, index=False)
    return path
