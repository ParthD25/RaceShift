from __future__ import annotations

from pathlib import Path

import pandas as pd


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def export_session(year: int, event: str | int, session_name: str, output_dir: str | Path, cache_dir: str | Path) -> Path:
    """Fetch one FastF1 session and export model-ready lap-level rows.

    This intentionally exports only values available at lap completion. The training
    pipeline later shifts the target so lap N features predict lap N+1.
    """
    import fastf1

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    session = fastf1.get_session(year, event, session_name)
    session.load(telemetry=False, weather=True, messages=True)
    laps = session.laps.copy()

    out = pd.DataFrame({
        "season": year,
        "series": "F1",
        "event": session.event.EventName,
        "circuit": session.event.EventName,
        "event_date": str(session.event.EventDate.date()) if hasattr(session.event.EventDate, "date") else str(session.event.EventDate),
        "session": session_name,
        "driver": laps["Driver"].astype(str),
        "team": laps.get("Team", pd.Series(index=laps.index, dtype="object")).astype(str),
        "lap_number": pd.to_numeric(laps["LapNumber"], errors="coerce"),
        "lap_time_s": _seconds(laps["LapTime"]),
        "sector1_s": _seconds(laps["Sector1Time"]),
        "sector2_s": _seconds(laps["Sector2Time"]),
        "sector3_s": _seconds(laps["Sector3Time"]),
        "compound": laps.get("Compound"),
        "tyre_life": pd.to_numeric(laps.get("TyreLife"), errors="coerce"),
        "fresh_tyre": pd.to_numeric(laps.get("FreshTyre"), errors="coerce"),
        "tyre_manufacturer": "Pirelli",
        "stint": pd.to_numeric(laps.get("Stint"), errors="coerce"),
        "position": pd.to_numeric(laps.get("Position"), errors="coerce"),
        "track_status": laps.get("TrackStatus").astype(str),
        "pit_in": laps.get("PitInTime").notna(),
        "pit_out": laps.get("PitOutTime").notna(),
        "is_accurate": laps.get("IsAccurate", True),
        "deleted": laps.get("Deleted", False),
    })

    # FastF1 exposes weather as separate samples. For phase 1 we merge each lap with
    # the nearest weather observation by session time when both timestamps exist.
    if session.weather_data is not None and not session.weather_data.empty and "Time" in laps:
        weather = session.weather_data.copy().sort_values("Time")
        lap_weather = laps[["Time"]].copy()
        lap_weather["_row_order"] = range(len(lap_weather))
        lap_weather = lap_weather.sort_values("Time")
        merged = pd.merge_asof(lap_weather, weather, on="Time", direction="nearest")
        merged = merged.sort_values("_row_order").reset_index(drop=True)
        mapping = {
            "AirTemp": "air_temp_c",
            "TrackTemp": "track_temp_c",
            "Humidity": "humidity_pct",
            "Pressure": "pressure_mbar",
            "Rainfall": "rainfall",
            "WindSpeed": "wind_speed_ms",
            "WindDirection": "wind_direction_deg",
        }
        for src, dst in mapping.items():
            if src in merged:
                out[dst] = merged[src].values

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{year}_{str(session.event.EventName).replace(' ', '_')}_{session_name}.parquet"
    out.to_parquet(path, index=False)
    return path
