from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_delta_vs_rolling5_s"
RAW_TARGET_COLUMN = "target_next_lap_time_s"

BASE_NUMERIC_FEATURES = [
    "lap_number",
    "lap_time_s",
    "sector1_s",
    "sector2_s",
    "sector3_s",
    "tyre_life",
    "fresh_tyre",
    "stint",
    "position",
    "air_temp_c",
    "track_temp_c",
    "humidity_pct",
    "pressure_mbar",
    "rainfall",
    "wind_speed_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "gap_ahead_s",
    "gap_behind_s",
    "mean_speed_kph",
    "max_speed_kph",
    "mean_throttle_pct",
    "brake_fraction",
    "mean_rpm",
    "max_rpm",
    "gear_changes",
    "drs_fraction",
    "rolling_median_3",
    "rolling_median_5",
    "sector1_mean_3",
    "sector2_mean_3",
    "sector3_mean_3",
    "pace_trend_3",
    "lap_delta_to_rolling5",
    "hist_driver_circuit_pace",
    "hist_team_circuit_pace",
    "hist_compound_circuit_pace",
    "hist_driver_circuit_compound_pace",
    "hist_team_circuit_compound_pace",
    "hist_driver_global_pace",
    "hist_team_global_pace",
    "hist_weather_compound_pace",
    "hist_driver_weather_pace",
    "hist_team_weather_pace",
]

CATEGORICAL_FEATURES = [
    "series",
    "event",
    "circuit",
    "session",
    "driver",
    "team",
    "manufacturer",
    "car_class",
    "compound",
    "tyre_manufacturer",
    "track_status",
]

LAG_SOURCE_FEATURES = [
    "lap_time_s",
    "sector1_s",
    "sector2_s",
    "sector3_s",
    "tyre_life",
    "track_temp_c",
    "air_temp_c",
    "wind_speed_ms",
    "gap_ahead_s",
    "mean_speed_kph",
    "mean_throttle_pct",
    "brake_fraction",
]

LEAKAGE_BLOCKLIST = {
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    "next_lap_number",
    "next_sector1_s",
    "next_sector2_s",
    "next_sector3_s",
    "next_mean_speed_kph",
    "next_max_speed_kph",
}


@dataclass(frozen=True)
class FeatureContract:
    numeric: list[str]
    categorical: list[str]
    target: str = TARGET_COLUMN
    raw_target: str = RAW_TARGET_COLUMN


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "series": "F1",
        "circuit": None,
        "team": None,
        "manufacturer": None,
        "car_class": None,
        "compound": None,
        "tyre_manufacturer": "Pirelli",
        "track_status": None,
        "fresh_tyre": np.nan,
        "rainfall": False,
        "wind_direction_deg": np.nan,
        "gap_ahead_s": np.nan,
        "gap_behind_s": np.nan,
        "mean_speed_kph": np.nan,
        "max_speed_kph": np.nan,
        "mean_throttle_pct": np.nan,
        "brake_fraction": np.nan,
        "mean_rpm": np.nan,
        "max_rpm": np.nan,
        "gear_changes": np.nan,
        "drs_fraction": np.nan,
        "is_accurate": True,
        "deleted": False,
        "pit_in": False,
        "pit_out": False,
    }
    out = df.copy()
    if "circuit" not in out and "event" in out:
        out["circuit"] = out["event"]
    for col, value in defaults.items():
        if col not in out:
            out[col] = value
    return out


def _event_order(df: pd.DataFrame) -> pd.Series:
    """Create a deterministic event index without using future outcomes."""
    if "event_date" in df:
        dates = pd.to_datetime(df["event_date"], errors="coerce")
        fallback = pd.to_datetime(df["season"].astype(str) + "-01-01", errors="coerce")
        return dates.fillna(fallback)
    if "round_number" in df:
        return pd.to_datetime(df["season"].astype(str) + "-01-01") + pd.to_timedelta(
            pd.to_numeric(df["round_number"], errors="coerce").fillna(0), unit="D"
        )
    # Input collectors write events in chronological order. The categorical event name
    # is never sorted alphabetically to manufacture a false chronology.
    event_key = df["season"].astype(str) + "|" + df["event"].astype(str)
    event_codes = pd.factorize(event_key, sort=False)[0]
    return pd.to_datetime(df["season"].astype(str) + "-01-01") + pd.to_timedelta(event_codes, unit="D")


def _prior_event_expanding_median(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    """Median from strictly earlier events, never rows from the current event."""
    event_cols = ["season", "event"]
    event_summary = (
        df.groupby(group_cols + event_cols, dropna=False, sort=False)[value_col]
        .median()
        .reset_index(name="event_median")
    )
    event_summary["_event_order"] = _event_order(event_summary)
    event_summary = event_summary.sort_values(group_cols + ["_event_order"], kind="stable")

    event_summary["prior"] = np.nan
    grouped_indices = event_summary.groupby(group_cols, dropna=False, sort=False).groups
    for _, indices in grouped_indices.items():
        history: list[float] = []
        for idx in list(indices):
            value = float(event_summary.at[idx, "event_median"])
            if history:
                event_summary.at[idx, "prior"] = float(np.nanmedian(history))
            if not np.isnan(value):
                history.append(value)
    merge_cols = group_cols + event_cols
    lookup = event_summary[merge_cols + ["prior"]]
    key_frame = df[merge_cols].copy()
    key_frame["_row"] = np.arange(len(df))
    merged = key_frame.merge(lookup, on=merge_cols, how="left", sort=False).sort_values("_row")
    return merged["prior"].reset_index(drop=True)


def _weather_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["track_temp_bin"] = (pd.to_numeric(out["track_temp_c"], errors="coerce") / 5.0).round() * 5.0
    out["air_temp_bin"] = (pd.to_numeric(out["air_temp_c"], errors="coerce") / 4.0).round() * 4.0
    out["humidity_bin"] = (pd.to_numeric(out["humidity_pct"], errors="coerce") / 10.0).round() * 10.0
    out["wind_bin"] = (pd.to_numeric(out["wind_speed_ms"], errors="coerce") / 2.0).round() * 2.0
    out["rain_state"] = out["rainfall"].fillna(False).astype(bool).astype(int)
    return out


def build_full_context_table(raw: pd.DataFrame, history: int = 5) -> pd.DataFrame:
    """Build the canonical leakage-safe lap-N -> lap-(N+1) forecasting table."""
    if history < 1:
        raise ValueError("history must be >= 1")
    df = _ensure_columns(raw)

    required = ["season", "event", "session", "driver", "lap_number", "lap_time_s"]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sort_cols = [c for c in ["season", "round_number", "event_date", "event", "session", "driver", "lap_number"] if c in df]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    df = df[df["is_accurate"].fillna(False).astype(bool) & ~df["deleted"].fillna(False).astype(bool)]
    df = df[~df["pit_in"].fillna(False).astype(bool) & ~df["pit_out"].fillna(False).astype(bool)].copy()
    df = df[pd.to_numeric(df["lap_time_s"], errors="coerce").notna()].reset_index(drop=True)

    # Circular wind representation.
    wind_deg = pd.to_numeric(df["wind_direction_deg"], errors="coerce")
    radians = np.deg2rad(wind_deg)
    df["wind_dir_sin"] = np.sin(radians)
    df["wind_dir_cos"] = np.cos(radians)

    group_cols = ["season", "event", "session", "driver"]
    group = df.groupby(group_cols, dropna=False, sort=False)
    df["rolling_median_3"] = group["lap_time_s"].transform(lambda s: s.rolling(3, min_periods=1).median())
    df["rolling_median_5"] = group["lap_time_s"].transform(lambda s: s.rolling(5, min_periods=1).median())
    for sector in ["sector1_s", "sector2_s", "sector3_s"]:
        if sector not in df:
            df[sector] = np.nan
        df[f"{sector.replace('_s', '')}_mean_3"] = group[sector].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["pace_trend_3"] = group["lap_time_s"].transform(lambda s: s.diff().rolling(3, min_periods=1).mean())
    df["lap_delta_to_rolling5"] = df["lap_time_s"] - df["rolling_median_5"]

    # Historical priors are calculated from event-level medians shifted by one event.
    prior_specs = [
        (["driver", "circuit"], "hist_driver_circuit_pace"),
        (["team", "circuit"], "hist_team_circuit_pace"),
        (["compound", "circuit"], "hist_compound_circuit_pace"),
        (["driver", "circuit", "compound"], "hist_driver_circuit_compound_pace"),
        (["team", "circuit", "compound"], "hist_team_circuit_compound_pace"),
        (["driver"], "hist_driver_global_pace"),
        (["team"], "hist_team_global_pace"),
    ]
    for groups, target_name in prior_specs:
        df[target_name] = _prior_event_expanding_median(df, groups, "lap_time_s")

    weather = _weather_bins(df)
    for c in ["track_temp_bin", "air_temp_bin", "humidity_bin", "wind_bin", "rain_state"]:
        df[c] = weather[c]
    weather_groups = ["circuit", "compound", "track_temp_bin", "air_temp_bin", "humidity_bin", "wind_bin", "rain_state"]
    df["hist_weather_compound_pace"] = _prior_event_expanding_median(df, weather_groups, "lap_time_s")
    df["hist_driver_weather_pace"] = _prior_event_expanding_median(df, ["driver"] + weather_groups, "lap_time_s")
    df["hist_team_weather_pace"] = _prior_event_expanding_median(df, ["team"] + weather_groups, "lap_time_s")

    # Explicit flattened temporal context. Lag 0 is represented by base features.
    group = df.groupby(group_cols, dropna=False, sort=False)
    for source in LAG_SOURCE_FEATURES:
        if source not in df:
            df[source] = np.nan
        for lag in range(1, history):
            df[f"{source}_lag{lag}"] = group[source].shift(lag)

    # Target is strictly the adjacent next completed lap.
    df[RAW_TARGET_COLUMN] = group["lap_time_s"].shift(-1)
    df["next_lap_number"] = group["lap_number"].shift(-1)
    df = df[(df["next_lap_number"] - df["lap_number"]) == 1].copy()
    df = df[df[RAW_TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = df[RAW_TARGET_COLUMN] - df["rolling_median_5"]
    return df.reset_index(drop=True)


def feature_contract(history: int = 5) -> FeatureContract:
    lag_features = [f"{source}_lag{lag}" for source in LAG_SOURCE_FEATURES for lag in range(1, history)]
    return FeatureContract(numeric=BASE_NUMERIC_FEATURES + lag_features, categorical=CATEGORICAL_FEATURES)


def assert_no_target_leakage(columns: Iterable[str]) -> None:
    bad = sorted(set(columns).intersection(LEAKAGE_BLOCKLIST))
    if bad:
        raise ValueError(f"Leakage columns present: {bad}")
