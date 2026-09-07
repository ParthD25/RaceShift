from __future__ import annotations

import numpy as np
import pandas as pd

LEAKAGE_BLOCKLIST = {
    "target_next_lap_time_s",
    "next_lap_time_s",
    "next_sector1_s",
    "next_sector2_s",
    "next_sector3_s",
}

NUMERIC_FEATURES = [
    "lap_number",
    "lap_time_s",
    "sector1_s",
    "sector2_s",
    "sector3_s",
    "tyre_life",
    "stint",
    "position",
    "air_temp_c",
    "track_temp_c",
    "humidity_pct",
    "pressure_mbar",
    "wind_speed_ms",
    "rolling_median_3",
    "rolling_median_5",
    "pace_trend_3",
    "lap_delta_to_rolling5",
]


def build_next_lap_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a leakage-safe next-lap forecasting table.

    Every feature row is information available after lap N. The target is lap N+1.
    Pit-in/out rows and clearly inaccurate/deleted laps are excluded from the first
    forecasting benchmark because they represent a different process.
    """
    df = raw.copy()
    sort_cols = [c for c in ["season", "event", "session", "driver", "lap_number"] if c in df]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for c in ["is_accurate", "deleted"]:
        if c not in df:
            df[c] = True if c == "is_accurate" else False

    df = df[(df["is_accurate"].fillna(False)) & (~df["deleted"].fillna(False))].copy()
    pit_in = df["pit_in"].fillna(False) if "pit_in" in df else pd.Series(False, index=df.index)
    pit_out = df["pit_out"].fillna(False) if "pit_out" in df else pd.Series(False, index=df.index)
    df = df[(~pit_in) & (~pit_out)].copy()

    group = df.groupby(["season", "event", "session", "driver"], dropna=False, sort=False)
    df["rolling_median_3"] = group["lap_time_s"].transform(lambda s: s.rolling(3, min_periods=1).median())
    df["rolling_median_5"] = group["lap_time_s"].transform(lambda s: s.rolling(5, min_periods=1).median())
    df["pace_trend_3"] = group["lap_time_s"].transform(lambda s: s.diff().rolling(3, min_periods=1).mean())
    df["lap_delta_to_rolling5"] = df["lap_time_s"] - df["rolling_median_5"]

    df["target_next_lap_time_s"] = group["lap_time_s"].shift(-1)
    df["next_lap_number"] = group["lap_number"].shift(-1)

    # Require adjacency so a missing/pit lap cannot silently become the prediction target.
    df = df[(df["next_lap_number"] - df["lap_number"]) == 1].copy()
    df = df[df["target_next_lap_time_s"].notna()].copy()

    # A robust relative target is useful for cross-circuit training; raw target is kept
    # for direct evaluation in seconds.
    df["target_delta_vs_rolling5_s"] = df["target_next_lap_time_s"] - df["rolling_median_5"]
    return df


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    bad = LEAKAGE_BLOCKLIST.intersection(feature_columns)
    if bad:
        raise ValueError(f"Target leakage columns present in features: {sorted(bad)}")


def make_history_windows(df: pd.DataFrame, feature_columns: list[str], history: int = 5):
    """Return fixed-length driver/event histories for deep-learning models."""
    assert_no_target_leakage(feature_columns)
    X, y, metadata = [], [], []

    group_cols = ["season", "event", "session", "driver"]
    for key, g in df.groupby(group_cols, sort=False, dropna=False):
        g = g.sort_values("lap_number").reset_index(drop=True)
        values = g[feature_columns].replace([np.inf, -np.inf], np.nan)
        values = values.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        target = g["target_delta_vs_rolling5_s"].to_numpy(dtype=np.float32)
        raw_target = g["target_next_lap_time_s"].to_numpy(dtype=np.float32)

        for i in range(history - 1, len(g)):
            X.append(values[i - history + 1 : i + 1])
            y.append([target[i], raw_target[i], g.loc[i, "rolling_median_5"]])
            metadata.append({"season": key[0], "event": key[1], "session": key[2], "driver": key[3], "lap_number": int(g.loc[i, "lap_number"])})

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), metadata
