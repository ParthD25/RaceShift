"""Leakage-safe full-context feature table for next-lap forecasting.

Every feature attached to lap N is computed from information available at the end of
lap N. Lags and rolling statistics only span *consecutive valid racing laps*: a pit-in,
pit-out, safety-car, virtual-safety-car, red-flag, deleted or inaccurate lap breaks the
sequence, so "previous lap" can never silently mean "the lap before a pit stop three
laps ago". Historical priors use strictly earlier events.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_delta_vs_rolling5_s"
RAW_TARGET_COLUMN = "target_next_lap_time_s"

GROUP_COLUMNS = ["season", "event", "session", "driver"]

# --- feature taxonomy -------------------------------------------------------------------
# Static / identity context (categorical, one-hot encoded downstream).
STATIC_CATEGORICAL = [
    "series",
    "event",
    "circuit",
    "session",
    "driver",
    "team",
    "driver_team",
    "manufacturer",
    "car_class",
    "compound",
    "tyre_manufacturer",
    "data_tier",  # fastf1_timing (2018+) vs legacy_timing (Ergast 1996-2017, no sectors/tyres/weather)
]

# Dynamic race state known at the end of lap N.
DYNAMIC_NUMERIC = [
    "lap_number",
    "sector1_rel_3",
    "sector2_rel_3",
    "sector3_rel_3",
    "tyre_life",
    "fresh_tyre",
    "stint",
    "laps_in_segment",
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
]

# Recent-pace statistics over the current run of consecutive valid laps.
TEMPORAL_NUMERIC = [
    "rolling_median_5",
    "pace_spread_3_5",
    "sector1_share_3",
    "sector2_share_3",
    "sector3_share_3",
    "pace_trend_3",
    "lap_delta_to_rolling5",
]

# Historical priors from strictly earlier events.
HISTORICAL_PRIORS = [
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
# Priors enter the model relative to the current rolling pace, so they are comparable
# across circuits and seasons ("historically this driver is 0.3 s quicker here than the
# current run suggests"), not as absolute lap times.
HISTORICAL_NUMERIC = [f"{name}_rel" for name in HISTORICAL_PRIORS]

BASE_NUMERIC_FEATURES = DYNAMIC_NUMERIC + TEMPORAL_NUMERIC + HISTORICAL_NUMERIC
CATEGORICAL_FEATURES = STATIC_CATEGORICAL

# Lag sources measured in lap-time seconds are stored relative to the current rolling pace
# (lap time) or the current 3-lap sector mean (sectors); the rest keep their own units.
PACE_SCALE_LAG_SOURCES = {"lap_time_s": "rolling_median_5", "sector1_s": "sector1_mean_3", "sector2_s": "sector2_mean_3", "sector3_s": "sector3_mean_3"}

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

# Lap-state columns. They are used to decide which laps are valid pace laps and are kept
# on the table for analysis, but they are not model inputs for the pure race-pace model.
LAP_STATE_COLUMNS = [
    "lap_valid",
    "is_safety_car",
    "is_vsc",
    "is_yellow",
    "is_red_flag",
    "is_red_flag_restart",
    "is_safety_car_restart",
    "is_pit_in",
    "is_pit_out",
    "is_deleted",
]

# Version of the lap-validity rules above. Recorded in every artifact and metrics file so
# results produced under different rules are never compared silently.
#   1  accurate, timed, not deleted, not pit-in/out, not SC/VSC/red-flag lap
#   2  v1 plus: the first timed lap after a red-flag stoppage (restart lap) is invalid
#   3  v2 plus: yellow-flag laps (status code 2) are invalid, and so is the first timed lap
#      after a safety-car period (the SC restart lap). Measured on 2018-2026 races before the
#      change: yellow laps were 3.6% of valid laps with 8.7% of them more than 5% slower than
#      the driver's race median (1.9% for all laps) and 4.1% more than 15% slower; the lap
#      after a safety car was 2.2% slower at the median and 17% of them more than 5% slower.
#      The lap after a VSC (median ratio 1.000) and the second lap after a safety car
#      (median 1.010) stay valid.
LAP_VALIDITY_VERSION = 3

LEAKAGE_BLOCKLIST = {
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    "next_lap_number",
    "next_lap_valid",
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
        "data_tier": "fastf1_timing",
        "track_status": "1",
        "fresh_tyre": np.nan,
        "tyre_life": np.nan,
        "stint": np.nan,
        "position": np.nan,
        "air_temp_c": np.nan,
        "track_temp_c": np.nan,
        "humidity_pct": np.nan,
        "pressure_mbar": np.nan,
        "wind_speed_ms": np.nan,
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
    # Files exported before the tier column existed are FastF1 timing by construction.
    out["data_tier"] = out["data_tier"].fillna("fastf1_timing")
    return out


def _flag(series: pd.Series) -> pd.Series:
    """Coerce a possibly-missing boolean-ish column to a clean bool Series."""
    if series.dtype == bool:
        return series
    return series.map(lambda v: bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False).astype(bool)


def _restart_lap_after(df: pd.DataFrame, flagged: pd.Series, has_time: pd.Series) -> pd.Series:
    """Flag the first timed, unflagged lap after one or more ``flagged`` laps within each
    driver/session: the lap on which racing resumes after a stoppage or a safety-car period."""
    if not all(c in df for c in GROUP_COLUMNS) or "lap_number" not in df:
        return pd.Series(False, index=df.index)
    order = df.assign(_lap=pd.to_numeric(df["lap_number"], errors="coerce")).sort_values(
        GROUP_COLUMNS + ["_lap"], kind="stable"
    ).index
    flag = flagged.loc[order]
    timed = has_time.loc[order]
    keys = [df.loc[order, c] for c in GROUP_COLUMNS]
    count = flag.groupby(keys, dropna=False, sort=False).cumsum()
    candidate = timed & ~flag
    # Flag count seen by the previous timed, unflagged lap of the same driver/session.
    previous = count.where(candidate).groupby(keys, dropna=False, sort=False).transform(
        lambda s: s.ffill().shift(1)
    ).fillna(0)
    restart = candidate & (count > previous)
    return restart.reindex(df.index).astype(bool)


def _red_flag_restart(df: pd.DataFrame, is_red_flag: pd.Series, has_time: pd.Series) -> pd.Series:
    """Flag the first timed lap after a red-flag stoppage within each driver/session.

    FastF1 marks the stoppage laps with status 5 and usually leaves them untimed, but the lap
    on which the race resumes (pit-lane exit, formation and standing restart) carries status 1
    or 12 and an "accurate" marker while being 30-90 s slower than racing pace. It is not a
    pace lap: it is excluded from training rows and targets, and it breaks temporal context.
    """
    return _restart_lap_after(df, is_red_flag, has_time)


def _safety_car_restart(df: pd.DataFrame, is_safety_car: pd.Series, has_time: pd.Series) -> pd.Series:
    """Flag the first timed lap after a safety-car period (status 4) within each driver/session.

    The safety car peels in at the end of its last lap, so the following lap starts from a
    bunched, slow restart and is 2.2% slower than the driver's race median at the median
    (17% of them more than 5% slower). The second lap after the safety car and the lap after
    a VSC are at racing pace and stay valid.
    """
    return _restart_lap_after(df, is_safety_car, has_time)


def lap_state_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive lap-state flags from FastF1-style track status codes and pit markers.

    FastF1 track status is a string of digit codes: 1 clear, 2 yellow, 4 safety car,
    5 red flag, 6 VSC deployed, 7 VSC ending. Several codes can be concatenated.
    The first timed lap after a red flag and after a safety-car period (restart laps) are
    flagged separately, see ``_red_flag_restart`` and ``_safety_car_restart``. Bump
    ``LAP_VALIDITY_VERSION`` whenever these rules change.
    """
    status = df["track_status"].astype("string").fillna("1").astype(str)
    status = status.where(~status.isin(["None", "nan", "<NA>", ""]), "1")
    out = pd.DataFrame(index=df.index)
    out["is_yellow"] = status.str.contains("2", regex=False)
    out["is_safety_car"] = status.str.contains("4", regex=False)
    out["is_red_flag"] = status.str.contains("5", regex=False)
    out["is_vsc"] = status.str.contains("6", regex=False) | status.str.contains("7", regex=False)
    out["is_pit_in"] = _flag(df["pit_in"])
    out["is_pit_out"] = _flag(df["pit_out"])
    out["is_deleted"] = _flag(df["deleted"])
    accurate = _flag(df["is_accurate"])
    has_time = pd.to_numeric(df["lap_time_s"], errors="coerce").notna()
    out["is_red_flag_restart"] = _red_flag_restart(df, out["is_red_flag"], has_time)
    out["is_safety_car_restart"] = _safety_car_restart(df, out["is_safety_car"], has_time)
    out["lap_valid"] = (
        accurate
        & has_time
        & ~out["is_deleted"]
        & ~out["is_pit_in"]
        & ~out["is_pit_out"]
        & ~out["is_yellow"]
        & ~out["is_safety_car"]
        & ~out["is_safety_car_restart"]
        & ~out["is_vsc"]
        & ~out["is_red_flag"]
        & ~out["is_red_flag_restart"]
    )
    return out


def _event_order(df: pd.DataFrame) -> pd.Series:
    """Chronological event index without using future outcomes."""
    if "event_date" in df:
        dates = pd.to_datetime(df["event_date"], errors="coerce")
        fallback = pd.to_datetime(df["season"].astype(str) + "-01-01", errors="coerce")
        return dates.fillna(fallback)
    if "round_number" in df:
        return pd.to_datetime(df["season"].astype(str) + "-01-01") + pd.to_timedelta(
            pd.to_numeric(df["round_number"], errors="coerce").fillna(0), unit="D"
        )
    event_key = df["season"].astype(str) + "|" + df["event"].astype(str)
    event_codes = pd.factorize(event_key, sort=False)[0]
    return pd.to_datetime(df["season"].astype(str) + "-01-01") + pd.to_timedelta(event_codes, unit="D")


def _prior_event_expanding_median(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    """Median of per-event medians from strictly earlier events, vectorised.

    Aggregates to one median per (group, season, event), orders events chronologically
    (event_date, then round_number, then appearance order) and takes the expanding
    median of everything *before* each event. The current event never contributes.
    """
    event_cols = ["season", "event"]
    order_cols = [c for c in ["event_date", "round_number"] if c in df.columns]
    key_cols = group_cols + event_cols
    # Group keys are stringified with an explicit missing marker so an all-missing category
    # (no compound on a legacy-tier lap) groups and merges consistently across dtypes.
    keyed = pd.DataFrame(index=df.index)
    for c in key_cols:
        col = df[c]
        keyed[c] = col.astype(object).where(col.notna(), "__missing__").astype(str)
    keyed[value_col] = df[value_col]
    for c in order_cols:
        keyed[c] = df[c]
    agg = {value_col: "median"}
    for c in order_cols:
        agg[c] = "first"
    event_summary = (
        keyed.groupby(key_cols, sort=False)
        .agg(agg)
        .rename(columns={value_col: "event_median"})
        .reset_index()
    )
    event_summary["_event_order"] = _event_order(event_summary)
    event_summary = event_summary.sort_values(group_cols + ["_event_order"], kind="stable")
    event_summary["prior"] = (
        event_summary.groupby(group_cols, sort=False)["event_median"]
        .transform(lambda s: s.shift(1).expanding().median())
    )
    lookup = event_summary[key_cols + ["prior"]]
    key_frame = keyed[key_cols].copy()
    key_frame["_row"] = np.arange(len(df))
    merged = key_frame.merge(lookup, on=key_cols, how="left", sort=False).sort_values("_row")
    return merged["prior"].to_numpy()


def _weather_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["track_temp_bin"] = (pd.to_numeric(df["track_temp_c"], errors="coerce") / 5.0).round() * 5.0
    out["air_temp_bin"] = (pd.to_numeric(df["air_temp_c"], errors="coerce") / 4.0).round() * 4.0
    out["humidity_bin"] = (pd.to_numeric(df["humidity_pct"], errors="coerce") / 10.0).round() * 10.0
    out["wind_bin"] = (pd.to_numeric(df["wind_speed_ms"], errors="coerce") / 2.0).round() * 2.0
    out["rain_state"] = _flag(df["rainfall"]).astype(int)
    return out


CHRONOLOGY_COLUMNS = ("event_date", "round_number")


def build_full_context_table(
    raw: pd.DataFrame, history: int = 5, require_chronology: bool = True, require_target: bool = True
) -> pd.DataFrame:
    """Build the leakage-safe feature table. See ``_build_full_context_table`` for the steps.

    ``require_target=True`` (training and evaluation) keeps only rows whose adjacent next lap
    is a valid racing lap, so every row has a target. ``require_target=False`` (inference)
    keeps every valid lap and leaves the target columns NaN where no usable next lap exists;
    the features of a row never depend on the next lap either way.
    """
    with warnings.catch_warnings():
        # ~60 derived columns are appended one at a time; pandas warns about block
        # fragmentation on every build, which drowns test output without changing results.
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _build_full_context_table(raw, history=history, require_chronology=require_chronology, require_target=require_target)


def _build_full_context_table(
    raw: pd.DataFrame, history: int = 5, require_chronology: bool = True, require_target: bool = True
) -> pd.DataFrame:
    """Build the canonical leakage-safe lap-N -> lap-(N+1) forecasting table.

    Returned rows are valid racing laps whose *next* lap is also a valid, adjacent racing
    lap. All lagged and rolling features are restricted to the current run of consecutive
    valid laps, so a pit stop or safety-car period resets the temporal context.

    Historical priors depend on the order of events. By default the table requires an
    explicit chronology column (``event_date`` or ``round_number``) and refuses to trust the
    order in which events happen to appear in the input. Pass ``require_chronology=False``
    only for single-event tables or deliberate exploratory work.
    """
    if history < 1:
        raise ValueError("history must be >= 1")
    df = _ensure_columns(raw)

    required = ["season", "event", "session", "driver", "lap_number", "lap_time_s"]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if require_chronology and not any(c in df.columns for c in CHRONOLOGY_COLUMNS):
        raise ValueError(
            "Event chronology is required: add an 'event_date' or 'round_number' column. "
            "Historical priors must not depend on the row order of the input file."
        )

    for col in ["season", "lap_number", "lap_time_s", "sector1_s", "sector2_s", "sector3_s"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sector1_s", "sector2_s", "sector3_s"]:
        if col not in df:
            df[col] = np.nan

    sort_cols = [c for c in ["season", "round_number", "event_date", "event", "session", "driver", "lap_number"] if c in df]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    df = df[df["lap_number"].notna()].reset_index(drop=True)

    # Lap-state flags and validity.
    flags = lap_state_flags(df)
    for c in LAP_STATE_COLUMNS:
        df[c] = flags[c].to_numpy()

    # Circular wind representation.
    radians = np.deg2rad(pd.to_numeric(df["wind_direction_deg"], errors="coerce"))
    df["wind_dir_sin"] = np.sin(radians)
    df["wind_dir_cos"] = np.cos(radians)

    # Interaction identity (driver within constructor) for one-hot encoding downstream.
    df["driver_team"] = df["driver"].astype(str) + "@" + df["team"].astype(str)

    # Segments of consecutive valid laps within each driver/session. A new segment starts
    # at every invalid lap and at every gap in lap numbering.
    group = df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    lap_gap = group["lap_number"].diff().fillna(1) != 1
    previous_invalid = ~group["lap_valid"].shift(1, fill_value=True).astype(bool)
    # A segment breaks at every invalid lap, at the first valid lap after one, and at any
    # gap in lap numbering, so an invalid lap never shares a segment with its neighbours.
    breaks = (~df["lap_valid"]) | previous_invalid | lap_gap
    df["_segment"] = breaks.groupby([df[c] for c in GROUP_COLUMNS], dropna=False).cumsum()
    segment_keys = GROUP_COLUMNS + ["_segment"]
    seg = df.groupby(segment_keys, dropna=False, sort=False)
    df["laps_in_segment"] = seg.cumcount() + 1

    # Recent pace over consecutive valid laps only.
    df["rolling_median_3"] = seg["lap_time_s"].transform(lambda s: s.rolling(3, min_periods=1).median())
    df["rolling_median_5"] = seg["lap_time_s"].transform(lambda s: s.rolling(5, min_periods=1).median())
    for sector in ["sector1_s", "sector2_s", "sector3_s"]:
        df[f"{sector.replace('_s', '')}_mean_3"] = seg[sector].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["pace_trend_3"] = seg["lap_time_s"].transform(lambda s: s.diff().rolling(3, min_periods=1).mean())
    df["lap_delta_to_rolling5"] = df["lap_time_s"] - df["rolling_median_5"]
    df["pace_spread_3_5"] = df["rolling_median_3"] - df["rolling_median_5"]
    for i in (1, 2, 3):
        df[f"sector{i}_rel_3"] = df[f"sector{i}_s"] - df[f"sector{i}_mean_3"]
        df[f"sector{i}_share_3"] = df[f"sector{i}_mean_3"] / df["rolling_median_5"]

    # Historical priors from strictly earlier events (valid laps only feed the medians).
    prior_specs = [
        (["driver", "circuit"], "hist_driver_circuit_pace"),
        (["team", "circuit"], "hist_team_circuit_pace"),
        (["compound", "circuit"], "hist_compound_circuit_pace"),
        (["driver", "circuit", "compound"], "hist_driver_circuit_compound_pace"),
        (["team", "circuit", "compound"], "hist_team_circuit_compound_pace"),
        (["driver"], "hist_driver_global_pace"),
        (["team"], "hist_team_global_pace"),
    ]
    weather = _weather_bins(df)
    for c in weather.columns:
        df[c] = weather[c].to_numpy()
    weather_groups = ["circuit", "compound", "track_temp_bin", "air_temp_bin", "humidity_bin", "wind_bin", "rain_state"]
    prior_specs += [
        (weather_groups, "hist_weather_compound_pace"),
        (["driver"] + weather_groups, "hist_driver_weather_pace"),
        (["team"] + weather_groups, "hist_team_weather_pace"),
    ]
    valid = df[df["lap_valid"]]
    for groups, target_name in prior_specs:
        df[target_name] = _prior_event_expanding_median(df, groups, "lap_time_s") if valid.empty else _prior_from_valid(df, valid, groups)
        df[f"{target_name}_rel"] = df[target_name] - df["rolling_median_5"]

    # Flattened temporal context: lag k is only defined when lap N-k is in the same run of
    # consecutive valid laps, which guarantees adjacency and validity.
    for source in LAG_SOURCE_FEATURES:
        if source not in df:
            df[source] = np.nan
        for lag in range(1, history):
            lagged = group[source].shift(lag)
            same_segment = group["_segment"].shift(lag) == df["_segment"]
            lagged = lagged.where(same_segment)
            reference = PACE_SCALE_LAG_SOURCES.get(source)
            if reference:
                df[f"{source}_lag{lag}_rel"] = lagged - df[reference]
            else:
                df[f"{source}_lag{lag}"] = lagged

    # Target: strictly the adjacent next lap, and that lap must itself be a valid racing lap.
    df[RAW_TARGET_COLUMN] = group["lap_time_s"].shift(-1)
    df["next_lap_number"] = group["lap_number"].shift(-1)
    df["next_lap_valid"] = group["lap_valid"].shift(-1)
    has_target = (
        ((df["next_lap_number"] - df["lap_number"]) == 1)
        & (df["next_lap_valid"] == True)  # noqa: E712 - shifted bools are object dtype
        & df[RAW_TARGET_COLUMN].notna()
    )
    keep = df["lap_valid"] & has_target if require_target else df["lap_valid"].astype(bool)
    df = df[keep].copy()
    has_target = has_target[keep]
    df[TARGET_COLUMN] = np.where(has_target, df[RAW_TARGET_COLUMN] - df["rolling_median_5"], np.nan)
    if not require_target:
        df.loc[~has_target, RAW_TARGET_COLUMN] = np.nan
    return df.drop(columns=["_segment"]).reset_index(drop=True)


def _prior_from_valid(df: pd.DataFrame, valid: pd.DataFrame, groups: list[str]) -> np.ndarray:
    """Compute priors on valid laps, then align them back to every row of `df` by event key."""
    prior_valid = pd.Series(_prior_event_expanding_median(valid, groups, "lap_time_s"), index=valid.index)
    # Priors are constant within (group, season, event); broadcast through a merge so rows
    # that are themselves invalid (and therefore absent from `valid`) still receive them.
    key_cols = groups + ["season", "event"]
    lookup = valid[key_cols].copy()
    lookup["_prior"] = prior_valid.to_numpy()
    lookup = lookup.drop_duplicates(key_cols)
    key_frame = df[key_cols].copy()
    key_frame["_row"] = np.arange(len(df))
    merged = key_frame.merge(lookup, on=key_cols, how="left", sort=False).sort_values("_row")
    return merged["_prior"].to_numpy()


def lag_feature_names(history: int = 5) -> list[str]:
    names = []
    for source in LAG_SOURCE_FEATURES:
        suffix = "_rel" if source in PACE_SCALE_LAG_SOURCES else ""
        names += [f"{source}_lag{lag}{suffix}" for lag in range(1, history)]
    return names


def feature_contract(history: int = 5) -> FeatureContract:
    return FeatureContract(numeric=BASE_NUMERIC_FEATURES + lag_feature_names(history), categorical=CATEGORICAL_FEATURES)


def feature_taxonomy(history: int = 5) -> dict[str, list[str]]:
    """Feature groups for documentation and ablations."""
    lag_features = lag_feature_names(history)
    return {
        "static_categorical": list(STATIC_CATEGORICAL),
        "dynamic_numeric": list(DYNAMIC_NUMERIC),
        "temporal_numeric": list(TEMPORAL_NUMERIC) + lag_features,
        "historical_numeric": list(HISTORICAL_NUMERIC),
        "lap_state_not_features": list(LAP_STATE_COLUMNS),
    }


def assert_no_target_leakage(columns: Iterable[str]) -> None:
    bad = sorted(set(columns).intersection(LEAKAGE_BLOCKLIST))
    if bad:
        raise ValueError(f"Leakage columns present: {bad}")
