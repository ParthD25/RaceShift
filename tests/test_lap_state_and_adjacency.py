from __future__ import annotations

import numpy as np
import pytest

from make_synthetic_fixture import build_fixture
from raceshift.features.full_context import build_full_context_table, lap_state_flags


def _one_driver(laps: int = 12):
    return build_fixture(seasons=(2024,), events_per_season=1, drivers=("AAA",), laps=laps).copy()


def _row(table, lap):
    rows = table[table["lap_number"] == lap]
    assert len(rows) == 1, f"expected exactly one row for lap {lap}, got {len(rows)}"
    return rows.iloc[0]


def test_pit_stop_breaks_lag_and_rolling_context():
    raw = _one_driver()
    raw.loc[raw["lap_number"] == 6, "pit_in"] = True
    raw.loc[raw["lap_number"] == 7, "pit_out"] = True
    table = build_full_context_table(raw, history=5)

    # Lap 5's next lap is a pit-in lap, laps 6 and 7 are pit laps: none are training rows.
    assert set(table["lap_number"]).isdisjoint({5, 6, 7})

    # First lap after the pit stop has no lagged context and restarts the rolling window.
    lap8 = _row(table, 8)
    assert np.isnan(lap8["lap_time_s_lag1_rel"])
    assert lap8["laps_in_segment"] == 1
    assert lap8["rolling_median_5"] == pytest.approx(lap8["lap_time_s"])

    # One lap later the context is exactly one valid lap deep (lags are relative to the
    # current rolling pace, so recover the absolute lag before comparing).
    lap9 = _row(table, 9)
    assert lap9["lap_time_s_lag1_rel"] + lap9["rolling_median_5"] == pytest.approx(lap8["lap_time_s"])
    assert np.isnan(lap9["lap_time_s_lag2_rel"])
    assert lap9["laps_in_segment"] == 2

    # Before the pit stop the context is intact.
    lap4 = _row(table, 4)
    assert lap4["lap_time_s_lag1_rel"] + lap4["rolling_median_5"] == pytest.approx(raw.loc[raw["lap_number"] == 3, "lap_time_s"].iloc[0])


def test_safety_car_and_vsc_laps_are_excluded_from_pace_rows():
    raw = _one_driver()
    raw.loc[raw["lap_number"] == 4, "track_status"] = "4"   # safety car
    raw.loc[raw["lap_number"] == 9, "track_status"] = "67"  # VSC deployed then ending
    flags = lap_state_flags(raw)
    assert bool(flags.loc[raw["lap_number"] == 4, "is_safety_car"].iloc[0])
    assert bool(flags.loc[raw["lap_number"] == 9, "is_vsc"].iloc[0])

    # Lap 5 is the safety-car restart lap (rules v3): invalid, so lap 6 starts the new segment.
    assert bool(flags.loc[raw["lap_number"] == 5, "is_safety_car_restart"].iloc[0])
    assert not bool(flags.loc[raw["lap_number"] == 10, "is_safety_car_restart"].iloc[0])
    table = build_full_context_table(raw, history=5)
    assert set(table["lap_number"]).isdisjoint({3, 4, 5, 8, 9})
    assert np.isnan(_row(table, 6)["lap_time_s_lag1_rel"])
    assert np.isnan(_row(table, 10)["lap_time_s_lag1_rel"])


def test_yellow_flag_laps_are_invalid_and_break_context():
    raw = _one_driver()
    raw.loc[raw["lap_number"] == 6, "track_status"] = "12"  # clear, then a yellow during the lap
    flags = lap_state_flags(raw)
    by_lap = flags.set_index(raw["lap_number"])
    assert bool(by_lap.loc[6, "is_yellow"]) and not bool(by_lap.loc[6, "lap_valid"])
    assert bool(by_lap.loc[7, "lap_valid"])
    table = build_full_context_table(raw, history=5)
    # Lap 5 has no valid next lap, lap 6 is invalid, lap 7 restarts the segment without lags.
    assert set(table["lap_number"]).isdisjoint({5, 6})
    assert np.isnan(_row(table, 7)["lap_time_s_lag1_rel"])
    assert _row(table, 8)["laps_in_segment"] == 2


def test_safety_car_restart_only_first_lap_after_period_and_untimed_sc_laps():
    raw = _one_driver(14)
    raw.loc[raw["lap_number"].isin([4, 5]), "track_status"] = "4"
    raw.loc[raw["lap_number"] == 5, "lap_time_s"] = np.nan  # untimed lap behind the safety car
    flags = lap_state_flags(raw).set_index(raw["lap_number"])
    assert flags["is_safety_car_restart"].sum() == 1
    assert bool(flags.loc[6, "is_safety_car_restart"]) and not bool(flags.loc[6, "lap_valid"])
    assert bool(flags.loc[7, "lap_valid"]) and not bool(flags.loc[7, "is_safety_car_restart"])


def test_red_flag_restart_lap_is_invalid_and_breaks_context():
    raw = _one_driver()
    # Red flag shown on lap 5; laps 5 and 6 are the untimed stoppage laps, lap 7 is the
    # restart lap FastF1 marks as accurate with a clear track status.
    raw.loc[raw["lap_number"] == 5, "track_status"] = "125"
    raw.loc[raw["lap_number"].isin([5, 6]), "lap_time_s"] = np.nan
    raw.loc[raw["lap_number"] == 7, "lap_time_s"] = 140.0
    flags = lap_state_flags(raw)
    by_lap = flags.set_index(raw["lap_number"].to_numpy())
    assert bool(by_lap.loc[5, "is_red_flag"])
    assert bool(by_lap.loc[7, "is_red_flag_restart"])
    assert not bool(by_lap.loc[7, "lap_valid"])
    assert bool(by_lap.loc[8, "lap_valid"]) and not bool(by_lap.loc[8, "is_red_flag_restart"])
    # Only the first timed lap after the stoppage is a restart lap.
    assert by_lap["is_red_flag_restart"].sum() == 1

    table = build_full_context_table(raw, history=5)
    assert set(table["lap_number"]).isdisjoint({4, 5, 6, 7})
    lap8 = _row(table, 8)
    assert lap8["laps_in_segment"] == 1
    assert np.isnan(lap8["lap_time_s_lag1_rel"])


def test_red_flag_on_first_lap_flags_first_timed_lap_and_shuffled_rows():
    raw = _one_driver()
    raw.loc[raw["lap_number"] == 1, "track_status"] = "5"
    raw.loc[raw["lap_number"] == 1, "lap_time_s"] = np.nan
    shuffled = raw.sample(frac=1.0, random_state=0)
    flags = lap_state_flags(shuffled)
    restart_laps = shuffled.loc[flags["is_red_flag_restart"], "lap_number"].tolist()
    assert restart_laps == [2]


def test_lap_validity_version_is_recorded_in_artifacts():
    import json
    from pathlib import Path

    from raceshift.features.full_context import LAP_VALIDITY_VERSION

    assert LAP_VALIDITY_VERSION >= 3
    root = Path(__file__).resolve().parents[1] / "artifacts"
    # Every artifact that ships a metrics.json (models and baseline reports alike) must have
    # been produced under the rules the runtime applies; the four named ones must exist.
    required = ["raceshift_ffr_demo", "f1_2025h2_ffr-m", "f1_2025h2_ffr-s", "f1_2025h2_legacy_ext_ffr-s"]
    for name in required:
        assert (root / name / "metrics.json").exists(), f"missing committed artifact metrics: {name}"
    checked = 0
    for metrics in sorted(root.glob("*/metrics.json")):
        version = json.loads(metrics.read_text()).get("lap_validity_version")
        assert version == LAP_VALIDITY_VERSION, f"{metrics.parent.name}: lap_validity_version {version} != {LAP_VALIDITY_VERSION}"
        checked += 1
    assert checked >= len(required)


def test_missing_lap_number_breaks_adjacency():
    raw = _one_driver()
    raw = raw[raw["lap_number"] != 6]
    table = build_full_context_table(raw, history=5)
    assert 5 not in set(table["lap_number"])  # next lap (6) is missing
    assert np.isnan(_row(table, 7)["lap_time_s_lag1_rel"])
    assert _row(table, 7)["laps_in_segment"] == 1


def test_deleted_and_inaccurate_laps_are_not_targets():
    raw = _one_driver()
    raw.loc[raw["lap_number"] == 5, "deleted"] = True
    raw.loc[raw["lap_number"] == 9, "is_accurate"] = False
    table = build_full_context_table(raw, history=5)
    assert set(table["lap_number"]).isdisjoint({4, 5, 8, 9})
    assert "lap_valid" in table.columns and bool(table["lap_valid"].all())


def test_lap_state_columns_are_not_model_features():
    from raceshift.features.full_context import LAP_STATE_COLUMNS, feature_contract

    contract = feature_contract(history=5)
    assert not set(LAP_STATE_COLUMNS).intersection(contract.numeric + contract.categorical)
    assert "driver_team" in contract.categorical
