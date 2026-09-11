from __future__ import annotations

import numpy as np

from raceshift.features.full_context import build_full_context_table, feature_contract, assert_no_target_leakage
from scripts.make_synthetic_fixture import build_fixture


def test_full_context_contains_requested_weather_tyre_driver_history_features():
    raw = build_fixture(seasons=(2022, 2023), events_per_season=2, laps=8)
    table = build_full_context_table(raw, history=5)
    expected = {
        "driver",
        "team",
        "compound",
        "tyre_life",
        "air_temp_c",
        "track_temp_c",
        "humidity_pct",
        "wind_speed_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "hist_driver_circuit_pace",
        "hist_team_circuit_pace",
        "hist_weather_compound_pace",
        "lap_time_s_lag4_rel",
        "hist_driver_circuit_pace_rel",
    }
    assert expected.issubset(table.columns)


def test_wind_direction_is_circular():
    raw = build_fixture(seasons=(2022,), events_per_season=1, laps=8)
    one = (raw["driver"] == "AAA") & (raw["lap_number"] == 1)
    two = (raw["driver"] == "AAA") & (raw["lap_number"] == 2)
    raw.loc[one, "wind_direction_deg"] = 359.0
    raw.loc[two, "wind_direction_deg"] = 1.0
    table = build_full_context_table(raw, history=2)
    assert np.isfinite(table["wind_dir_sin"]).all()
    assert np.isfinite(table["wind_dir_cos"]).all()
    # 359 deg and 1 deg are two degrees apart on the circle, so their encodings must be close
    # (a linear encoding would put them 358 units apart) and lie on the unit circle.
    aaa = table[table["driver"] == "AAA"].set_index("lap_number")
    first, second = aaa.loc[1], aaa.loc[2]
    assert abs(first["wind_dir_sin"] - second["wind_dir_sin"]) < 0.05
    assert abs(first["wind_dir_cos"] - second["wind_dir_cos"]) < 0.05
    assert np.isclose(first["wind_dir_sin"], np.sin(np.deg2rad(359.0)))
    assert np.isclose(first["wind_dir_cos"], np.cos(np.deg2rad(359.0)))
    assert np.allclose(table["wind_dir_sin"] ** 2 + table["wind_dir_cos"] ** 2, 1.0)


def test_feature_contract_blocks_target_columns():
    contract = feature_contract(history=5)
    assert_no_target_leakage(contract.numeric + contract.categorical)
