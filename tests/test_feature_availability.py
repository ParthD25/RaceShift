"""Every model input for lap N must be computable from information known at the end of lap N.

The test perturbs everything that happens *after* a chosen lap (later laps of the same
race, for every driver, and every later event) and asserts that no feature attached to
that lap or to any earlier lap changes. This catches target leakage, off-by-one shifts
and historical priors that peek at the current or future events.
"""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from make_synthetic_fixture import build_fixture
from raceshift.features.full_context import (
    HISTORICAL_PRIORS,
    LEAKAGE_BLOCKLIST,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    build_full_context_table,
    feature_contract,
    feature_taxonomy,
)

CUTOFF = (2024, 1, 6)  # season, round_number, lap_number
NUMERIC_TO_PERTURB = [
    "lap_time_s", "sector1_s", "sector2_s", "sector3_s", "air_temp_c", "track_temp_c",
    "humidity_pct", "pressure_mbar", "wind_speed_ms", "wind_direction_deg", "gap_ahead_s",
    "gap_behind_s", "mean_speed_kph", "mean_throttle_pct", "brake_fraction", "tyre_life", "position",
]


def _order_key(frame: pd.DataFrame) -> pd.Series:
    return list(zip(frame["season"].astype(int), frame["round_number"].astype(int), frame["lap_number"].astype(int)))


def _after_cutoff(frame: pd.DataFrame) -> pd.Series:
    return pd.Series([k > CUTOFF for k in _order_key(frame)], index=frame.index)


def test_features_at_lap_n_are_invariant_to_everything_after_lap_n():
    raw = build_fixture(seasons=(2023, 2024, 2025), events_per_season=2, laps=12)
    rng = np.random.default_rng(11)
    perturbed = raw.copy()
    future = _after_cutoff(perturbed)
    for col in NUMERIC_TO_PERTURB:
        noise = rng.normal(0, 5.0, size=int(future.sum()))
        # Integer fixture columns must become float before a float assignment (pandas 2.x
        # otherwise emits a FutureWarning that prints the whole array).
        perturbed[col] = perturbed[col].astype(float)
        perturbed.loc[future, col] = perturbed.loc[future, col].to_numpy(dtype=float) + noise
    perturbed.loc[future, "compound"] = "WET"

    before = build_full_context_table(raw, history=5)
    after = build_full_context_table(perturbed, history=5)

    contract = feature_contract(history=5)
    features = contract.numeric + contract.categorical
    key_cols = ["season", "round_number", "driver", "lap_number"]

    past_before = before[~_after_cutoff(before)].set_index(key_cols).sort_index()
    past_after = after.set_index(key_cols).loc[past_before.index].reset_index()
    past_before = past_before.reset_index()
    assert len(past_before) > 50

    for col in features:
        a = past_before[col].to_numpy()
        b = past_after[col].to_numpy()
        if np.issubdtype(np.asarray(a).dtype, np.number):
            same = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            same = (pd.Series(a).astype(str) == pd.Series(b).astype(str)).to_numpy()
        assert same.all(), f"feature {col} changed when only future laps were modified"

    # The target for the cutoff lap *does* depend on the future, proving the perturbation bit.
    cutoff_rows = [k == CUTOFF for k in _order_key(past_before)]
    assert not np.allclose(
        past_before[RAW_TARGET_COLUMN].to_numpy()[cutoff_rows],
        past_after[RAW_TARGET_COLUMN].to_numpy()[cutoff_rows],
    )


def test_historical_priors_ignore_the_current_event():
    raw = build_fixture(seasons=(2023, 2024), events_per_season=2, laps=10)
    shifted = raw.copy()
    current = (shifted["season"] == 2024) & (shifted["round_number"] == 2)
    shifted.loc[current, "lap_time_s"] += 30.0  # wildly slower race
    before = build_full_context_table(raw, history=5)
    after = build_full_context_table(shifted, history=5)
    # Compare the absolute priors: the model-facing *_rel versions are relative to the
    # current rolling pace, which legitimately changes when the current event changes.
    hist_cols = list(HISTORICAL_PRIORS)
    key = ["season", "round_number", "driver", "lap_number"]
    b = before[(before["season"] == 2024) & (before["round_number"] == 2)].set_index(key)[hist_cols]
    a = after.set_index(key).loc[b.index][hist_cols]
    assert np.allclose(b.to_numpy(dtype=float), a.to_numpy(dtype=float), equal_nan=True)
    # ...while the temporal pace features of that event do change, as they should.
    b_pace = before[(before["season"] == 2024) & (before["round_number"] == 2)].set_index(key)["rolling_median_5"]
    a_pace = after.set_index(key).loc[b_pace.index]["rolling_median_5"]
    assert not np.allclose(b_pace.to_numpy(), a_pace.to_numpy())


def test_contract_never_contains_blocklisted_columns_and_taxonomy_covers_contract():
    contract = feature_contract(history=5)
    assert not LEAKAGE_BLOCKLIST.intersection(contract.numeric + contract.categorical)
    taxonomy = feature_taxonomy(history=5)
    covered = set(taxonomy["static_categorical"]) | set(taxonomy["dynamic_numeric"]) | set(taxonomy["temporal_numeric"]) | set(taxonomy["historical_numeric"])
    assert covered == set(contract.numeric + contract.categorical)
    assert TARGET_COLUMN not in covered


def test_event_chronology_is_required_by_default():
    raw = build_fixture(seasons=(2024,), events_per_season=2, laps=8).drop(columns=["event_date", "round_number"])
    with pytest.raises(ValueError, match="chronology"):
        build_full_context_table(raw, history=5)
    # Explicit opt-out keeps working for exploratory single-file use.
    assert len(build_full_context_table(raw, history=5, require_chronology=False)) > 0
