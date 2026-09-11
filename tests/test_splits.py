from __future__ import annotations

import pytest

from make_synthetic_fixture import build_fixture
from raceshift.data.splits import chronological_split, leave_event_out, season_forward_split, season_round_split
from raceshift.features.lap_features import build_next_lap_table


def test_season_forward_split_never_mixes_future_rows():
    table = build_next_lap_table(build_fixture())
    train, val, test = season_forward_split(table, train_end=2023, val_year=2024, test_year=2025)
    assert train["season"].max() <= 2023
    assert set(val["season"].unique()) == {2024}
    assert set(test["season"].unique()) == {2025}


def test_season_forward_split_rejects_non_chronological_arguments():
    table = build_next_lap_table(build_fixture())
    with pytest.raises(ValueError):
        season_forward_split(table, train_end=2025, val_year=2024, test_year=2023)


def test_season_round_split_carves_holdout_season_by_round():
    table = build_next_lap_table(build_fixture(events_per_season=3))
    train, val, test = season_round_split(table, train_end=2024, holdout_year=2025, cutoff_round=1)
    assert train["season"].max() <= 2024
    assert set(val["season"]) == {2025} and set(val["round_number"]) == {1}
    assert set(test["season"]) == {2025} and set(test["round_number"]) == {2, 3}
    t2, v2, s2 = chronological_split(table, 2024, 2025, 2025, split_round=1)
    assert len(t2) == len(train) and len(v2) == len(val) and len(s2) == len(test)


def test_leave_event_out_holds_out_every_season_of_that_event():
    table = build_next_lap_table(build_fixture())
    train, test = leave_event_out(table, "Synthetic_GP_2")
    assert set(test["event"]) == {"Synthetic_GP_2"}
    assert "Synthetic_GP_2" not in set(train["event"])
    assert set(test["season"]) == {2022, 2023, 2024, 2025}
