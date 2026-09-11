from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_fixture import build_fixture
from raceshift.features.lap_features import build_next_lap_table, assert_no_target_leakage


def test_next_lap_target_is_shifted_one_lap():
    raw = build_fixture(seasons=(2023,), events_per_season=1, drivers=("AAA",), laps=8)
    table = build_next_lap_table(raw)
    assert len(table) == 7
    original = raw.sort_values("lap_number")
    assert table.iloc[0]["target_next_lap_time_s"] == pytest.approx(original.iloc[1]["lap_time_s"])


def test_target_columns_are_blocked_from_features():
    with pytest.raises(ValueError):
        assert_no_target_leakage(["lap_time_s", "target_next_lap_time_s"])
