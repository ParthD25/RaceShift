from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from make_synthetic_fixture import build_fixture
from raceshift.models.artifact import RaceShiftArtifact, missing_artifact_files

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "raceshift_ffr_demo"


@pytest.fixture(scope="module")
def artifact() -> RaceShiftArtifact:
    assert missing_artifact_files(DEMO) == []
    return RaceShiftArtifact(DEMO)


def test_demo_artifact_is_labelled_synthetic(artifact: RaceShiftArtifact):
    assert artifact.is_synthetic is True
    assert artifact.data_source == "synthetic_fixture"


def test_forecast_uses_latest_session_and_returns_finite_interval(artifact: RaceShiftArtifact):
    raw = build_fixture(seasons=(2024, 2025), events_per_season=3, laps=20)
    # Make the latest event have fewer laps than an earlier one; the old code picked by lap_number alone.
    latest_mask = (raw["season"] == 2025) & (raw["round_number"] == 3)
    raw = raw[~(latest_mask & (raw["lap_number"] > 12))]
    result = artifact.forecast_last_available(raw, driver="BBB")
    assert result["season"] == 2025
    assert result["event"] == "Synthetic_GP_3"
    assert result["driver"] == "BBB"
    assert result["lap_number_completed"] == 12
    assert result["available_drivers"] == ["AAA", "BBB", "CCC", "DDD"]
    assert np.isfinite(result["predicted_next_lap_s"])
    assert result["lower_80_s"] <= result["predicted_next_lap_s"] <= result["upper_80_s"]
    assert result["is_synthetic"] is True


def test_forecast_defaults_to_driver_with_most_laps(artifact: RaceShiftArtifact):
    raw = build_fixture(seasons=(2025,), events_per_season=1, laps=15)
    raw = raw[~((raw["driver"] != "CCC") & (raw["lap_number"] > 10))]
    result = artifact.forecast_last_available(raw)
    assert result["driver"] == "CCC"
    assert result["lap_number_completed"] == 15


def test_forecast_rejects_missing_columns_unknown_driver_and_short_history(artifact: RaceShiftArtifact):
    raw = build_fixture(seasons=(2025,), events_per_season=1, laps=8)
    with pytest.raises(ValueError, match="Missing required columns"):
        artifact.forecast_last_available(raw.drop(columns=["lap_time_s"]))
    with pytest.raises(ValueError, match="not found"):
        artifact.forecast_last_available(raw, driver="ZZZ")
    one_lap = raw[(raw["driver"] == "AAA") & (raw["lap_number"] == 1)]
    with pytest.raises(ValueError, match="at least two"):
        artifact.forecast_last_available(one_lap, driver="AAA")


def test_forecast_rejects_when_latest_lap_is_a_pit_lap(artifact: RaceShiftArtifact):
    raw = build_fixture(seasons=(2025,), events_per_season=1, drivers=("AAA",), laps=10).copy()
    raw.loc[raw["lap_number"] == 10, "pit_in"] = True
    with pytest.raises(ValueError, match="not a usable completed lap"):
        artifact.forecast_last_available(raw, driver="AAA")


def test_missing_artifact_files_reports_gaps(tmp_path: Path):
    assert len(missing_artifact_files(tmp_path)) == 6
    (tmp_path / "metrics.json").write_text("{}")
    assert "metrics.json" not in missing_artifact_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        RaceShiftArtifact(tmp_path)
