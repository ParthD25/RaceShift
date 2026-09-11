"""The Kaggle Ergast CSV adapter yields legacy-tier rows with the same rules as the API loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from raceshift.data.ergast_csv_loader import load_tables, season_frame
from raceshift.data.jolpica_loader import DATA_TIER, team_name


def _write_csvs(folder: Path) -> None:
    pd.DataFrame({"raceId": [1], "year": [2025], "round": [3], "circuitId": [5], "name": ["Test Grand Prix"], "date": ["2025-04-06"]}).to_csv(folder / "races.csv", index=False)
    pd.DataFrame({"circuitId": [5], "circuitRef": ["suzuka"], "name": ["Suzuka Circuit"], "location": ["Suzuka"]}).to_csv(folder / "circuits.csv", index=False)
    pd.DataFrame({"driverId": [1, 2], "driverRef": ["hakkinen", "norris"], "code": ["\\N", "NOR"], "surname": ["Häkkinen", "Norris"]}).to_csv(folder / "drivers.csv", index=False)
    pd.DataFrame({"constructorId": [10, 11], "constructorRef": ["rb", "mclaren"], "name": ["RB F1 Team", "McLaren"]}).to_csv(folder / "constructors.csv", index=False)
    pd.DataFrame({"resultId": [1, 2], "raceId": [1, 1], "driverId": [1, 2], "constructorId": [10, 11]}).to_csv(folder / "results.csv", index=False)
    rows = []
    for driver in (1, 2):
        for lap in range(1, 8):
            ms = 90000 + lap * 10 + (30000 if (driver == 1 and lap == 4) else 0)
            rows.append({"raceId": 1, "driverId": driver, "lap": lap, "position": driver, "time": "1:30.0", "milliseconds": ms})
    pd.DataFrame(rows).to_csv(folder / "lap_times.csv", index=False)
    pd.DataFrame({"raceId": [1], "driverId": [1], "stop": [1], "lap": [4], "time": ["14:20:00"], "duration": ["22.0"], "milliseconds": [22000]}).to_csv(folder / "pit_stops.csv", index=False)


def test_season_frame_matches_the_legacy_schema(tmp_path: Path):
    _write_csvs(tmp_path)
    frame = season_frame(load_tables(tmp_path), 2025)
    assert len(frame) == 14 and frame["data_tier"].eq(DATA_TIER).all() and frame["session"].eq("R").all()
    assert set(frame["driver"]) == {"HAK", "NOR"}  # accents stripped, official code kept
    assert set(frame["team"]) == {"Racing Bulls", "McLaren"}  # 2025 name of the rb constructor
    assert frame["circuit"].iloc[0] == "Suzuka" and frame["round_number"].iloc[0] == 3 and frame["event_date"].iloc[0] == "2025-04-06"
    hak = frame[frame["driver"] == "HAK"].set_index("lap_number")
    assert bool(hak.loc[4, "pit_in"]) and bool(hak.loc[5, "pit_out"]) and not bool(hak.loc[3, "pit_in"])
    assert not bool(hak.loc[1, "is_accurate"]) and not bool(hak.loc[4, "is_accurate"]) and bool(hak.loc[6, "is_accurate"])
    assert hak["track_status"].isna().all() and (~hak["deleted"]).all()
    assert abs(hak.loc[2, "lap_time_s"] - 90.02) < 1e-9


def test_team_name_follows_renames():
    assert team_name("rb", 2024) == "RB" and team_name("rb", 2025) == "Racing Bulls"
    assert team_name("sauber", 2018) == "Sauber" and team_name("sauber", 2025) == "Kick Sauber"
    assert team_name("alfa", 2020) == "Alfa Romeo Racing" and team_name("alfa", 2023) == "Alfa Romeo"
    assert team_name("audi", 2026) == "Audi" and team_name("unknown_ref", 2026, "Fallback") == "Fallback"


def test_timing_era_seasons_need_an_explicit_opt_in(tmp_path: Path):
    import pytest
    from raceshift.data.ergast_csv_loader import export_seasons

    with pytest.raises(ValueError, match="include_timing_era"):
        export_seasons(tmp_path, [2017, 2018], tmp_path / "out.parquet")


def test_check_only_rows_never_reach_the_feature_builder():
    import pytest
    from raceshift.data.ergast_csv_loader import CHECK_ONLY_TIER
    from raceshift.features.full_context import build_full_context_table

    raw = pd.DataFrame({"season": [2024], "event": ["A"], "session": ["R"], "driver": ["VER"], "lap_number": [2], "lap_time_s": [90.0], "data_tier": [CHECK_ONLY_TIER]})
    with pytest.raises(ValueError, match="check-only"):
        build_full_context_table(raw)
