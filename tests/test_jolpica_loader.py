"""The legacy (Ergast/Jolpica) tier must export the RaceShift lap schema without network access."""
from __future__ import annotations

import pandas as pd
import pytest

from raceshift.data import jolpica_loader as jl
from raceshift.features.full_context import build_full_context_table


class FakeClient:
    """Serves a two-driver, six-lap race with one pit stop, mimicking Jolpica payloads."""

    def paged(self, path: str, table: str, item: str):
        race = {
            "season": "2012", "round": "3", "raceName": "Test Grand Prix", "date": "2012-04-15",
            "Circuit": {"circuitId": "catalunya", "Location": {"locality": "Montmeló", "country": "Spain"}},
        }
        if path.endswith("/laps"):
            laps = []
            for n in range(1, 7):
                base = 95.0 if n > 1 else 105.0  # opening lap is slow
                ham = base + (25.0 if n == 4 else 0.0)  # pit stop on lap 4
                laps.append({"number": str(n), "Timings": [
                    {"driverId": "hamilton", "position": "1", "time": f"1:{ham - 60:06.3f}"},
                    {"driverId": "alonso", "position": "2", "time": f"1:{base + 0.4 - 60:06.3f}"},
                ]})
            return laps, [dict(race, Laps=laps)]
        if path.endswith("/results"):
            results = [
                {"Driver": {"driverId": "hamilton", "code": "HAM", "familyName": "Hamilton"}, "Constructor": {"constructorId": "mclaren", "name": "McLaren"}},
                {"Driver": {"driverId": "alonso", "familyName": "Alonso"}, "Constructor": {"constructorId": "ferrari", "name": "Ferrari"}},
            ]
            return results, [dict(race, Results=results)]
        if path.endswith("/pitstops"):
            stops = [{"driverId": "hamilton", "lap": "4", "stop": "1", "duration": "22.1"}]
            return stops, [dict(race, PitStops=stops)]
        raise AssertionError(path)


def test_export_race_writes_schema_with_tier_and_validity_heuristic(tmp_path):
    path = jl.export_race(FakeClient(), 2012, 3, tmp_path)
    frame = pd.read_parquet(path)
    assert len(frame) == 12
    assert set(frame["data_tier"]) == {"legacy_timing"}
    assert set(frame["driver"]) == {"HAM", "ALO"}  # code when present, else family-name prefix
    assert set(frame["team"]) == {"McLaren", "Ferrari"}
    assert set(frame["circuit"]) == {"Barcelona"}  # mapped onto the FastF1 location name
    assert frame["event_date"].iloc[0] == "2012-04-15" and frame["round_number"].iloc[0] == 3
    ham = frame[frame["driver"] == "HAM"].set_index("lap_number")
    assert bool(ham.loc[4, "pit_in"]) and bool(ham.loc[5, "pit_out"])
    assert not bool(ham.loc[1, "is_accurate"])  # opening lap
    assert not bool(ham.loc[4, "is_accurate"]) and not bool(ham.loc[5, "is_accurate"])
    assert bool(ham.loc[3, "is_accurate"]) and bool(ham.loc[6, "is_accurate"])
    # Columns the tier cannot provide stay absent/missing instead of being invented.
    for col in ("sector1_s", "compound", "tyre_life", "air_temp_c"):
        assert col not in frame.columns
    # The feature table accepts the tier and keeps the tier tag as a categorical input.
    table = build_full_context_table(frame, history=5)
    assert set(table["data_tier"]) == {"legacy_timing"}
    assert table["sector1_rel_3"].isna().all()


def test_lap_time_parsing_and_driver_codes():
    assert jl._lap_time_seconds("1:34.567") == pytest.approx(94.567)
    assert jl._lap_time_seconds("1:02:03.500") == pytest.approx(3723.5)
    assert jl._lap_time_seconds("bad") != jl._lap_time_seconds("bad")  # NaN
    assert jl._driver_code({"code": "msc"}) == "MSC"
    assert jl._driver_code({"familyName": "Räikkönen"}) == "RAI"  # accents stripped, matches the FastF1 abbreviation
