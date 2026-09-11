"""The TracingInsights adapter maps the columnar session_laptimes.json onto the RaceShift schema."""
from __future__ import annotations

import pandas as pd

from raceshift.data.tracinginsights_loader import DATA_TIER, assign_rounds, raw_url, session_frame


def _payload():
    return {
        "lap": [1, 2, 3, 1, 2],
        "time": [95.1, 91.2, "None", 96.0, 92.5],
        "s1": ["None", 30.0, "None", 31.0, 30.5],
        "s2": [30.0, 30.5, "None", 32.0, 31.0],
        "s3": [35.0, 30.7, "None", 33.0, 31.0],
        "compound": ["MEDIUM", "MEDIUM", "HARD", "SOFT", "SOFT"],
        "life": [1, 2, 1, 3, 4],
        "fresh": [True, True, True, False, False],
        "stint": [1, 1, 2, 1, 1],
        "pos": [3, 3, 5, 8, 7],
        "status": ["12", "1", "4", "1", "1"],
        "pin": ["None", 3700.0, "None", "None", "None"],
        "pout": ["None", "None", 3730.0, "None", "None"],
        "iacc": [False, True, False, True, True],
        "del": [False, False, False, False, True],
        "drv": ["VER", "VER", "VER", "NOR", "NOR"],
        "team": ["Red Bull Racing", "Red Bull Racing", "Red Bull Racing", "McLaren", "McLaren"],
        "lSD": ["2025-03-23T07:03:38.698000000", "2025-03-23T07:05:13.000000000", "2025-03-23T07:06:44", "2025-03-23T07:03:39.000000000", "2025-03-23T07:05:15.000000000"],
        "wAT": [27.3, 27.4, 27.4, 27.3, 27.4],
        "wTT": [35.8, 35.6, "None", 35.8, 35.6],
        "wH": [18.0, 18.0, 18.0, 18.0, 18.0],
        "wP": [1010.4, 1010.4, 1010.4, 1010.4, 1010.4],
        "wR": [False, False, False, False, False],
        "wWS": [1.8, 2.2, 2.9, 1.8, 2.2],
        "wWD": [249, 264, 254, 249, 264],
    }


def test_session_frame_maps_columns_and_flags():
    frame = session_frame(_payload(), 2025, "Chinese Grand Prix", "R", round_number=2)
    assert len(frame) == 5 and frame["data_tier"].eq(DATA_TIER).all()
    assert frame["circuit"].iloc[0] == "Shanghai" and frame["event_date"].iloc[0] == "2025-03-23" and frame["session"].iloc[0] == "R"
    ver = frame[frame["driver"] == "VER"].set_index("lap_number")
    assert ver.loc[2.0, "lap_time_s"] == 91.2 and pd.isna(ver.loc[3.0, "lap_time_s"]) and pd.isna(ver.loc[1.0, "sector1_s"])
    assert ver["pit_in"].tolist() == [False, True, False] and ver["pit_out"].tolist() == [False, False, True]
    assert ver["track_status"].tolist() == ["12", "1", "4"] and ver["is_accurate"].tolist() == [False, True, False]
    assert ver["compound"].tolist() == ["MEDIUM", "MEDIUM", "HARD"] and ver["stint"].tolist() == [1.0, 1.0, 2.0]
    nor = frame[frame["driver"] == "NOR"].set_index("lap_number")
    assert bool(nor.loc[2.0, "deleted"]) and not bool(nor.loc[2.0, "fresh_tyre"]) and nor.loc[1.0, "wind_direction_deg"] == 249
    assert pd.isna(ver.loc[3.0, "track_temp_c"]) and ver.loc[1.0, "air_temp_c"] == 27.3


def test_assign_rounds_orders_events_by_first_lap_date():
    a = session_frame(_payload(), 2025, "Chinese Grand Prix", "R")
    later = _payload()
    later["lSD"] = ["2025-04-06T05:03:00", "2025-04-06T05:04:30", "2025-04-06T05:06:00", "2025-04-06T05:03:01", "2025-04-06T05:04:31"]
    b = session_frame(later, 2025, "Japanese Grand Prix", "R")
    rounds = {f["event"].iloc[0]: int(f["round_number"].iloc[0]) for f in assign_rounds([b, a])}
    assert rounds == {"Chinese Grand Prix": 1, "Japanese Grand Prix": 2}


def test_raw_url_encodes_folder_names():
    assert raw_url(2025, "São Paulo Grand Prix", "Sprint Qualifying").endswith("/2025/main/S%C3%A3o%20Paulo%20Grand%20Prix/Sprint%20Qualifying/session_laptimes.json")
