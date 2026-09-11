"""Offline tests for the OpenF1 adapter: race-control messages become FastF1 status codes,
deleted-lap messages are matched, and a whole session assembles from stub endpoint data."""
from __future__ import annotations

import pandas as pd

from raceshift.data.openf1_loader import (
    DATA_TIER,
    OpenF1Client,
    deleted_laps,
    lap_track_status,
    session_frame,
    track_status_intervals,
)

T0 = pd.Timestamp("2025-03-16T04:00:00Z")


def _at(seconds: float) -> str:
    return (T0 + pd.Timedelta(seconds=seconds)).isoformat()


def test_race_control_becomes_fastf1_status_codes():
    messages = [
        {"date": _at(10), "category": "Flag", "flag": "YELLOW", "scope": "Sector", "sector": 3, "message": "YELLOW IN TRACK SECTOR 3"},
        {"date": _at(40), "category": "Flag", "flag": "CLEAR", "scope": "Sector", "sector": 3, "message": "CLEAR IN TRACK SECTOR 3"},
        {"date": _at(100), "category": "SafetyCar", "flag": None, "scope": None, "message": "SAFETY CAR DEPLOYED"},
        {"date": _at(120), "category": "Flag", "flag": "DOUBLE YELLOW", "scope": "Sector", "sector": 1, "message": "DOUBLE YELLOW IN TRACK SECTOR 1"},
        {"date": _at(200), "category": "SafetyCar", "flag": None, "scope": None, "message": "SAFETY CAR IN THIS LAP"},
        {"date": _at(260), "category": "Flag", "flag": "GREEN", "scope": "Track", "sector": None, "message": "GREEN LIGHT"},
        {"date": _at(400), "category": "SafetyCar", "flag": None, "scope": None, "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": _at(450), "category": "SafetyCar", "flag": None, "scope": None, "message": "VIRTUAL SAFETY CAR ENDING"},
        {"date": _at(470), "category": "Flag", "flag": "CLEAR", "scope": "Track", "sector": None, "message": "TRACK CLEAR"},
        {"date": _at(600), "category": "Flag", "flag": "RED", "scope": "Track", "sector": None, "message": "RED FLAG"},
    ]
    intervals = track_status_intervals(messages, T0 + pd.Timedelta(seconds=700))
    codes = sorted({code for _, _, code in intervals})
    assert codes == ["2", "4", "5", "6", "7"]
    # The sector yellow raised under the safety car is not reported separately, as in FastF1.
    yellows = [(s, e) for s, e, c in intervals if c == "2"]
    assert len(yellows) == 1 and yellows[0][0] == T0 + pd.Timedelta(seconds=10)

    starts = pd.Series([T0 + pd.Timedelta(seconds=s) for s in (0, 90, 180, 300, 420, 610)])
    ends = starts + pd.Timedelta(seconds=80)
    status = lap_track_status(starts, ends, intervals).tolist()
    # The fifth lap runs under the VSC, its ending phase, and then a clear track: all three codes.
    assert status == ["12", "14", "4", "1", "167", "5"]


def test_start_behind_the_safety_car_is_a_safety_car_period():
    messages = [
        {"date": _at(-300), "category": "Other", "flag": None, "scope": None, "message": "RACE WILL START BEHIND THE SAFETY CAR"},
        {"date": _at(0), "category": "SessionStatus", "flag": None, "scope": None, "message": "SESSION STARTED"},
        {"date": _at(500), "category": "Other", "flag": None, "scope": None, "message": "ROLLING START"},
    ]
    intervals = track_status_intervals(messages, T0 + pd.Timedelta(seconds=3000))
    assert [(int((s - T0).total_seconds()), int((e - T0).total_seconds()), c) for s, e, c in intervals] == [(0, 500, "4")]
    starts = pd.Series([T0 + pd.Timedelta(seconds=s) for s in (0, 180, 360, 540)])
    assert lap_track_status(starts, starts + pd.Timedelta(seconds=170), intervals).tolist() == ["4", "4", "14", "1"]


def test_session_abort_is_a_red_flag_that_closes_the_safety_car():
    messages = [
        {"date": _at(0), "category": "SessionStatus", "flag": None, "scope": None, "message": "SESSION STARTED"},
        {"date": _at(200), "category": "SafetyCar", "flag": None, "scope": None, "message": "SAFETY CAR DEPLOYED"},
        {"date": _at(260), "category": "SessionStatus", "flag": None, "scope": None, "message": "SESSION ABORTED"},
        {"date": _at(2000), "category": "SessionStatus", "flag": None, "scope": None, "message": "SESSION STARTED"},
    ]
    intervals = sorted(track_status_intervals(messages, T0 + pd.Timedelta(seconds=5000)))
    assert [(int((s - T0).total_seconds()), int((e - T0).total_seconds()), c) for s, e, c in intervals] == [(200, 260, "4"), (260, 2000, "5")]
    starts = pd.Series([T0 + pd.Timedelta(seconds=s) for s in (100, 2100)])
    assert lap_track_status(starts, starts + pd.Timedelta(seconds=180), intervals).tolist() == ["145", "1"]


def test_deleted_lap_messages_match_time_or_lap_number():
    laps = pd.DataFrame({
        "driver_number": [1, 1, 22],
        "lap_number": [10, 11, 5],
        "lap_time_s": [91.5, 91.9, 95.123],
        "date_start": [T0, T0 + pd.Timedelta(seconds=91.5), T0],
    })
    messages = [
        {"date": _at(300), "message": "CAR 22 (TSU) TIME 1:35.123 DELETED - TRACK LIMITS AT TURN 14 LAP 5"},
        {"date": _at(300), "message": "CAR 1 (VER) LAP DELETED - TRACK LIMITS AT TURN 8 LAP 11"},
    ]
    assert deleted_laps(messages, laps, laps["date_start"]).tolist() == [False, True, True]


class _StubClient(OpenF1Client):
    """Serves canned endpoint payloads instead of the network."""

    def __init__(self, payloads: dict[str, list[dict]]):
        super().__init__(cache_dir=None)
        self.payloads = payloads

    def get(self, endpoint: str, **params):  # noqa: D401 - test stub
        return self.payloads.get(endpoint, [])


def test_session_frame_assembles_the_raceshift_schema():
    laps = []
    for lap in range(1, 6):
        laps.append({
            "driver_number": 1, "lap_number": lap, "date_start": _at(90 * (lap - 1)), "lap_duration": 90.0 + lap * 0.1,
            "duration_sector_1": 30.0, "duration_sector_2": 30.0, "duration_sector_3": 30.0 + lap * 0.1,
            "is_pit_out_lap": lap == 4,
        })
    payloads = {
        "laps": laps,
        "drivers": [{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"}],
        "stints": [
            {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3, "compound": "MEDIUM", "tyre_age_at_start": 0},
            {"driver_number": 1, "stint_number": 2, "lap_start": 4, "lap_end": 5, "compound": "HARD", "tyre_age_at_start": 2},
        ],
        "pit": [{"driver_number": 1, "lap_number": 3, "pit_duration": 22.1}],
        "weather": [{"date": _at(0), "air_temperature": 20.0, "track_temperature": 30.0, "humidity": 50.0, "pressure": 1010.0, "rainfall": 0, "wind_speed": 2.0, "wind_direction": 90}],
        "race_control": [{"date": _at(100), "category": "Flag", "flag": "YELLOW", "scope": "Sector", "sector": 2, "message": "YELLOW"},
                         {"date": _at(150), "category": "Flag", "flag": "CLEAR", "scope": "Sector", "sector": 2, "message": "CLEAR"}],
        "position": [{"driver_number": 1, "date": _at(0), "position": 3}, {"driver_number": 1, "date": _at(200), "position": 1}],
    }
    meeting = {"meeting_key": 1, "meeting_name": "Test Grand Prix", "location": "Yas Marina", "date_end": "2025-03-16T06:00:00+00:00", "round_number": 7, "year": 2025}
    session = {"session_key": 9, "session_code": "R", "year": 2025, "date_end": "2025-03-16T06:00:00+00:00"}
    frame = session_frame(_StubClient(payloads), meeting, session)

    assert len(frame) == 5 and frame["data_tier"].eq(DATA_TIER).all()
    assert frame["circuit"].iloc[0] == "Yas Island" and frame["round_number"].iloc[0] == 7 and frame["event_date"].iloc[0] == "2025-03-16"
    assert frame["driver"].tolist() == ["VER"] * 5 and frame["team"].iloc[0] == "Red Bull Racing"
    assert frame["compound"].tolist() == ["MEDIUM"] * 3 + ["HARD"] * 2
    assert frame["tyre_life"].tolist() == [1.0, 2.0, 3.0, 3.0, 4.0]
    assert frame["fresh_tyre"].tolist() == [True, True, True, False, False]
    assert frame["pit_in"].tolist() == [False, False, True, False, False]
    assert frame["pit_out"].tolist() == [False, False, False, True, False]
    assert frame["track_status"].tolist() == ["1", "12", "1", "1", "1"]
    assert frame["is_accurate"].tolist() == [False, True, False, False, True]
    assert frame["position"].tolist() == [3.0, 3.0, 1.0, 1.0, 1.0]
    assert frame["air_temp_c"].iloc[0] == 20.0 and frame["wind_direction_deg"].iloc[0] == 90
