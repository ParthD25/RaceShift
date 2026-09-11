"""The provider check matches laps one to one, on official rounds when both sides have them."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cross_provider_check import KEYS, ROUND_KEYS, compare, match_keys  # noqa: E402


def _table(event: str, rounds: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame({
        "season": [2026] * 4,
        "event": [event] * 4,
        "session": ["R"] * 4,
        "driver": ["VER", "VER", "VER", "NOR"],
        "lap_number": [1.0, 2.0, 3.0, 2.0],
        "lap_time_s": [100.0, 90.0, 90.5, 91.0],
        "compound": ["SOFT"] * 4,
    })
    if rounds:
        frame["round_number"] = 5
    return frame


def test_different_event_spellings_match_on_the_official_round():
    left, right = _table("Barcelona Grand Prix"), _table("Barcelona-Catalunya Grand Prix")
    assert match_keys(left, right) == ROUND_KEYS
    report = compare(left, right, ("fastf1", "ergast"))
    assert report["matched_on"] == ROUND_KEYS
    assert report["laps"]["shared"] == 3 and report["laps"]["left_only"] == 0 and report["laps"]["right_only"] == 0
    assert report["agreement"]["lap_time_s"]["agree_share"] == 1.0
    assert report["agreement"]["event"]["agree_share"] == 0.0
    assert match_keys(left, _table("x", rounds=False)) == KEYS


def test_opening_lap_is_excluded_from_every_count_and_duplicates_are_reduced():
    left, right = _table("A"), _table("A")
    right = pd.concat([right, right.iloc[[1]]], ignore_index=True)  # duplicated key
    right.loc[3, "driver"] = None  # missing key
    report = compare(left, right, ("l", "r"))
    laps = report["laps"]
    assert laps["left"] == 3 and laps["shared"] == 2 and laps["right_only"] == 0 and laps["left_only"] == 1
    assert laps["dropped_missing_key"] == {"left": 0, "right": 1}
    assert laps["dropped_duplicate_key"] == {"left": 0, "right": 1}
    full = compare(left, _table("A"), ("l", "r"), exclude_first_lap=False)
    assert full["laps"]["left"] == 4 and full["laps"]["shared"] == 4
