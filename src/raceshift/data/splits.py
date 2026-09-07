from __future__ import annotations

import pandas as pd


def season_forward_split(df: pd.DataFrame, train_end: int, val_year: int, test_year: int):
    """Chronological split. Never randomly mixes laps across the same future season."""
    train = df[df["season"] <= train_end].copy()
    val = df[df["season"] == val_year].copy()
    test = df[df["season"] == test_year].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("One or more chronological splits are empty.")
    return train, val, test


def leave_event_out(df: pd.DataFrame, event: str):
    test = df[df["event"] == event].copy()
    train = df[df["event"] != event].copy()
    if test.empty:
        raise ValueError(f"No rows found for holdout event: {event}")
    return train, test
