from __future__ import annotations

import pandas as pd


def season_forward_split(df: pd.DataFrame, train_end: int, val_year: int, test_year: int):
    """Chronological split by season. Never randomly mixes laps across the same future season."""
    if not (train_end < val_year <= test_year):
        raise ValueError("Expected train_end < val_year <= test_year")
    train = df[df["season"] <= train_end].copy()
    val = df[df["season"] == val_year].copy()
    test = df[df["season"] == test_year].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("One or more chronological splits are empty.")
    return train, val, test


def season_round_split(df: pd.DataFrame, train_end: int, holdout_year: int, cutoff_round: int, train_through_round: int | None = None):
    """Chronological split that carves one season into an early validation and late test half.

    train: season <= train_end; validation: holdout_year rounds <= cutoff_round;
    test: holdout_year rounds > cutoff_round. Requires a round_number column.

    With ``train_through_round`` the holdout season's rounds up to that number join the
    training split (the "retrain on the first races of the new season" setting) and
    validation is rounds in (train_through_round, cutoff_round].
    """
    if "round_number" not in df.columns:
        raise ValueError("season_round_split needs a round_number column")
    if train_end >= holdout_year:
        raise ValueError("Expected train_end < holdout_year")
    if train_through_round is not None and not (0 < train_through_round < cutoff_round):
        raise ValueError("Expected 0 < train_through_round < cutoff_round")
    rounds = pd.to_numeric(df["round_number"], errors="coerce")
    train = df[df["season"] <= train_end].copy()
    val_from = 0 if train_through_round is None else train_through_round
    if train_through_round is not None:
        early = df[(df["season"] == holdout_year) & (rounds <= train_through_round)].copy()
        train = pd.concat([train, early], ignore_index=False)
    val = df[(df["season"] == holdout_year) & (rounds > val_from) & (rounds <= cutoff_round)].copy()
    test = df[(df["season"] == holdout_year) & (rounds > cutoff_round)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("One or more chronological splits are empty.")
    return train, val, test


def chronological_split(df: pd.DataFrame, train_end: int, val_year: int, test_year: int, split_round: int | None = None, train_through_round: int | None = None):
    """Dispatch to the season-forward or season-round split.

    When `split_round` is given and val_year == test_year the holdout season is split at
    that round; otherwise validation and test are whole seasons.
    """
    if split_round is not None and val_year == test_year:
        return season_round_split(df, train_end, val_year, split_round, train_through_round)
    if train_through_round is not None:
        raise ValueError("train_through_round needs a season-round split (--split-round with val-year == test-year)")
    return season_forward_split(df, train_end, val_year, test_year)


def leave_event_out(df: pd.DataFrame, event: str):
    """Hold out every season of one event/circuit name."""
    test = df[df["event"] == event].copy()
    train = df[df["event"] != event].copy()
    if test.empty:
        raise ValueError(f"No rows found for holdout event: {event}")
    if train.empty:
        raise ValueError("Holdout would leave no training rows")
    return train, test


def split_from_args(
    df: pd.DataFrame,
    train_end: int,
    val_year: int,
    test_year: int,
    split_round: int | None = None,
    holdout_event: str | None = None,
    train_through_round: int | None = None,
):
    """Build (train, validation, test) for the experiment scripts.

    Default: chronological seasons (optionally carving the holdout season by round).
    With `holdout_event`: every season of that event becomes the test set (circuit
    holdout); the remaining rows are split into train (<= train_end) and validation
    (== val_year) chronologically.
    """
    if holdout_event:
        if train_through_round is not None:
            raise ValueError("train_through_round belongs to the season-round split and cannot be combined with a circuit holdout")
        rest, test = leave_event_out(df, holdout_event)
        train = rest[rest["season"] <= train_end].copy()
        val = rest[rest["season"] == val_year].copy()
        if train.empty or val.empty:
            raise ValueError("Circuit holdout left an empty train or validation split.")
        return train, val, test
    return chronological_split(df, train_end, val_year, test_year, split_round=split_round, train_through_round=train_through_round)
