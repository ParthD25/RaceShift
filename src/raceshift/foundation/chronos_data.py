from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["season", "event", "session", "driver"]


@dataclass(frozen=True)
class WalkForwardExample:
    """A univariate lap-time history ending at lap N and the true lap N+1.

    Used to benchmark frozen time-series foundation models zero-shot. The context
    never contains lap N+1, matching the RaceShift leakage rule.
    """

    season: int
    event: str
    session: str
    driver: str
    lap_number: int
    context: np.ndarray
    actual_next: float


def make_walk_forward_examples(
    table: pd.DataFrame,
    context_length: int = 32,
    stride: int = 1,
    value_col: str = "lap_time_s",
    target_col: str = "target_next_lap_time_s",
    min_context: int = 4,
) -> list[WalkForwardExample]:
    """Slice each driver/session lap sequence into fixed-length walk-forward windows.

    The table must already be a next-lap table (rows carry `target_col` = lap N+1).
    Windows shorter than `context_length` but at least `min_context` long are kept
    so short stints still produce benchmark rows.
    """
    if context_length < 1 or stride < 1:
        raise ValueError("context_length and stride must be >= 1")
    missing = [c for c in GROUP_COLUMNS + ["lap_number", value_col, target_col] if c not in table.columns]
    if missing:
        raise ValueError(f"Missing columns for walk-forward examples: {missing}")

    examples: list[WalkForwardExample] = []
    for key, group in table.groupby(GROUP_COLUMNS, sort=False, dropna=False):
        group = group.sort_values("lap_number")
        values = group[value_col].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        laps = group["lap_number"].to_numpy()
        for end in range(min_context - 1, len(group), stride):
            start = max(0, end - context_length + 1)
            if not np.isfinite(targets[end]):
                continue
            examples.append(
                WalkForwardExample(
                    season=int(key[0]),
                    event=str(key[1]),
                    session=str(key[2]),
                    driver=str(key[3]),
                    lap_number=int(laps[end]),
                    context=values[start : end + 1].copy(),
                    actual_next=float(targets[end]),
                )
            )
    return examples


def example_to_chronos_input(example: WalkForwardExample) -> np.ndarray:
    """Chronos pipelines accept a 1-D float array per series."""
    return np.asarray(example.context, dtype=np.float32)
