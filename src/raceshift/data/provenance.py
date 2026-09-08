from __future__ import annotations

import pandas as pd

SYNTHETIC_EVENT_PREFIX = "Synthetic_"
SYNTHETIC_SOURCE = "synthetic_fixture"


def infer_data_source(raw: pd.DataFrame) -> str:
    """Label a lap table so metrics can never be mistaken for real Formula 1 results."""
    if "event" not in raw.columns or raw.empty:
        return "unknown"
    events = raw["event"].astype(str)
    if events.str.startswith(SYNTHETIC_EVENT_PREFIX).all():
        return SYNTHETIC_SOURCE
    if "data_tier" in raw.columns:
        # Files exported before the tier column existed are FastF1 timing by construction.
        tiers = sorted(str(t) for t in raw["data_tier"].fillna("fastf1_timing").unique())
        if tiers:
            return "+".join(tiers)
    return "user_supplied"


def is_synthetic_source(source: str | None) -> bool:
    return source == SYNTHETIC_SOURCE
