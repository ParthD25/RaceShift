from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class LapRecord:
    series: str
    season: int
    event: str
    session: str
    driver: str
    team: str | None
    lap_number: int
    lap_time_s: float | None
    sector1_s: float | None
    sector2_s: float | None
    sector3_s: float | None
    compound: str | None
    tyre_life: float | None
    stint: int | None
    position: int | None
    track_status: str | None
    pit_in: bool
    pit_out: bool
    air_temp_c: float | None
    track_temp_c: float | None
    humidity_pct: float | None
    pressure_mbar: float | None
    rainfall: bool | None
    wind_speed_ms: float | None
    gap_ahead_s: float | None
    gap_behind_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
