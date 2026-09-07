from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any

# Columns that every forecast input must carry. build_full_context_table enforces them too.
REQUIRED_FORECAST_COLUMNS = ["season", "event", "session", "driver", "lap_number", "lap_time_s"]


@dataclass(frozen=True)
class LapRecord:
    """One normalized completed lap. Mirrors section 22 of RACESHIFT_MASTER_SPEC.md.

    Adapters may leave optional fields as None. Feature engineering fills defaults.
    """

    series: str
    season: int
    event: str
    session: str
    driver: str
    lap_number: int
    round_number: int | None = None
    event_date: str | None = None
    circuit: str | None = None
    driver_number: str | None = None
    team: str | None = None
    manufacturer: str | None = None
    car_model: str | None = None
    car_class: str | None = None
    lap_time_s: float | None = None
    sector1_s: float | None = None
    sector2_s: float | None = None
    sector3_s: float | None = None
    compound: str | None = None
    tyre_manufacturer: str | None = None
    tyre_life: float | None = None
    fresh_tyre: bool | None = None
    stint: int | None = None
    position: int | None = None
    track_status: str | None = None
    pit_in: bool = False
    pit_out: bool = False
    is_accurate: bool = True
    deleted: bool = False
    air_temp_c: float | None = None
    track_temp_c: float | None = None
    humidity_pct: float | None = None
    pressure_mbar: float | None = None
    rainfall: bool | None = None
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    gap_ahead_s: float | None = None
    gap_behind_s: float | None = None
    mean_speed_kph: float | None = None
    max_speed_kph: float | None = None
    mean_throttle_pct: float | None = None
    brake_fraction: float | None = None
    mean_rpm: float | None = None
    max_rpm: float | None = None
    gear_changes: float | None = None
    drs_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA_COLUMNS = [f.name for f in fields(LapRecord)]
