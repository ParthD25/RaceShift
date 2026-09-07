#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def build_fixture(seasons=(2022, 2023, 2024, 2025), events_per_season=3, drivers=("AAA", "BBB", "CCC", "DDD"), laps=28, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    teams = {"AAA": "SYN-A", "BBB": "SYN-A", "CCC": "SYN-B", "DDD": "SYN-B"}
    circuits = ["Synthetic_Ring", "Synthetic_Park", "Synthetic_Street"]
    for season in seasons:
        for event_idx in range(events_per_season):
            event = f"Synthetic_GP_{event_idx + 1}"
            circuit = circuits[event_idx % len(circuits)]
            event_date = f"{season}-{3 + event_idx * 2:02d}-15"
            base_event = 88.0 + 1.6 * event_idx + 0.12 * (season - min(seasons))
            for d_idx, driver in enumerate(drivers):
                driver_delta = 0.16 * d_idx
                for lap in range(1, laps + 1):
                    tyre_life = float((lap - 1) % 14 + 1)
                    compound = "MEDIUM" if tyre_life <= 8 else "HARD"
                    track_temp = 31 + 4 * np.sin(lap / 7) + 1.2 * event_idx + rng.normal(0, 0.5)
                    air_temp = 23 + 0.8 * event_idx + rng.normal(0, 0.3)
                    humidity = 46 - 2 * event_idx + rng.normal(0, 1.0)
                    wind_speed = 2.5 + 0.5 * event_idx + rng.normal(0, 0.25)
                    wind_direction = float((70 + 22 * lap + 35 * event_idx + rng.normal(0, 8)) % 360)
                    degradation = (0.040 if compound == "HARD" else 0.060) * tyre_life
                    weather_penalty = 0.014 * (track_temp - 31) + 0.012 * wind_speed
                    team_delta = -0.12 if teams[driver] == "SYN-A" else 0.10
                    lap_time = base_event + driver_delta + team_delta + degradation + weather_penalty + rng.normal(0, 0.14)
                    s1 = lap_time * 0.31 + rng.normal(0, 0.03)
                    s2 = lap_time * 0.37 + rng.normal(0, 0.03)
                    s3 = lap_time - s1 - s2
                    mean_speed = 210.0 - (lap_time - base_event) * 4.0 + rng.normal(0, 1.5)
                    rows.append({
                        "series": "F1",
                        "season": season,
                        "round_number": event_idx + 1,
                        "event_date": event_date,
                        "event": event,
                        "circuit": circuit,
                        "session": "R",
                        "driver": driver,
                        "team": teams[driver],
                        "manufacturer": teams[driver],
                        "car_class": "F1",
                        "lap_number": lap,
                        "lap_time_s": lap_time,
                        "sector1_s": s1,
                        "sector2_s": s2,
                        "sector3_s": s3,
                        "compound": compound,
                        "tyre_manufacturer": "Pirelli",
                        "tyre_life": tyre_life,
                        "fresh_tyre": 1 if tyre_life == 1 else 0,
                        "stint": 1,
                        "position": d_idx + 1,
                        "track_status": "1",
                        "pit_in": False,
                        "pit_out": False,
                        "is_accurate": True,
                        "deleted": False,
                        "air_temp_c": air_temp,
                        "track_temp_c": track_temp,
                        "humidity_pct": humidity,
                        "pressure_mbar": 1010 + rng.normal(0, 1.5),
                        "rainfall": False,
                        "wind_speed_ms": wind_speed,
                        "wind_direction_deg": wind_direction,
                        "gap_ahead_s": max(0.15, 1.0 + 0.3 * d_idx + rng.normal(0, 0.1)),
                        "gap_behind_s": max(0.15, 1.1 + 0.25 * (len(drivers) - d_idx) + rng.normal(0, 0.1)),
                        "mean_speed_kph": mean_speed,
                        "max_speed_kph": 323 + rng.normal(0, 2.0),
                        "mean_throttle_pct": 66 + rng.normal(0, 2.0),
                        "brake_fraction": np.clip(0.18 + rng.normal(0, 0.01), 0.05, 0.4),
                        "mean_rpm": 10400 + rng.normal(0, 130),
                        "max_rpm": 11800 + rng.normal(0, 100),
                        "gear_changes": 48 + rng.integers(-3, 4),
                        "drs_fraction": np.clip(0.11 + rng.normal(0, 0.02), 0, 0.5),
                    })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw/synthetic_fixture.csv")
    args = p.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_fixture()
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
