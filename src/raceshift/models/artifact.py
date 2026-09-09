from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from raceshift.data.schema import REQUIRED_FORECAST_COLUMNS
from raceshift.features.full_context import RAW_TARGET_COLUMN, build_full_context_table
from raceshift.models.forward_forward_regressor import ForwardForwardRegressor

ARTIFACT_FILES = (
    "model_weights.npz",
    "model_config.json",
    "training_history.json",
    "preprocessor.joblib",
    "feature_contract.json",
    "metrics.json",
)


def missing_artifact_files(directory: str | Path) -> list[str]:
    """Names of required artifact members that are absent. Empty means complete."""
    path = Path(directory)
    return [name for name in ARTIFACT_FILES if not (path / name).is_file()]


def _chronological_order(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["season", "round_number", "event_date", "event", "session", "lap_number"] if c in frame.columns]


class RaceShiftArtifact:
    """A saved RaceShift FFR model plus its train-only preprocessor and feature contract."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        missing = missing_artifact_files(self.directory)
        if missing:
            raise FileNotFoundError(f"Artifact {self.directory.name} is incomplete, missing: {missing}")
        self.model = ForwardForwardRegressor.load(self.directory)
        self.preprocessor = joblib.load(self.directory / "preprocessor.joblib")
        self.contract = json.loads((self.directory / "feature_contract.json").read_text())
        self.metrics = json.loads((self.directory / "metrics.json").read_text())

    @property
    def is_synthetic(self) -> bool:
        return bool(self.metrics.get("is_synthetic", False))

    @property
    def data_source(self) -> str:
        return str(self.metrics.get("data_source", "unknown"))

    @staticmethod
    def latest_session(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
        """Return the rows of the chronologically latest season/event/session and its key."""
        ordered = raw.sort_values(_chronological_order(raw), kind="stable")
        last = ordered.iloc[-1]
        key = {"season": last["season"], "event": last["event"], "session": last["session"]}
        mask = (
            (raw["season"] == key["season"]) & (raw["event"] == key["event"]) & (raw["session"] == key["session"])
        )
        return raw[mask], key

    def forecast_last_available(self, raw: pd.DataFrame, driver: str | None = None) -> dict:
        """Forecast the next lap for the latest completed lap in the supplied history.

        Scope is the chronologically latest session in the file. When `driver` is not
        given, the driver who has completed the most laps in that session is used.
        Historical priors still see the whole file because earlier events feed them.

        The training target requires lap N+1, so a temporary sentinel copy of the final
        lap is appended to make feature generation emit that lap as an inference row.
        The sentinel target is discarded and never reaches the model.
        """
        missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        data = raw.copy()
        data = data[pd.to_numeric(data["lap_time_s"], errors="coerce").notna()]
        if data.empty:
            raise ValueError("No rows with a lap time available for forecast")

        scope, key = self.latest_session(data)
        scope = scope.copy()
        scope["driver"] = scope["driver"].astype(str)
        available_drivers = sorted(scope["driver"].unique().tolist())
        if driver is None:
            laps_by_driver = scope.groupby("driver")["lap_number"].max().sort_values(ascending=False, kind="stable")
            driver = str(laps_by_driver.index[0])
        driver = str(driver)
        if driver not in available_drivers:
            raise ValueError(f"Driver {driver!r} not found in the latest session. Available: {available_drivers}")

        history = scope[scope["driver"] == driver].sort_values("lap_number")
        if len(history) < 2:
            raise ValueError("Need at least two completed laps for the selected driver to forecast the next one")
        latest = history.iloc[-1].copy()

        sentinel = latest.copy()
        sentinel["lap_number"] = float(latest["lap_number"]) + 1
        for c in ["pit_in", "pit_out", "deleted"]:
            if c in sentinel.index:
                sentinel[c] = False
        if "is_accurate" in sentinel.index:
            sentinel["is_accurate"] = True
        augmented = pd.concat([data, pd.DataFrame([sentinel])], ignore_index=True)

        history_laps = int(self.contract.get("history", 5))
        table = build_full_context_table(augmented, history=history_laps)
        candidates = table[
            (table["season"] == key["season"])
            & (table["event"] == key["event"])
            & (table["session"] == key["session"])
            & (table["driver"].astype(str) == driver)
        ].sort_values("lap_number")
        if candidates.empty or int(candidates.iloc[-1]["lap_number"]) != int(latest["lap_number"]):
            raise ValueError(
                "The latest lap for this driver is not a usable completed lap (pit, deleted or inaccurate). "
                "Import more completed laps or choose another driver."
            )
        row = candidates.iloc[-1:]

        features = self.contract["numeric"] + self.contract["categorical"]
        x = np.asarray(self.preprocessor.transform(row[features]), dtype=np.float32)
        pred = self.model.predict_with_uncertainty(x)
        baseline = float(row["rolling_median_5"].iloc[0])
        predicted = baseline + float(pred["prediction"][0])
        if not np.isfinite(predicted):
            raise ValueError("Model produced a non-finite forecast for this row")

        completed = int(len(history))

        def _num(col: str):
            if col not in row.columns:
                return None
            value = row[col].iloc[0]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def _text(col: str):
            if col not in row.columns:
                return None
            value = row[col].iloc[0]
            return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

        context = {
            "compound": _text("compound"),
            "tyre_life": _num("tyre_life"),
            "stint": _num("stint"),
            "position": _num("position"),
            "laps_in_segment": _num("laps_in_segment"),
            "track_temp_c": _num("track_temp_c"),
            "air_temp_c": _num("air_temp_c"),
            "humidity_pct": _num("humidity_pct"),
            "wind_speed_ms": _num("wind_speed_ms"),
            "wind_direction_deg": _num("wind_direction_deg"),
            "rainfall": _text("rainfall"),
            "gap_ahead_s": _num("gap_ahead_s"),
            "gap_behind_s": _num("gap_behind_s"),
            "team": _text("team"),
            "circuit": _text("circuit"),
        }
        historical_context = {
            "driver_circuit_pace_s": _num("hist_driver_circuit_pace"),
            "team_circuit_pace_s": _num("hist_team_circuit_pace"),
            "compound_circuit_pace_s": _num("hist_compound_circuit_pace"),
            "driver_circuit_compound_pace_s": _num("hist_driver_circuit_compound_pace"),
            "matched_weather_compound_pace_s": _num("hist_weather_compound_pace"),
            "driver_matched_weather_pace_s": _num("hist_driver_weather_pace"),
            "driver_overall_pace_s": _num("hist_driver_global_pace"),
            "team_overall_pace_s": _num("hist_team_global_pace"),
            "note": "Medians of earlier events only. None means no earlier event matched.",
        }
        return {
            "context": context,
            "historical_context": historical_context,
            "season": int(key["season"]),
            "event": str(key["event"]),
            "session": str(key["session"]),
            "driver": driver,
            "available_drivers": available_drivers,
            "lap_number_completed": int(row["lap_number"].iloc[0]),
            "completed_laps_in_session": completed,
            "history_laps_used": history_laps,
            "short_history": completed < history_laps,
            "predicted_next_lap_s": predicted,
            "lower_80_s": baseline + float(pred["lower_80"][0]),
            "upper_80_s": baseline + float(pred["upper_80"][0]),
            "layer_disagreement_s": float(pred["layer_disagreement"][0]),
            "rolling5_baseline_s": baseline,
            "last_lap_time_s": float(latest["lap_time_s"]),
            "artifact": self.directory.name,
            "data_source": self.data_source,
            "is_synthetic": self.is_synthetic,
        }

    def backtest_session(self, raw: pd.DataFrame, driver: str | None = None, laps: int = 10) -> dict:
        """Score the model on the last ``laps`` completed lap pairs of one driver in the latest session.

        Every row is a lap N whose next lap N+1 was actually driven, so predicted and actual
        can be compared. Only pairs where both laps are valid racing laps count (pit, safety-car,
        red-flag and deleted laps are skipped, exactly as in training). Nothing here looks past
        lap N when predicting lap N+1: the feature table is the same leakage-safe table used for
        training and the model never sees the actual next lap.
        """
        if laps < 1:
            raise ValueError("laps must be >= 1")
        missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        data = raw.copy()
        data = data[pd.to_numeric(data["lap_time_s"], errors="coerce").notna()]
        if data.empty:
            raise ValueError("No rows with a lap time available for a backtest")

        scope, key = self.latest_session(data)
        scope = scope.copy()
        scope["driver"] = scope["driver"].astype(str)
        available_drivers = sorted(scope["driver"].unique().tolist())
        if driver is None:
            laps_by_driver = scope.groupby("driver")["lap_number"].max().sort_values(ascending=False, kind="stable")
            driver = str(laps_by_driver.index[0])
        driver = str(driver)
        if driver not in available_drivers:
            raise ValueError(f"Driver {driver!r} not found in the latest session. Available: {available_drivers}")
        driven = int((scope["driver"] == driver).sum())

        history_laps = int(self.contract.get("history", 5))
        table = build_full_context_table(data, history=history_laps)
        rows = table[
            (table["season"] == key["season"])
            & (table["event"] == key["event"])
            & (table["session"] == key["session"])
            & (table["driver"].astype(str) == driver)
        ].sort_values("lap_number")
        rows = rows[pd.to_numeric(rows[RAW_TARGET_COLUMN], errors="coerce").notna()]
        usable = int(len(rows))
        if rows.empty:
            raise ValueError(
                "No completed lap pairs for this driver where both laps are valid racing laps "
                "(pit, safety-car, red-flag and deleted laps are excluded, as in training)."
            )
        rows = rows.tail(int(laps))

        features = self.contract["numeric"] + self.contract["categorical"]
        x = np.asarray(self.preprocessor.transform(rows[features]), dtype=np.float32)
        pred = self.model.predict_with_uncertainty(x)
        baseline = rows["rolling_median_5"].to_numpy(dtype=float)
        predicted = baseline + np.asarray(pred["prediction"], dtype=float)
        lower = baseline + np.asarray(pred["lower_80"], dtype=float)
        upper = baseline + np.asarray(pred["upper_80"], dtype=float)
        actual = rows[RAW_TARGET_COLUMN].to_numpy(dtype=float)
        current = rows["lap_time_s"].to_numpy(dtype=float)
        error = predicted - actual
        abs_error = np.abs(error)
        inside = (actual >= lower) & (actual <= upper)

        def _opt(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        lap_rows = []
        for i in range(len(rows)):
            r = rows.iloc[i]
            lap_rows.append(
                {
                    "lap_number_completed": int(r["lap_number"]),
                    "next_lap_number": int(r["lap_number"]) + 1,
                    "actual_next_lap_s": float(actual[i]),
                    "predicted_next_lap_s": float(predicted[i]),
                    "lower_80_s": float(lower[i]),
                    "upper_80_s": float(upper[i]),
                    "rolling5_baseline_s": float(baseline[i]),
                    "previous_lap_s": float(current[i]),
                    "error_s": float(error[i]),
                    "abs_error_s": float(abs_error[i]),
                    "within_interval": bool(inside[i]),
                    "compound": None if pd.isna(r.get("compound")) else str(r.get("compound")),
                    "tyre_life": _opt(r.get("tyre_life")),
                    "position": _opt(r.get("position")),
                }
            )
        summary = {
            "rows": int(len(rows)),
            "mae_s": float(abs_error.mean()),
            "rmse_s": float(np.sqrt(np.mean(error**2))),
            "p90_ae_s": float(np.quantile(abs_error, 0.9)),
            "signed_bias_s": float(error.mean()),
            "within_0_5s_share": float((abs_error <= 0.5).mean()),
            "within_1s_share": float((abs_error <= 1.0).mean()),
            "interval80_coverage": float(inside.mean()),
            "previous_lap_mae_s": float(np.abs(current - actual).mean()),
            "rolling5_mae_s": float(np.abs(baseline - actual).mean()),
        }
        return {
            "season": int(key["season"]),
            "event": str(key["event"]),
            "session": str(key["session"]),
            "driver": driver,
            "available_drivers": available_drivers,
            "laps_driven": driven,
            "usable_lap_pairs": usable,
            "skipped_laps": max(driven - usable, 0),
            "laps": lap_rows,
            "summary": summary,
            "artifact": self.directory.name,
            "data_source": self.data_source,
            "is_synthetic": self.is_synthetic,
        }
