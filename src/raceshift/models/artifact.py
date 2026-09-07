from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from raceshift.data.schema import REQUIRED_FORECAST_COLUMNS
from raceshift.features.full_context import build_full_context_table
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
        return {
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
