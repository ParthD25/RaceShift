from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from raceshift.features.full_context import build_full_context_table
from raceshift.models.forward_forward_regressor import ForwardForwardRegressor


class RaceShiftArtifact:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.model = ForwardForwardRegressor.load(self.directory)
        self.preprocessor = joblib.load(self.directory / "preprocessor.joblib")
        self.contract = json.loads((self.directory / "feature_contract.json").read_text())

    def forecast_last_available(self, raw: pd.DataFrame, driver: str | None = None) -> dict:
        """Forecast the next lap using the latest feature row available in a supplied history.

        The training target requires N+1, so we append a temporary sentinel copy to make
        feature generation produce the final completed lap as an inference row. The
        sentinel target is discarded and never reaches the model.
        """
        data = raw.copy()
        if driver is not None:
            data = data[data["driver"].astype(str) == str(driver)].copy()
        if data.empty:
            raise ValueError("No rows available for forecast")

        latest = data.sort_values(["season", "event", "session", "driver", "lap_number"]).iloc[-1].copy()
        sentinel = latest.copy()
        sentinel["lap_number"] = float(latest["lap_number"]) + 1
        sentinel["lap_time_s"] = float(latest["lap_time_s"])
        for c in ["sector1_s", "sector2_s", "sector3_s"]:
            if c in sentinel:
                sentinel[c] = latest.get(c)
        augmented = pd.concat([data, pd.DataFrame([sentinel])], ignore_index=True)
        history = int(self.contract.get("history", 5))
        table = build_full_context_table(augmented, history=history)
        candidates = table[table["driver"].astype(str) == str(latest["driver"])].sort_values("lap_number")
        row = candidates.iloc[-1:]

        features = self.contract["numeric"] + self.contract["categorical"]
        x = np.asarray(self.preprocessor.transform(row[features]), dtype=np.float32)
        pred = self.model.predict_with_uncertainty(x)
        baseline = float(row["rolling_median_5"].iloc[0])
        return {
            "driver": str(row["driver"].iloc[0]),
            "event": str(row["event"].iloc[0]),
            "lap_number_completed": int(row["lap_number"].iloc[0]),
            "predicted_next_lap_s": baseline + float(pred["prediction"][0]),
            "lower_80_s": baseline + float(pred["lower_80"][0]),
            "upper_80_s": baseline + float(pred["upper_80"][0]),
            "layer_disagreement_s": float(pred["layer_disagreement"][0]),
            "rolling5_baseline_s": baseline,
        }
