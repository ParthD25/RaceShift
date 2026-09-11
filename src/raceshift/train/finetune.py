"""Forward-forward fine-tuning helpers shared by ``scripts/finetune_ffr.py`` and
``scripts/score_artifact.py``.

Fine-tuning in RaceShift means exactly what training means: further local, layer-wise
updates of a saved model (``ForwardForwardRegressor.continue_fit``), followed by a
closed-form refit of the ridge readout. The base artifact's preprocessor and feature
contract are reused unchanged, so the fine-tuned model reads the same columns, scaled the
same way, as the model it started from; nothing is refitted on the new rows except the
layers and the readout.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from raceshift.features.full_context import LAP_VALIDITY_VERSION, RAW_TARGET_COLUMN, TARGET_COLUMN, build_full_context_table
from raceshift.models.artifact import RaceShiftArtifact
from raceshift.train.metrics import interval_metrics, regression_metrics


def parse_rounds(text: str | None) -> tuple[int, int] | None:
    """``"1-5"`` -> (1, 5); ``"8-"`` -> (8, 99); ``"6"`` -> (6, 6); None -> None."""
    if text is None or not str(text).strip():
        return None
    text = str(text).strip()
    if "-" in text:
        lo, hi = text.split("-", 1)
        return int(lo or 1), int(hi or 99)
    return int(text), int(text)


def select_rows(table: pd.DataFrame, season: int | tuple[int, int] | None, rounds: tuple[int, int] | None, sessions: list[str] | None) -> pd.DataFrame:
    """Rows of one season (or an inclusive season range), optionally limited to a round range
    and to session codes."""
    mask = pd.Series(True, index=table.index)
    if season is not None:
        years = pd.to_numeric(table["season"], errors="coerce")
        lo, hi = (season, season) if isinstance(season, int) else season
        mask &= (years >= int(lo)) & (years <= int(hi))
    if rounds is not None:
        r = pd.to_numeric(table["round_number"], errors="coerce")
        mask &= (r >= rounds[0]) & (r <= rounds[1])
    if sessions:
        mask &= table["session"].astype(str).isin(sessions)
    return table[mask].copy()


def contract_features(artifact: RaceShiftArtifact) -> list[str]:
    return list(artifact.contract["numeric"]) + list(artifact.contract["categorical"])


def build_table(artifact: RaceShiftArtifact, raw: pd.DataFrame, cache: str | Path | None = None) -> pd.DataFrame:
    """The full-context table with exactly the artifact's history length. Rows without a
    usable next lap are dropped (they cannot be trained on or scored).

    ``cache`` names a parquet that stores the built table for the same raw file and
    history, so a sweep of fine-tuning variants builds the features once. The cache is
    keyed on the raw table's shape and the history length; anything else is rebuilt.
    """
    key = f"rows={len(raw)} cols={len(raw.columns)} history={artifact.history_laps} lap_validity={LAP_VALIDITY_VERSION}"
    if cache is not None and Path(cache).exists():
        cached = pd.read_parquet(cache)
        if cached.attrs.get("raceshift_cache_key") == key or (Path(cache).with_suffix(".key").exists() and Path(cache).with_suffix(".key").read_text() == key):
            return cached
    table = build_full_context_table(raw, history=artifact.history_laps)
    table = table[table[TARGET_COLUMN].notna()].copy()
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, index=False)
        Path(cache).with_suffix(".key").write_text(key)
    return table


def encode(artifact: RaceShiftArtifact, rows: pd.DataFrame, target_clip: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    features = contract_features(artifact)
    missing = [c for c in features if c not in rows.columns]
    if missing:
        raise ValueError(f"The lap table lacks contract features {missing}")
    x = np.asarray(artifact.preprocessor.transform(rows[features]), dtype=np.float32)
    y = rows[TARGET_COLUMN].to_numpy(dtype=np.float32)
    if target_clip is not None:
        y = np.clip(y, -float(target_clip), float(target_clip))
    return x, y


def predict_frame(artifact: RaceShiftArtifact, rows: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    baseline = rows["rolling_median_5"].to_numpy(dtype=np.float64)
    out = artifact.model.predict_with_uncertainty(x)
    keep = [c for c in ["season", "round_number", "event", "session", "driver", "lap_number", "data_tier"] if c in rows.columns]
    frame = rows[keep].copy()
    frame["actual_next_lap_s"] = rows[RAW_TARGET_COLUMN].to_numpy(dtype=np.float64)
    frame["predicted_next_lap_s"] = baseline + out["prediction"].astype(np.float64)
    frame["lower_80_s"] = baseline + out["lower_80"].astype(np.float64)
    frame["upper_80_s"] = baseline + out["upper_80"].astype(np.float64)
    frame["layer_disagreement_s"] = out["layer_disagreement"].astype(np.float64)
    frame["rolling5_baseline_s"] = baseline
    frame["previous_lap_s"] = rows["lap_time_s"].to_numpy(dtype=np.float64)
    return frame


def evaluate(predictions: pd.DataFrame) -> dict[str, float | int]:
    metrics = regression_metrics(predictions["actual_next_lap_s"], predictions["predicted_next_lap_s"])
    metrics.update(interval_metrics(predictions["actual_next_lap_s"], predictions["lower_80_s"], predictions["upper_80_s"]))
    return metrics


def naive_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """The two training-free references every RaceShift table reports."""
    return {
        "previous_lap": regression_metrics(predictions["actual_next_lap_s"], predictions["previous_lap_s"]),
        "rolling_median_5": regression_metrics(predictions["actual_next_lap_s"], predictions["rolling5_baseline_s"]),
    }


def describe(rows: pd.DataFrame) -> dict[str, object]:
    info: dict[str, object] = {"rows": int(len(rows))}
    if len(rows):
        info["seasons"] = sorted(int(s) for s in pd.to_numeric(rows["season"], errors="coerce").dropna().unique())
        if "round_number" in rows.columns:
            info["rounds"] = sorted(int(r) for r in pd.to_numeric(rows["round_number"], errors="coerce").dropna().unique())
        info["events"] = int(rows.groupby(["season", "event"]).ngroups)
        if "data_tier" in rows.columns:
            info["data_tiers"] = rows["data_tier"].astype(str).value_counts().to_dict()
    return info


def copy_preprocessor(base: Path, target: Path) -> None:
    """The fine-tuned artifact reads the same inputs as its base: copy the fitted
    preprocessor (joblib and, when present, the pickle-free JSON spec)."""
    target.mkdir(parents=True, exist_ok=True)
    joblib_file = base / "preprocessor.joblib"
    if joblib_file.exists():
        shutil.copy2(joblib_file, target / "preprocessor.joblib")
    export = base / "export"
    if export.is_dir():
        (target / "export").mkdir(exist_ok=True)
        for spec in export.glob("*_preprocessor.json"):
            shutil.copy2(spec, target / "export" / spec.name)


def write_contract(base: RaceShiftArtifact, target: Path, extra: dict) -> None:
    contract = dict(base.contract)
    contract.update(extra)
    (target / "feature_contract.json").write_text(json.dumps(contract, indent=2))
