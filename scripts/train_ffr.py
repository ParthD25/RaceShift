#!/usr/bin/env python
"""Train the RaceShift multi-layer Forward-Forward regressor with local layer updates only."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift import __version__  # noqa: E402
from raceshift.data.provenance import infer_data_source, is_synthetic_source  # noqa: E402
from raceshift.data.splits import season_forward_split  # noqa: E402
from raceshift.features.full_context import (  # noqa: E402
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    assert_no_target_leakage,
    build_full_context_table,
    feature_contract,
)
from raceshift.features.preprocessing import make_preprocessor  # noqa: E402
from raceshift.models.forward_forward_regressor import FFRConfig, ForwardForwardRegressor  # noqa: E402
from raceshift.train.metrics import interval_metrics, regression_metrics  # noqa: E402

CONFIG_METADATA_KEYS = ("history_laps", "model", "training_policy", "notes")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_config(path: Path) -> tuple[FFRConfig, int]:
    cfg_json = json.loads(path.read_text())
    history = int(cfg_json.get("history_laps", 5))
    params = {k: v for k, v in cfg_json.items() if k not in CONFIG_METADATA_KEYS}
    unknown = sorted(set(params) - set(FFRConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown FFR config keys in {path.name}: {unknown}")
    params["layer_nodes"] = tuple(params["layer_nodes"])
    params["ordinal_groups"] = tuple(params["ordinal_groups"])
    return FFRConfig(**params), history


def predict_frame(model: ForwardForwardRegressor, x: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame["rolling_median_5"].to_numpy(dtype=np.float64)
    uncertainty = model.predict_with_uncertainty(x)
    out = frame[[c for c in ["season", "event", "session", "driver", "lap_number"] if c in frame.columns]].copy()
    out["actual_next_lap_s"] = frame[RAW_TARGET_COLUMN].to_numpy(dtype=np.float64)
    out["predicted_next_lap_s"] = baseline + uncertainty["prediction"].astype(np.float64)
    out["lower_80_s"] = baseline + uncertainty["lower_80"].astype(np.float64)
    out["upper_80_s"] = baseline + uncertainty["upper_80"].astype(np.float64)
    out["layer_disagreement_s"] = uncertainty["layer_disagreement"].astype(np.float64)
    out["rolling5_baseline_s"] = baseline
    return out


def evaluate(predictions: pd.DataFrame) -> dict[str, float | int]:
    metrics = regression_metrics(predictions["actual_next_lap_s"], predictions["predicted_next_lap_s"])
    metrics.update(interval_metrics(predictions["actual_next_lap_s"], predictions["lower_80_s"], predictions["upper_80_s"]))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train the RaceShift multi-layer Forward-Forward regressor.")
    p.add_argument("--input", required=True, help="CSV or Parquet lap table")
    p.add_argument("--config", default=str(ROOT / "configs" / "ffr_production.json"))
    p.add_argument("--output", default=str(ROOT / "artifacts" / "raceshift_ffr"))
    p.add_argument("--train-end", type=int, default=2023)
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--data-source", help="Provenance label stored in metrics.json. Inferred when omitted.")
    args = p.parse_args()

    raw = load_table(Path(args.input))
    data_source = args.data_source or infer_data_source(raw)
    cfg, history = load_config(Path(args.config))

    table = build_full_context_table(raw, history=history)
    train, val, test = season_forward_split(table, args.train_end, args.val_year, args.test_year)
    contract = feature_contract(history=history)
    numeric = [c for c in contract.numeric if c in table.columns]
    categorical = [c for c in contract.categorical if c in table.columns]
    features = numeric + categorical
    assert_no_target_leakage(features)

    prep = make_preprocessor(numeric, categorical)
    x_train = np.asarray(prep.fit_transform(train[features]), dtype=np.float32)
    x_val = np.asarray(prep.transform(val[features]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[features]), dtype=np.float32)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float32)

    model = ForwardForwardRegressor(x_train.shape[1], cfg)
    model.fit(x_train, y_train, validation=(x_val, y_val))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    joblib.dump(prep, out / "preprocessor.joblib")

    val_predictions = predict_frame(model, x_val, val)
    test_predictions = predict_frame(model, x_test, test)
    test_predictions.to_csv(out / "test_predictions.csv", index=False)

    metrics = {
        "model": "RaceShiftFFR",
        "raceshift_version": __version__,
        "training_policy": "forward-forward-local-updates-no-global-backprop",
        "global_backprop": False,
        "config_file": Path(args.config).name,
        "hyperparameters": json.loads(json.dumps(cfg.__dict__)),
        "architecture": model.architecture_summary(),
        "validation": evaluate(val_predictions),
        "test": evaluate(test_predictions),
        "split": {"train_end": args.train_end, "validation": args.val_year, "test": args.test_year},
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "feature_count_raw": len(features),
        "feature_contract_version": "full_context_v1",
        "history_laps": history,
        "input_nodes_after_encoding": int(x_train.shape[1]),
        "data_source": data_source,
        "is_synthetic": is_synthetic_source(data_source),
        "input_file": Path(args.input).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "feature_contract.json").write_text(
        json.dumps({"history": history, "numeric": numeric, "categorical": categorical}, indent=2)
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
