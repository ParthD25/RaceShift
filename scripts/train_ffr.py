#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from raceshift.data.splits import season_forward_split
from raceshift.features.full_context import (
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    assert_no_target_leakage,
    build_full_context_table,
    feature_contract,
)
from raceshift.models.forward_forward_regressor import FFRConfig, ForwardForwardRegressor


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", sparse_threshold=0.0)


def evaluate(model: ForwardForwardRegressor, x: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    residual_pred = model.predict(x)
    baseline = frame["rolling_median_5"].to_numpy(dtype=np.float32)
    actual = frame[RAW_TARGET_COLUMN].to_numpy(dtype=np.float32)
    predicted = baseline + residual_pred
    absolute = np.abs(actual - predicted)
    uncertainty = model.predict_with_uncertainty(x)
    lower = baseline + uncertainty["lower_80"]
    upper = baseline + uncertainty["upper_80"]
    return {
        "mae_s": float(mean_absolute_error(actual, predicted)),
        "rmse_s": float(mean_squared_error(actual, predicted) ** 0.5),
        "median_ae_s": float(np.median(absolute)),
        "p90_ae_s": float(np.quantile(absolute, 0.90)),
        "signed_bias_s": float(np.mean(predicted - actual)),
        "interval80_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "rows": int(len(frame)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train the RaceShift multi-layer Forward-Forward regressor.")
    p.add_argument("--input", required=True, help="CSV or Parquet lap table")
    p.add_argument("--config", default="configs/ffr_production.json")
    p.add_argument("--output", default="artifacts/raceshift_ffr")
    p.add_argument("--train-end", type=int, default=2023)
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    args = p.parse_args()

    raw = load_table(Path(args.input))
    cfg_json = json.loads(Path(args.config).read_text())
    history = int(cfg_json.pop("history_laps", 5))
    cfg_json.pop("model", None)
    cfg_json.pop("training_policy", None)
    cfg_json.pop("notes", None)
    cfg_json["layer_nodes"] = tuple(cfg_json["layer_nodes"])
    cfg_json["ordinal_groups"] = tuple(cfg_json["ordinal_groups"])
    cfg = FFRConfig(**cfg_json)

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

    metrics = {
        "model": "RaceShiftFFR",
        "training_policy": "forward-forward-local-updates-no-global-backprop",
        "architecture": model.architecture_summary(),
        "validation": evaluate(model, x_val, val),
        "test": evaluate(model, x_test, test),
        "split": {"train_end": args.train_end, "validation": args.val_year, "test": args.test_year},
        "feature_count_raw": len(features),
        "input_nodes_after_encoding": int(x_train.shape[1]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "feature_contract.json").write_text(json.dumps({"history": history, "numeric": numeric, "categorical": categorical}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
