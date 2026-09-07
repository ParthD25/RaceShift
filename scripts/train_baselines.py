#!/usr/bin/env python
"""Required RaceShift baselines on the same full-context table and split as RaceShift FFR.

Baselines (master spec section 18): previous lap, rolling-five median, ridge regression on
the residual target and gradient-boosted trees on the residual target. Metrics are written
as an artifact-style metrics.json so the local API lists them next to FFR runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

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
from raceshift.train.metrics import regression_metrics  # noqa: E402


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def evaluate(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float | int]:
    return regression_metrics(frame[RAW_TARGET_COLUMN].to_numpy(dtype=np.float64), predicted)


def main() -> None:
    p = argparse.ArgumentParser(description="RaceShift deterministic, linear and tree baselines.")
    p.add_argument("--input", required=True, help="CSV or Parquet lap table (same input as train_ffr.py)")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    p.add_argument("--output", default=str(ROOT / "artifacts" / "baselines"))
    p.add_argument("--train-end", type=int)
    p.add_argument("--val-year", type=int)
    p.add_argument("--test-year", type=int)
    p.add_argument("--data-source", help="Provenance label. Inferred when omitted.")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    split = cfg.get("split", {})
    train_end = args.train_end if args.train_end is not None else int(split["train_end"])
    val_year = args.val_year if args.val_year is not None else int(split["val_year"])
    test_year = args.test_year if args.test_year is not None else int(split["test_year"])
    history = int(cfg.get("history_laps", 5))

    raw = load_table(Path(args.input))
    data_source = args.data_source or infer_data_source(raw)
    table = build_full_context_table(raw, history=history)
    train, val, test = season_forward_split(table, train_end, val_year, test_year)
    trainval = pd.concat([train, val], ignore_index=True)

    contract = feature_contract(history=history)
    numeric = [c for c in contract.numeric if c in table.columns]
    categorical = [c for c in contract.categorical if c in table.columns]
    features = numeric + categorical
    assert_no_target_leakage(features)

    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, column in [("previous_lap", "lap_time_s"), ("rolling_median_5", "rolling_median_5")]:
        results[name] = {
            "validation": evaluate(val, val[column].to_numpy(dtype=np.float64)),
            "test": evaluate(test, test[column].to_numpy(dtype=np.float64)),
        }

    # Learned baselines predict the residual against the rolling-five median, like FFR.
    # Validation metrics use a train-only fit; test metrics refit on train+validation.
    hgb_cfg = cfg.get("hist_gradient_boosting", {})
    learners = {
        "ridge": lambda: Ridge(alpha=float(cfg.get("ridge_alpha", 2.0))),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(**hgb_cfg),
    }
    for name, make_model in learners.items():
        prep = make_preprocessor(numeric, categorical)
        model = make_model()
        model.fit(prep.fit_transform(train[features]), train[TARGET_COLUMN].to_numpy(dtype=np.float64))
        val_pred = val["rolling_median_5"].to_numpy(dtype=np.float64) + model.predict(prep.transform(val[features]))

        prep_full = make_preprocessor(numeric, categorical)
        model_full = make_model()
        model_full.fit(prep_full.fit_transform(trainval[features]), trainval[TARGET_COLUMN].to_numpy(dtype=np.float64))
        test_pred = test["rolling_median_5"].to_numpy(dtype=np.float64) + model_full.predict(prep_full.transform(test[features]))
        results[name] = {"validation": evaluate(val, val_pred), "test": evaluate(test, test_pred)}

    metrics = {
        "model": "baselines",
        "raceshift_version": __version__,
        "training_policy": "baseline-no-forward-forward",
        "global_backprop": False,
        "models": results,
        "split": {"train_end": train_end, "validation": val_year, "test": test_year},
        "feature_count_raw": len(features),
        "history_laps": history,
        "data_source": data_source,
        "is_synthetic": is_synthetic_source(data_source),
        "input_file": Path(args.input).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
