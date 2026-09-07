#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.splits import season_forward_split
from raceshift.features.lap_features import NUMERIC_FEATURES
from raceshift.train.metrics import regression_metrics


def main():
    p = argparse.ArgumentParser(description="RaceShift deterministic and tree baselines.")
    p.add_argument("--data", default=str(ROOT / "data" / "processed" / "next_lap.parquet"))
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    p.add_argument("--output", default=str(ROOT / "artifacts" / "baselines.json"))
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    data_path = Path(args.data)
    df = pd.read_csv(data_path) if data_path.suffix.lower() == ".csv" else pd.read_parquet(data_path)
    train, val, test = season_forward_split(df, cfg["split"]["train_end"], cfg["split"]["val_year"], cfg["split"]["test_year"])
    trainval = pd.concat([train, val], ignore_index=True)

    results = {}
    results["rolling_median_5"] = regression_metrics(test["target_next_lap_time_s"], test["rolling_median_5"])
    results["last_lap"] = regression_metrics(test["target_next_lap_time_s"], test["lap_time_s"])

    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in ["driver", "team", "event", "compound", "track_status"] if c in df.columns]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, random_state=7)
    pipe = Pipeline([("pre", pre), ("model", model)])
    features = numeric + categorical
    pipe.fit(trainval[features], trainval["target_delta_vs_rolling5_s"])
    delta = pipe.predict(test[features])
    pred = test["rolling_median_5"].to_numpy() + delta
    results["hist_gradient_boosting"] = regression_metrics(test["target_next_lap_time_s"], pred)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
