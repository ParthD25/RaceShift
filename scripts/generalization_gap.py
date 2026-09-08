#!/usr/bin/env python
"""Memorisation check: score a saved FFR artifact (and refit baselines) on train, validation and test.

A model that memorises its training laps shows a large gap between training error and
validation/test error. This script rebuilds the exact table and split an artifact was trained
on, applies the saved preprocessor and weights to every split, and writes the gaps next to
ridge and gradient-boosted-tree baselines refit on the same training rows.

Usage:
    python scripts/generalization_gap.py --input data/processed/f1_laps_fastf1.parquet \
        --artifacts artifacts/f1_2025h2_ffr-m artifacts/f1_2025h2_ffr-s --report f1_2025h2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.splits import split_from_args  # noqa: E402
from raceshift.features.full_context import RAW_TARGET_COLUMN, TARGET_COLUMN, build_full_context_table  # noqa: E402
from raceshift.features.preprocessing import make_preprocessor  # noqa: E402
from raceshift.models.forward_forward_regressor import ForwardForwardRegressor  # noqa: E402
from raceshift.train.metrics import regression_metrics  # noqa: E402


def score(frame: pd.DataFrame, residual_pred: np.ndarray) -> dict:
    pred = frame["rolling_median_5"].to_numpy(dtype=np.float64) + residual_pred.astype(np.float64)
    return regression_metrics(frame[RAW_TARGET_COLUMN].to_numpy(dtype=np.float64), pred)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--artifacts", nargs="+", required=True, help="FFR artifact directories trained on --input")
    p.add_argument("--report", required=True)
    p.add_argument("--reports-dir", default=str(ROOT / "reports"))
    p.add_argument("--skip-baselines", action="store_true")
    args = p.parse_args()

    first = json.loads((Path(args.artifacts[0]) / "metrics.json").read_text())
    sp = first["split"]
    clip = float(first.get("features", {}).get("target_clip_s", 6.0))
    history = int(first.get("features", {}).get("history_laps", 5))

    raw = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    table = build_full_context_table(raw, history=history)
    train, val, test = split_from_args(table, sp["train_end"], sp["validation"], sp["test"], sp.get("split_round"), sp.get("holdout_event"))
    splits = {"train": train, "validation": val, "test": test}

    rows: list[dict] = []
    for directory in args.artifacts:
        art = Path(directory)
        metrics = json.loads((art / "metrics.json").read_text())
        contract = json.loads((art / "feature_contract.json").read_text())
        features = contract["numeric"] + contract["categorical"]
        prep = joblib.load(art / "preprocessor.joblib")
        model = ForwardForwardRegressor.load(art)
        result = {"model": metrics.get("name", art.name), "family": "forward-forward"}
        for name, frame in splits.items():
            x = np.asarray(prep.transform(frame[features]), dtype=np.float32)
            result[name] = score(frame, model.predict(x))
        rows.append(result)

    if not args.skip_baselines:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge

        contract = json.loads((Path(args.artifacts[0]) / "feature_contract.json").read_text())
        numeric, categorical = contract["numeric"], contract["categorical"]
        features = numeric + categorical
        cfg = json.loads((ROOT / "configs" / "baseline.json").read_text())
        y_train = np.clip(train[TARGET_COLUMN].to_numpy(dtype=np.float64), -clip, clip)
        prep = make_preprocessor(numeric, categorical)
        x_train = prep.fit_transform(train[features])
        for name, model in [("ridge", Ridge(alpha=float(cfg.get("ridge_alpha", 2.0)))),
                            ("hist_gradient_boosting", HistGradientBoostingRegressor(**cfg.get("hist_gradient_boosting", {})))]:
            model.fit(x_train, y_train)
            result = {"model": name, "family": "baseline"}
            for split_name, frame in splits.items():
                x = x_train if split_name == "train" else prep.transform(frame[features])
                result[split_name] = score(frame, model.predict(x))
            rows.append(result)

    lines = [f"# Generalisation gap: {args.report}", "",
             f"Rows: train {len(train)}, validation {len(val)}, test {len(test)}. MAE in seconds on the true next lap. "
             "Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).", "",
             "| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        gap = r["test"]["mae_s"] - r["train"]["mae_s"]
        r["gap_test_minus_train_mae_s"] = round(gap, 4)
        lines.append(f"| {r['model']} | {r['train']['mae_s']:.3f} | {r['validation']['mae_s']:.3f} | {r['test']['mae_s']:.3f} | {gap:+.3f} | {r['train']['rmse_s']:.3f} | {r['test']['rmse_s']:.3f} |")
    lines.append("")
    lines.append("Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.")

    # Per-season error of the first FFR artifact on its own training seasons, so a higher
    # training error can be traced to specific seasons rather than read as a model defect.
    art = Path(args.artifacts[0])
    contract = json.loads((art / "feature_contract.json").read_text())
    features = contract["numeric"] + contract["categorical"]
    prep = joblib.load(art / "preprocessor.joblib")
    model = ForwardForwardRegressor.load(art)
    lines += ["", f"## {rows[0]['model']} error by season (training seasons are fit, later seasons are unseen)", "",
              "| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    by_season: dict[str, dict] = {}
    for split_name, frame in splits.items():
        x = np.asarray(prep.transform(frame[features]), dtype=np.float32)
        pred = frame["rolling_median_5"].to_numpy(dtype=np.float64) + model.predict(x).astype(np.float64)
        err = np.abs(pred - frame[RAW_TARGET_COLUMN].to_numpy(dtype=np.float64))
        for season, idx in frame.groupby("season").indices.items():
            e = err[idx]
            by_season[f"{season}/{split_name}"] = {"season": int(season), "split": split_name, "rows": int(len(e)), "mae_s": float(e.mean()), "rmse_s": float(np.sqrt((e ** 2).mean())), "share_over_5s": float((e > 5).mean())}
    order = {"train": 0, "validation": 1, "test": 2}
    for key in sorted(by_season, key=lambda k: (by_season[k]["season"], order[by_season[k]["split"]])):
        r = by_season[key]
        lines.append(f"| {r['season']} | {r['split']} | {r['rows']} | {r['mae_s']:.3f} | {r['rmse_s']:.3f} | {100 * r['share_over_5s']:.2f}% |")
    rows.append({"model": rows[0]["model"], "by_season": by_season})

    report_dir = Path(args.reports_dir) / args.report
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "generalization.md").write_text("\n".join(lines) + "\n")
    (report_dir / "generalization.json").write_text(json.dumps({"split": sp, "rows": rows}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
