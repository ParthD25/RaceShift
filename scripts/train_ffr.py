#!/usr/bin/env python
"""Train the RaceShift multi-layer Forward-Forward regressor with local layer updates only.

Records accuracy, calibration, training wall time, peak memory, inference latency and
artifact size so Forward-Forward can be judged on the resource axis it exists for.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift import __version__  # noqa: E402
from raceshift.data.provenance import infer_data_source, is_synthetic_source  # noqa: E402
from raceshift.data.splits import split_from_args  # noqa: E402
from raceshift.features.full_context import (  # noqa: E402
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    assert_no_target_leakage,
    build_full_context_table,
)
from raceshift.features.preprocessing import make_preprocessor  # noqa: E402
from raceshift.features.selection import ABLATION_GROUPS, drop_sparse_features, select_features  # noqa: E402
from raceshift.models.forward_forward_regressor import FFRConfig, ForwardForwardRegressor  # noqa: E402
from raceshift.train.metrics import interval_metrics, regression_metrics  # noqa: E402
from raceshift.train.resources import ResourceReport, directory_bytes, measure  # noqa: E402

CONFIG_METADATA_KEYS = ("name", "history_laps", "model", "training_policy", "notes")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_config(path: Path) -> tuple[FFRConfig, int, str]:
    cfg_json = json.loads(path.read_text())
    history = int(cfg_json.get("history_laps", 5))
    name = str(cfg_json.get("name", path.stem))
    params = {k: v for k, v in cfg_json.items() if k not in CONFIG_METADATA_KEYS}
    unknown = sorted(set(params) - set(FFRConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown FFR config keys in {path.name}: {unknown}")
    params["layer_nodes"] = tuple(params["layer_nodes"])
    params["ordinal_groups"] = tuple(params["ordinal_groups"])
    return FFRConfig(**params), history, name


def predict_frame(model: ForwardForwardRegressor, x: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame["rolling_median_5"].to_numpy(dtype=np.float64)
    uncertainty = model.predict_with_uncertainty(x)
    out = frame[[c for c in ["season", "round_number", "event", "session", "driver", "lap_number"] if c in frame.columns]].copy()
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


def breakdown(predictions: pd.DataFrame, by: str) -> dict[str, dict[str, float | int]]:
    if by not in predictions.columns:
        return {}
    out: dict[str, dict[str, float | int]] = {}
    for key, group in predictions.groupby(by, sort=True):
        if len(group) >= 20:
            out[str(key)] = regression_metrics(group["actual_next_lap_s"], group["predicted_next_lap_s"])
    return out


def log_to_wandb(run_name: str, cfg, metrics: dict, history: list[dict]) -> None:
    """Optional experiment tracking. Never required: import lazily and fail loudly but late."""
    import os

    import wandb  # type: ignore

    run = wandb.init(project=os.environ.get("RACESHIFT_WANDB_PROJECT", "raceshift"), name=run_name,
                     config={**metrics.get("hyperparameters", {}), "split": metrics.get("split"), "features": metrics.get("features")})
    for row in history:
        run.log({f"layer{row['layer']}/local_loss": row["local_loss"], "epoch": row["epoch"]})
    summary = {f"{split}/{k}": v for split in ("train", "validation", "test") for k, v in metrics.get(split, {}).items()}
    summary["gap/test_minus_train_mae_s"] = metrics["test"]["mae_s"] - metrics["train"]["mae_s"]
    summary.update({f"resources/{k}": v for k, v in metrics.get("resources", {}).get("training", {}).items()})
    run.summary.update(summary)
    run.finish()


def describe_split(frame: pd.DataFrame) -> dict[str, object]:
    info: dict[str, object] = {"rows": int(len(frame)), "seasons": sorted(int(s) for s in frame["season"].unique())}
    if "round_number" in frame.columns:
        info["rounds"] = {int(s): sorted(int(r) for r in pd.to_numeric(g["round_number"], errors="coerce").dropna().unique()) for s, g in frame.groupby("season")}
    info["events"] = int(frame.groupby(["season", "event"]).ngroups)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description="Train the RaceShift multi-layer Forward-Forward regressor.")
    p.add_argument("--input", required=True, help="CSV or Parquet lap table")
    p.add_argument("--config", default=str(ROOT / "configs" / "ffr_production.json"))
    p.add_argument("--output", default=str(ROOT / "artifacts" / "raceshift_ffr"))
    p.add_argument("--train-end", type=int, default=2023)
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--split-round", type=int, help="When val-year == test-year, validation = rounds <= this, test = rounds > this")
    p.add_argument("--holdout-event", help="Circuit holdout: every season of this event becomes the test set")
    p.add_argument("--drop-feature-group", action="append", default=[], choices=ABLATION_GROUPS, help="Ablate a feature group (repeatable)")
    p.add_argument("--name", help="Run name recorded in metrics.json (defaults to the config name)")
    p.add_argument("--data-source", help="Provenance label stored in metrics.json. Inferred when omitted.")
    p.add_argument("--target-clip", type=float, default=6.0, help="Winsorize the training residual target to +/- this many seconds (evaluation is never clipped)")
    p.add_argument("--min-feature-coverage", type=float, default=0.05, help="Drop numeric features observed in fewer than this share of training rows")
    p.add_argument("--wandb", action="store_true", help="Log per-layer local losses and final metrics to Weights & Biases (project RACESHIFT_WANDB_PROJECT or 'raceshift'; honours WANDB_MODE=offline)")
    args = p.parse_args()

    raw = load_table(Path(args.input))
    data_source = args.data_source or infer_data_source(raw)
    cfg, history, config_name = load_config(Path(args.config))
    run_name = args.name or config_name

    features_report = ResourceReport()
    with measure(features_report):
        table = build_full_context_table(raw, history=history)
    train, val, test = split_from_args(table, args.train_end, args.val_year, args.test_year, args.split_round, args.holdout_event)
    numeric, categorical = select_features(table.columns, history=history, drop_groups=args.drop_feature_group)
    numeric, sparse_dropped = drop_sparse_features(train, numeric, min_coverage=args.min_feature_coverage)
    features = numeric + categorical
    assert_no_target_leakage(features)

    prep = make_preprocessor(numeric, categorical)
    x_train = np.asarray(prep.fit_transform(train[features]), dtype=np.float32)
    x_val = np.asarray(prep.transform(val[features]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[features]), dtype=np.float32)
    # The residual target has heavy tails (restarts, traffic, damage). Training is winsorized
    # so a handful of laps cannot dominate the fit; evaluation always uses the raw target.
    clip = float(args.target_clip)
    y_train = np.clip(train[TARGET_COLUMN].to_numpy(dtype=np.float32), -clip, clip)
    y_val = np.clip(val[TARGET_COLUMN].to_numpy(dtype=np.float32), -clip, clip)

    model = ForwardForwardRegressor(x_train.shape[1], cfg)
    train_report = ResourceReport()
    with measure(train_report):
        model.fit(x_train, y_train, validation=(x_val, y_val))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    joblib.dump(prep, out / "preprocessor.joblib")

    # Inference latency on the test split (CPU, batch of one row repeated for stability).
    t0 = time.perf_counter()
    val_predictions = predict_frame(model, x_val, val)
    test_predictions = predict_frame(model, x_test, test)
    # Training-split error is recorded so every run exposes its generalisation gap
    # (test minus train MAE); a model that memorises shows a large positive gap.
    train_predictions = predict_frame(model, x_train, train)
    batch_ms_per_row = (time.perf_counter() - t0) * 1000.0 / max(1, len(x_val) + len(x_test))
    single = x_test[:1]
    t1 = time.perf_counter()
    for _ in range(50):
        model.predict_with_uncertainty(single)
    single_row_ms = (time.perf_counter() - t1) * 1000.0 / 50
    test_predictions.to_csv(out / "test_predictions.csv", index=False)

    metrics = {
        "model": "RaceShiftFFR",
        "name": run_name,
        "raceshift_version": __version__,
        "training_policy": "forward-forward-local-updates-no-global-backprop",
        "global_backprop": False,
        "config_file": Path(args.config).name,
        "hyperparameters": json.loads(json.dumps(cfg.__dict__)),
        "architecture": model.architecture_summary(),
        "train": evaluate(train_predictions),
        "validation": evaluate(val_predictions),
        "test": evaluate(test_predictions),
        "test_by_event": breakdown(test_predictions, "event"),
        "test_by_driver": breakdown(test_predictions, "driver"),
        "split": {
            "mode": "circuit_holdout" if args.holdout_event else ("season_round" if args.split_round and args.val_year == args.test_year else "season_forward"),
            "train_end": args.train_end,
            "validation": args.val_year,
            "test": args.test_year,
            "split_round": args.split_round,
            "holdout_event": args.holdout_event,
            "train_rows": describe_split(train),
            "validation_rows": describe_split(val),
            "test_rows": describe_split(test),
        },
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "features": {
            "count_raw": len(features),
            "numeric": len(numeric),
            "categorical": len(categorical),
            "input_nodes_after_encoding": int(x_train.shape[1]),
            "dropped_groups": list(args.drop_feature_group),
            "dropped_sparse": sparse_dropped,
            "min_feature_coverage": args.min_feature_coverage,
            "target_clip_s": clip,
            "contract_version": "full_context_v2_adjacency_safe_relative",
            "history_laps": history,
        },
        "resources": {
            "feature_build": features_report.as_dict(),
            "training": train_report.as_dict(),
            "inference_batch_ms_per_row": round(batch_ms_per_row, 4),
            "inference_single_row_ms": round(single_row_ms, 3),
            "artifact_bytes": directory_bytes(out),
            "device": "cpu-numpy",
        },
        "data_source": data_source,
        "is_synthetic": is_synthetic_source(data_source),
        "input_file": Path(args.input).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.wandb:
        log_to_wandb(run_name, cfg, metrics, model.training_history)
    (out / "feature_contract.json").write_text(
        json.dumps({"history": history, "numeric": numeric, "categorical": categorical, "dropped_groups": list(args.drop_feature_group), "dropped_sparse": sparse_dropped}, indent=2)
    )
    print(json.dumps({k: metrics[k] for k in ["name", "validation", "test", "rows", "resources"]}, indent=2))


if __name__ == "__main__":
    main()
