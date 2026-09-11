#!/usr/bin/env python
"""Fine-tune a saved RaceShift Forward-Forward model on new laps, forward-forward only.

The base artifact's layers are trained further with their own local objectives, one layer
at a time, from the saved weights; the ridge readout is then re-solved in closed form.
There is no global backward pass at any point (``CLAUDE.md``, training policy). The base
preprocessor and feature contract are reused unchanged.

The input table must hold the new rows *and* the history they need (historical priors are
computed from earlier events in the same table), e.g. the full 2018-2026 FastF1 table, or
that table concatenated with OpenF1 rows.

Example, adapting the 2026 domain-shift model to the first five races of 2026 and scoring
it on rounds 8-13 (rounds 6-7 calibrate the interval):

    python scripts/finetune_ffr.py --base artifacts/domain_shift_2026_ffr-m \
        --input data/processed/f1_laps_fastf1.parquet --season 2026 \
        --train-rounds 1-5 --val-rounds 6-7 --test-rounds 8- \
        --epochs-per-layer 20 --output artifacts/ft2026_ffr-m

``--replay-rows N`` mixes N rows sampled from the base model's own training seasons into
the readout refit (and the layer updates when ``--replay-in-layers`` is set) to limit
forgetting; ``--epochs-per-layer 0`` refits only the readout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift import __version__  # noqa: E402
from raceshift.data.provenance import infer_data_source, is_synthetic_source  # noqa: E402
from raceshift.features.full_context import LAP_VALIDITY_VERSION  # noqa: E402
from raceshift.models.artifact import RaceShiftArtifact  # noqa: E402
from raceshift.train.finetune import (  # noqa: E402
    build_table,
    copy_preprocessor,
    describe,
    encode,
    evaluate,
    naive_metrics,
    parse_rounds,
    predict_frame,
    select_rows,
    write_contract,
)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Forward-forward fine-tuning of a saved RaceShift model.")
    p.add_argument("--base", required=True, help="Artifact directory to start from")
    p.add_argument("--input", required=True, help="Lap table holding the new rows and their history")
    p.add_argument("--output", required=True)
    p.add_argument("--season", type=int, help="Season of the new rows (round-based splits within one season)")
    p.add_argument("--train-rounds", help="Rounds to fine-tune on, e.g. 1-5")
    p.add_argument("--val-rounds", help="Rounds that recalibrate the 80% interval, e.g. 6-7")
    p.add_argument("--test-rounds", help="Rounds to score, e.g. 8- (never seen in training)")
    p.add_argument("--train-seasons", help="Alternative to rounds: whole seasons to fine-tune on, e.g. 2023-2025 (with --sessions S: every sprint of those seasons)")
    p.add_argument("--val-seasons", help="Whole seasons that recalibrate the interval")
    p.add_argument("--test-seasons", help="Whole seasons to score, e.g. 2026")
    p.add_argument("--sessions", nargs="+", default=["R"], help="Session codes to use (R, S)")
    p.add_argument("--epochs-per-layer", type=int, default=20)
    p.add_argument("--learning-rate", type=float, help="Override the base model's Adam learning rate for the local updates")
    p.add_argument("--replay-rows", type=int, default=0, help="Rows sampled from the base model's training seasons to include in the readout refit")
    p.add_argument("--replay-in-layers", action="store_true", help="Also include the replay rows in the local layer updates")
    p.add_argument("--replay-seed", type=int, default=7)
    p.add_argument("--target-clip", type=float, default=6.0)
    p.add_argument("--name")
    p.add_argument("--data-source", help="Provenance label stored in metrics.json. Inferred when omitted.")
    p.add_argument("--table-cache", help="Parquet path caching the built feature table for this input (a sweep builds features once)")
    args = p.parse_args()

    base_dir = Path(args.base)
    base = RaceShiftArtifact(base_dir)
    base_split = base.metrics.get("split", {})
    by_season = args.train_seasons is not None
    if by_season:
        if not args.test_seasons:
            raise SystemExit("--train-seasons needs --test-seasons")
        train_sel, val_sel, test_sel = parse_rounds(args.train_seasons), parse_rounds(args.val_seasons), parse_rounds(args.test_seasons)
        if val_sel and not (train_sel[1] < val_sel[0] <= val_sel[1] < test_sel[0]):
            raise SystemExit("Seasons must be ordered: train < val < test")
        if train_sel[1] >= test_sel[0]:
            raise SystemExit("Test seasons must come after the fine-tuning seasons")
        pick = lambda sel: select_rows(table, sel, None, args.sessions)  # noqa: E731
    else:
        if args.season is None or not args.train_rounds or not args.test_rounds:
            raise SystemExit("Give --season with --train-rounds and --test-rounds, or --train-seasons with --test-seasons")
        train_sel, val_sel, test_sel = parse_rounds(args.train_rounds), parse_rounds(args.val_rounds), parse_rounds(args.test_rounds)
        if val_sel and not (train_sel[1] < val_sel[0] <= val_sel[1] < test_sel[0]):
            raise SystemExit("Rounds must be ordered: train < val < test")
        if train_sel[1] >= test_sel[0]:
            raise SystemExit("Test rounds must come after the fine-tuning rounds")
        pick = lambda sel: select_rows(table, args.season, sel, args.sessions)  # noqa: E731

    raw = load_table(Path(args.input))
    data_source = args.data_source or infer_data_source(raw)
    table = build_table(base, raw, args.table_cache)
    train = pick(train_sel)
    val = pick(val_sel) if val_sel else pd.DataFrame(columns=table.columns)
    test = pick(test_sel)
    if train.empty or test.empty:
        raise SystemExit(f"Empty split: train {len(train)} rows, test {len(test)} rows")

    replay = pd.DataFrame(columns=table.columns)
    if args.replay_rows > 0:
        train_end = int(base_split.get("train_end", (args.season or train_sel[0]) - 1))
        pool = table[pd.to_numeric(table["season"], errors="coerce") <= train_end]
        replay = pool.sample(n=min(args.replay_rows, len(pool)), random_state=args.replay_seed) if len(pool) else replay

    x_train, y_train = encode(base, train, args.target_clip)
    x_test, y_test = encode(base, test)
    x_val, y_val = encode(base, val, args.target_clip) if len(val) else (None, None)
    x_replay, y_replay = encode(base, replay, args.target_clip) if len(replay) else (np.zeros((0, x_train.shape[1]), np.float32), np.zeros(0, np.float32))

    zero_shot = predict_frame(base, test, x_test)
    zero_shot_val = predict_frame(base, val, x_val) if len(val) else None

    layer_x, layer_y = (np.concatenate([x_train, x_replay]), np.concatenate([y_train, y_replay])) if (args.replay_in_layers and len(x_replay)) else (x_train, y_train)
    readout = (np.concatenate([x_train, x_replay]), np.concatenate([y_train, y_replay])) if len(x_replay) else None
    t0 = time.perf_counter()
    base.model.continue_fit(
        layer_x, layer_y,
        epochs_per_layer=args.epochs_per_layer,
        readout=readout,
        validation=(x_val, y_val) if x_val is not None else None,
        learning_rate=args.learning_rate,
    )
    train_seconds = time.perf_counter() - t0

    tuned = predict_frame(base, test, x_test)
    tuned_val = predict_frame(base, val, x_val) if len(val) else None
    tuned_train = predict_frame(base, train, x_train)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base.model.save(out)
    copy_preprocessor(base_dir, out)
    write_contract(base, out, {"fine_tuned_from": base_dir.name})
    tuned.to_csv(out / "test_predictions.csv", index=False)
    zero_shot.to_csv(out / "base_test_predictions.csv", index=False)

    name = args.name or f"{base.metrics.get('name', base_dir.name)} fine-tuned"
    metrics = {
        "model": "RaceShiftFFR",
        "name": name,
        "raceshift_version": __version__,
        "training_policy": "forward-forward-local-updates-no-global-backprop",
        "global_backprop": False,
        "fine_tuned_from": base_dir.name,
        "fine_tune": {
            "method": "continue_fit: local layer-wise updates from the saved weights, then a closed-form ridge readout refit",
            "epochs_per_layer": args.epochs_per_layer,
            "learning_rate": args.learning_rate if args.learning_rate is not None else base.model.config.learning_rate,
            "replay_rows": int(len(replay)),
            "replay_in_layers": bool(args.replay_in_layers and len(replay)),
            "target_clip_s": args.target_clip,
            "sessions": args.sessions,
            "train_seconds": round(train_seconds, 2),
        },
        "config_file": base.metrics.get("config_file"),
        "hyperparameters": json.loads(json.dumps(base.model.config.__dict__)),
        "architecture": base.model.architecture_summary(),
        "train": evaluate(tuned_train),
        "validation": evaluate(tuned_val) if tuned_val is not None else {},
        "test": evaluate(tuned),
        "base_zero_shot": {
            "test": evaluate(zero_shot),
            "validation": evaluate(zero_shot_val) if zero_shot_val is not None else {},
        },
        "naive": naive_metrics(tuned),
        "test_by_event": {str(k): evaluate(g) for k, g in tuned.groupby("event") if len(g) >= 20},
        "split": {
            "mode": "fine_tune_seasons" if by_season else "fine_tune_rounds",
            "season": args.season,
            "train_rounds": None if by_season else list(train_sel),
            "validation_rounds": None if (by_season or not val_sel) else list(val_sel),
            "test_rounds": None if by_season else list(test_sel),
            "train_seasons": list(train_sel) if by_season else None,
            "validation_seasons": list(val_sel) if (by_season and val_sel) else None,
            "test_seasons": list(test_sel) if by_season else None,
            "base_split": base_split,
            "train_rows": describe(train),
            "validation_rows": describe(val),
            "test_rows": describe(test),
            "replay_rows": describe(replay),
        },
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "features": dict(base.metrics.get("features", {}), input_nodes_after_encoding=int(x_train.shape[1])),
        "data_source": data_source,
        "is_synthetic": is_synthetic_source(data_source),
        "input_file": Path(args.input).name,
        "lap_validity_version": LAP_VALIDITY_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({
        "name": name,
        "rows": metrics["rows"],
        "base_zero_shot_test_mae_s": round(metrics["base_zero_shot"]["test"]["mae_s"], 4),
        "fine_tuned_test_mae_s": round(metrics["test"]["mae_s"], 4),
        "previous_lap_test_mae_s": round(metrics["naive"]["previous_lap"]["mae_s"], 4),
        "rolling5_test_mae_s": round(metrics["naive"]["rolling_median_5"]["mae_s"], 4),
        "train_seconds": metrics["fine_tune"]["train_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
