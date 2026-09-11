#!/usr/bin/env python
"""Score a saved RaceShift artifact on any lap table, or a season/round/session subset of it.

Used for cross-provider checks (the same model on FastF1 rows and on OpenF1 rows of the
same races) and for zero-shot evaluation on data the model never saw. Writes
``predictions.csv`` and ``metrics.json`` into ``--output``.

    python scripts/score_artifact.py --artifact artifacts/f1_2025h2_ffr-m \
        --input data/imports/f1_2025_openf1.parquet --season 2025 --rounds 13- \
        --output reports/cross_provider/ffr-m_openf1_2025
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.data.provenance import infer_data_source  # noqa: E402
from raceshift.features.full_context import LAP_VALIDITY_VERSION  # noqa: E402
from raceshift.models.artifact import RaceShiftArtifact  # noqa: E402
from raceshift.train.finetune import build_table, describe, encode, evaluate, naive_metrics, parse_rounds, predict_frame, select_rows  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Score a RaceShift artifact on a lap table subset.")
    p.add_argument("--artifact", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--season", type=int)
    p.add_argument("--rounds", help="e.g. 13- or 1-5")
    p.add_argument("--sessions", nargs="+", help="Session codes, e.g. R S")
    p.add_argument("--table-cache", help="Parquet path caching the built feature table for this input")
    args = p.parse_args()

    artifact = RaceShiftArtifact(args.artifact)
    raw = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    table = build_table(artifact, raw, args.table_cache)
    rows = select_rows(table, args.season, parse_rounds(args.rounds), args.sessions)
    if rows.empty:
        raise SystemExit("No rows selected")
    x, _ = encode(artifact, rows)
    predictions = predict_frame(artifact, rows, x)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    metrics = {
        "artifact": Path(args.artifact).name,
        "model_name": artifact.metrics.get("name"),
        "input_file": Path(args.input).name,
        "data_source": infer_data_source(raw),
        "selection": {"season": args.season, "rounds": args.rounds, "sessions": args.sessions},
        "rows": describe(rows),
        "model": evaluate(predictions),
        "naive": naive_metrics(predictions),
        "by_event": {str(k): evaluate(g) for k, g in predictions.groupby("event") if len(g) >= 20},
        "lap_validity_version": LAP_VALIDITY_VERSION,
        "artifact_lap_validity_version": artifact.contract.get("lap_validity_version"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"rows": len(rows), "model_mae_s": round(metrics["model"]["mae_s"], 4), "previous_lap_mae_s": round(metrics["naive"]["previous_lap"]["mae_s"], 4), "rolling5_mae_s": round(metrics["naive"]["rolling_median_5"]["mae_s"], 4)}, indent=2))


if __name__ == "__main__":
    main()
