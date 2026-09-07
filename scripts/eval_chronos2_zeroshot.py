#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raceshift.data.splits import season_forward_split
from raceshift.foundation.chronos_data import example_to_chronos_input, make_walk_forward_examples
from raceshift.train.metrics import regression_metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate stock Chronos-2 on unseen RaceShift laps.")
    p.add_argument("--data", default=str(ROOT / "data" / "processed" / "next_lap.parquet"))
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--output", default=str(ROOT / "artifacts" / "chronos2_zeroshot"))
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    _, _, test = season_forward_split(df, args.test_year - 2, args.test_year - 1, args.test_year)
    examples = make_walk_forward_examples(test, context_length=args.context, stride=args.stride)

    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    pred, truth, lo, hi = [], [], [], []
    for s in range(0, len(examples), 128):
        b = examples[s:s+128]
        q, m = pipe.predict_quantiles([example_to_chronos_input(x) for x in b], prediction_length=1, quantile_levels=[0.1,0.5,0.9], batch_size=128, context_length=args.context)
        for ex, qi, mi in zip(b, q, m):
            arr = qi.detach().cpu().numpy().reshape(-1,3)[0]
            pred.append(float(mi.detach().cpu().numpy().reshape(-1)[0])); truth.append(ex.actual_next); lo.append(arr[0]); hi.append(arr[2])
    metrics = regression_metrics(np.asarray(truth), np.asarray(pred))
    metrics["interval80_coverage"] = float(np.mean((np.asarray(truth)>=np.asarray(lo))&(np.asarray(truth)<=np.asarray(hi))))
    metrics["n_walk_forward_examples"] = len(truth)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__ == "__main__": main()
