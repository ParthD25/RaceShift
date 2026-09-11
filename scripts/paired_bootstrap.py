"""Paired bootstrap confidence intervals for the difference in absolute error between two
models scored on the same lap pairs.

Seed sweeps measure model variance (would a different initialisation change the number?).
They say nothing about sampling variance (would a different set of test laps change it?).
This script answers the second question for the headline comparison "FFR beats the
previous-lap stopwatch by a few thousandths": it resamples test rows (plain bootstrap) and
event x driver clusters (cluster bootstrap, which respects that laps of one driver in one race
are not independent) and reports 95% intervals of the mean paired |error| difference.

    python scripts/paired_bootstrap.py --ffr artifacts/f1_2025h2_ffr-m --baselines artifacts/f1_2025h2_baselines

Both directories must hold a test_predictions.csv with season/event/driver/lap_number,
actual_next_lap_s and predicted_next_lap_s (baselines: one column per model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["season", "event", "session", "driver", "lap_number"]


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "test_predictions.csv")
    missing = [k for k in KEYS if k not in frame.columns]
    if missing:
        raise SystemExit(f"{path}: test_predictions.csv lacks {missing}")
    return frame


def paired_ci(diff: np.ndarray, clusters: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    row = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(draws)])
    ids, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse)
    cluster = []
    for _ in range(draws):
        pick = rng.integers(0, len(ids), len(ids))
        cluster.append(sums[pick].sum() / counts[pick].sum())
    cluster = np.array(cluster)
    return {
        "mean_diff_s": float(diff.mean()),
        "row_ci_low": float(np.quantile(row, 0.025)),
        "row_ci_high": float(np.quantile(row, 0.975)),
        "cluster_ci_low": float(np.quantile(cluster, 0.025)),
        "cluster_ci_high": float(np.quantile(cluster, 0.975)),
        "clusters": int(len(ids)),
        "share_ffr_better": float((diff < 0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ffr", required=True, help="FFR artifact directory")
    p.add_argument("--baselines", required=True, help="baselines report directory")
    p.add_argument("--draws", type=_positive_int, default=2000, help="bootstrap resamples (positive)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ffr = _load(Path(args.ffr))
    base = _load(Path(args.baselines))
    merged = ffr.merge(base, on=KEYS, suffixes=("", "_base"), validate="one_to_one")
    if merged.empty:
        raise SystemExit("no common lap pairs between the two prediction files")
    actual = merged["actual_next_lap_s"].to_numpy(float)
    ffr_err = np.abs(merged["predicted_next_lap_s"].to_numpy(float) - actual)
    clusters = (merged["season"].astype(str) + "|" + merged["event"].astype(str) + "|" + merged["driver"].astype(str)).to_numpy()
    print(f"{len(merged)} shared lap pairs, {Path(args.ffr).name} vs {Path(args.baselines).name}")
    print(f"{'baseline':24s} {'mean diff (s)':>14s} {'row 95% CI':>22s} {'cluster 95% CI':>22s} {'FFR better':>11s}")
    for column in [c for c in merged.columns if c.startswith("pred_") or c in ("previous_lap", "rolling_median_5", "ridge", "hist_gradient_boosting")]:
        other = np.abs(merged[column].to_numpy(float) - actual)
        ok = np.isfinite(ffr_err) & np.isfinite(other)
        r = paired_ci((ffr_err - other)[ok], clusters[ok], args.draws, args.seed)
        name = column.replace("pred_", "")
        print(
            f"{name:24s} {r['mean_diff_s']:+14.4f} "
            f"[{r['row_ci_low']:+.4f}, {r['row_ci_high']:+.4f}] "
            f"[{r['cluster_ci_low']:+.4f}, {r['cluster_ci_high']:+.4f}] {r['share_ffr_better']:10.1%}"
        )
    print("negative = FFR error lower. A cluster interval that includes 0 means the edge is not distinguishable from sampling noise.")


if __name__ == "__main__":
    main()
