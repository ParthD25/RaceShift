from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float | int]:
    """Point-forecast metrics in seconds, shared by every RaceShift model and baseline.

    Rows with a non-finite truth or prediction are dropped so a single bad row cannot
    turn the whole report into NaN.
    """
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    keep = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[keep], pred[keep]
    if truth.size == 0:
        raise ValueError("No finite rows available for metrics")
    absolute = np.abs(truth - pred)
    return {
        "mae_s": float(np.mean(absolute)),
        "rmse_s": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "median_ae_s": float(np.median(absolute)),
        "p90_ae_s": float(np.quantile(absolute, 0.90)),
        "signed_bias_s": float(np.mean(pred - truth)),
        "rows": int(truth.size),
    }


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    """Coverage and mean width of a prediction interval."""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    keep = np.isfinite(truth) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return {"interval80_coverage": float("nan"), "interval80_width_s": float("nan")}
    truth, lo, hi = truth[keep], lo[keep], hi[keep]
    return {
        "interval80_coverage": float(np.mean((truth >= lo) & (truth <= hi))),
        "interval80_width_s": float(np.mean(hi - lo)),
    }
