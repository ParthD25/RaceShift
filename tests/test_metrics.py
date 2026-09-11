from __future__ import annotations

import numpy as np
import pytest

from raceshift.train.metrics import interval_metrics, regression_metrics


def test_regression_metrics_match_hand_computation():
    truth = np.array([80.0, 81.0, 82.0, 83.0])
    pred = np.array([80.5, 80.5, 82.0, 84.0])
    m = regression_metrics(truth, pred)
    assert m["rows"] == 4
    assert m["mae_s"] == pytest.approx(0.5)
    assert m["rmse_s"] == pytest.approx(np.sqrt((0.25 + 0.25 + 0 + 1) / 4))
    assert m["signed_bias_s"] == pytest.approx(0.25)
    assert m["median_ae_s"] == pytest.approx(0.5)


def test_regression_metrics_drop_non_finite_rows_and_reject_empty():
    m = regression_metrics([80.0, np.nan, 82.0], [80.0, 81.0, np.inf])
    assert m["rows"] == 1
    with pytest.raises(ValueError):
        regression_metrics([np.nan], [1.0])
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])


def test_interval_metrics_coverage_and_width():
    m = interval_metrics([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 3.5, 3.5], [1.5, 2.5, 4.0, 4.5])
    assert m["interval80_coverage"] == pytest.approx(0.75)
    assert m["interval80_width_s"] == pytest.approx(0.875)
