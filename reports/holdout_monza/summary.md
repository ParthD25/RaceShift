# RaceShift experiment: holdout_monza

Generated 2026-09-09T00:58:11+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: circuit_holdout · train ≤ 2024 · validation 2025 · test 2025 · holdout Italian Grand Prix
Rows: train 123362 · validation 20404 · test 6133

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.400 | 0.346 | 0.615 | 0.809 | 81.0% | 93.1% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.464 | 0.356 | 0.572 | 0.806 | 78.7% | 93.3% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.459 | 0.345 | 0.552 | 0.735 | 80.2% | 94.9% | — | — | 23.4 | 2575 | 830 | — |
| hist_gradient_boosting | 0.360 | 0.298 | 0.512 | 0.664 | 84.7% | 95.5% | — | — | 50.4 | 2921 | 854 | — |
| FFR-M | 0.394 | 0.347 | 0.538 | 0.694 | 80.0% | 95.4% | 0.838 | 1.105 | 2237.8 | 2761 | 1371 | 2.72 |
| FFR-S | 0.394 | 0.350 | 0.540 | 0.707 | 79.6% | 95.5% | 0.830 | 1.106 | 369.6 | 1996 | 614 | 1.24 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
