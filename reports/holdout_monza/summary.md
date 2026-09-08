# RaceShift experiment: holdout_monza

Generated 2026-09-08T17:57:22+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: circuit_holdout · train ≤ 2024 · validation 2025 · test 2025 · holdout Italian Grand Prix
Rows: train 123559 · validation 20404 · test 6166

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.400 | 0.509 | 2.793 | 0.826 | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.464 | 0.600 | 3.103 | 0.848 | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.452 | 0.576 | 2.550 | 0.790 | — | — | 25.1 | 2494 | 831 | — |
| hist_gradient_boosting | 0.362 | 0.498 | 2.619 | 0.692 | — | — | 49.4 | 2915 | 855 | — |
| FFR-M | 0.398 | 0.558 | 2.783 | 0.726 | 0.837 | 1.135 | 2051.9 | 2767 | 1373 | 2.72 |
| FFR-S | 0.402 | 0.569 | 2.798 | 0.744 | 0.826 | 1.134 | 355.8 | 2087 | 615 | 1.24 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds. Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
