# RaceShift experiment: holdout_monza

Generated 2026-09-11T00:44:56+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: circuit_holdout · train ≤ 2024 · validation 2025 · test 2025 · holdout Italian Grand Prix
Rows: train 114924 · validation 19463 · test 5857

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.367 | 0.332 | 0.543 | 0.781 | 81.7% | 93.6% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.412 | 0.341 | 0.524 | 0.762 | 79.7% | 94.0% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.405 | 0.321 | 0.493 | 0.685 | 82.5% | 95.3% | — | — | 0.6 | 1812 | 352 | — |
| hist_gradient_boosting | 0.334 | 0.289 | 0.465 | 0.638 | 85.2% | 95.9% | — | — | 34.3 | 2190 | 452 | — |
| FFR-M | 0.367 | 0.335 | 0.487 | 0.669 | 81.3% | 96.0% | 0.836 | 1.065 | 2473.4 | 2620 | 1278 | 2.69 |
| FFR-S | 0.362 | 0.333 | 0.485 | 0.671 | 81.3% | 96.0% | 0.826 | 1.049 | 455.6 | 1913 | 572 | 1.20 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
