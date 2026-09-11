# RaceShift experiment: domain_shift_2026

Generated 2026-09-11T01:52:23+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: season_forward · train ≤ 2024 · validation 2025 · test 2026
Rows: train 119182 · validation 20270 · test 10778

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.361 | 0.453 | 0.751 | 1.104 | 73.3% | 88.3% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.405 | 0.475 | 0.751 | 1.133 | 69.9% | 87.8% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.398 | 0.426 | 0.680 | 0.973 | 73.8% | 90.4% | — | — | 2.3 | 1906 | 368 | — |
| hist_gradient_boosting | 0.330 | 0.409 | 0.667 | 0.938 | 75.6% | 90.9% | — | — | 36.3 | 2293 | 471 | — |
| FFR-M | 0.363 | 0.413 | 0.665 | 0.935 | 75.2% | 91.1% | 0.767 | 1.048 | 2572.4 | 2693 | 1325 | 3.31 |
| FFR-S | 0.359 | 0.412 | 0.662 | 0.935 | 75.5% | 91.0% | 0.767 | 1.040 | 494.8 | 1966 | 593 | 1.83 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
