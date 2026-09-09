# RaceShift experiment: domain_shift_2026

Generated 2026-09-09T13:13:54+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: season_forward · train ≤ 2024 · validation 2025 · test 2026
Rows: train 127861 · validation 21242 · test 11445

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.394 | 0.472 | 0.858 | 1.129 | 72.7% | 87.9% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.457 | 0.501 | 0.882 | 1.168 | 69.3% | 87.4% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.451 | 0.457 | 0.778 | 1.003 | 71.5% | 89.9% | — | — | 25.1 | 2575 | 867 | — |
| hist_gradient_boosting | 0.352 | 0.432 | 0.755 | 0.972 | 74.5% | 90.5% | — | — | 51.5 | 2960 | 891 | — |
| FFR-M | 0.389 | 0.433 | 0.768 | 0.957 | 74.4% | 90.7% | 0.769 | 1.088 | 2233.3 | 2814 | 1421 | 3.39 |
| FFR-S | 0.389 | 0.434 | 0.767 | 0.970 | 74.3% | 90.5% | 0.768 | 1.086 | 390.6 | 1995 | 636 | 1.91 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
