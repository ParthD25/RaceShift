# RaceShift experiment: f1_2025h2_legacy_ext

Generated 2026-09-09T17:52:55+00:00 from `f1_laps_all_tiers.parquet` (data source: fastf1_timing+legacy_timing).
Split: season_round · train ≤ 2024 · validation 2025 · test 2025 · split round 12
Rows: train 429546 · validation 10248 · test 10994

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.434 | 0.357 | 0.645 | 0.822 | 80.6% | 92.7% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.524 | 0.394 | 0.675 | 0.889 | 76.9% | 91.9% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.463 | 0.368 | 0.614 | 0.781 | 78.7% | 93.7% | — | — | 128.4 | 9240 | 4859 | — |
| hist_gradient_boosting | 0.395 | 0.315 | 0.570 | 0.679 | 83.6% | 94.7% | — | — | 279.6 | 11138 | 5667 | — |
| FFR-M | 0.425 | 0.351 | 0.595 | 0.740 | 80.2% | 94.4% | 0.856 | 1.201 | 7958.8 | 8953 | 4762 | 3.99 |
| FFR-S | 0.425 | 0.348 | 0.592 | 0.739 | 80.7% | 94.4% | 0.858 | 1.202 | 1775.8 | 7403 | 2570 | 2.21 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
