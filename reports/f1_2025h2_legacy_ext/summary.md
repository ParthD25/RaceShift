# RaceShift experiment: f1_2025h2_legacy_ext

Generated 2026-09-08T21:59:20+00:00 from `f1_laps_all_tiers.parquet` (data source: fastf1_timing+legacy_timing).
Split: season_round · train ≤ 2024 · validation 2025 · test 2025 · split round 12
Rows: train 429756 · validation 10248 · test 10994

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.434 | 0.357 | 0.645 | 0.822 | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.524 | 0.394 | 0.675 | 0.889 | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.468 | 0.372 | 0.615 | 0.784 | — | — | 117.8 | 9373 | 4861 | — |
| hist_gradient_boosting | 0.394 | 0.314 | 0.569 | 0.680 | — | — | 246.4 | 11273 | 5668 | — |
| FFR-M | 0.420 | 0.347 | 0.592 | 0.742 | 0.856 | 1.191 | 8156.7 | 8941 | 4764 | 3.99 |
| FFR-S | 0.418 | 0.342 | 0.588 | 0.740 | 0.855 | 1.189 | 1640.6 | 7389 | 2571 | 2.21 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds. Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
