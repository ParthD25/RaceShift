# RaceShift experiment: f1_2025h2_legacy_ext

Generated 2026-09-11T15:32:47+00:00 from `f1_laps_all_tiers.parquet` (data source: fastf1_timing+legacy_timing).
Split: season_round · train ≤ 2024 · validation 2025 · test 2025 · split round 12
Rows: train 420867 · validation 9751 · test 10519

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.388 | 0.336 | 0.551 | 0.768 | 81.8% | 93.5% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.448 | 0.366 | 0.576 | 0.825 | 78.6% | 93.0% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.420 | 0.342 | 0.523 | 0.732 | 81.0% | 94.4% | — | — | 3.5 | 6957 | 2258 | — |
| hist_gradient_boosting | 0.357 | 0.303 | 0.493 | 0.647 | 84.6% | 95.0% | — | — | 190.2 | 9987 | 3321 | — |
| FFR-M | 0.382 | 0.328 | 0.508 | 0.706 | 82.1% | 94.8% | 0.854 | 1.122 | 10173.5 | 8771 | 4666 | 3.94 |
| FFR-S | 0.385 | 0.325 | 0.503 | 0.695 | 82.4% | 94.9% | 0.858 | 1.136 | 2373.4 | 7291 | 2518 | 2.15 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
