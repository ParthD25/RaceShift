# RaceShift experiment: domain_shift_2026

Generated 2026-09-08T18:48:23+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: season_forward · train ≤ 2024 · validation 2025 · test 2026
Rows: train 128071 · validation 21242 · test 11483

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.394 | 0.556 | 2.250 | 1.151 | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.457 | 0.628 | 2.501 | 1.210 | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.445 | 0.553 | 2.065 | 1.026 | — | — | 23.5 | 2583 | 868 | — |
| hist_gradient_boosting | 0.352 | 0.544 | 2.145 | 1.018 | — | — | 48.0 | 2970 | 892 | — |
| FFR-M | 0.390 | 0.545 | 2.238 | 0.993 | 0.766 | 1.098 | 2124.6 | 2876 | 1423 | 3.40 |
| FFR-S | 0.395 | 0.553 | 2.267 | 1.004 | 0.765 | 1.113 | 362.1 | 1996 | 637 | 1.91 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds. Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
