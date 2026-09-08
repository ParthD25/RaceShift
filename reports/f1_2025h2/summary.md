# RaceShift experiment: f1_2025h2

Generated 2026-09-08T17:08:19+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: season_round · train ≤ 2024 · validation 2025 · test 2025 · split round 12
Rows: train 128071 · validation 10248 · test 10994

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.434 | 0.357 | 0.645 | 0.822 | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.524 | 0.394 | 0.675 | 0.889 | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.493 | 0.393 | 0.624 | 0.811 | — | — | 21.4 | 2478 | 804 | — |
| hist_gradient_boosting | 0.391 | 0.317 | 0.570 | 0.680 | — | — | 43.4 | 2888 | 830 | — |
| FFR-M | 0.430 | 0.352 | 0.595 | 0.740 | 0.857 | 1.213 | 2064.1 | 2973 | 1423 | 3.37 |
| FFR-S | 0.436 | 0.357 | 0.596 | 0.747 | 0.856 | 1.221 | 375.2 | 1990 | 637 | 1.88 |
| FFR-L | 0.428 | 0.349 | 0.595 | 0.735 | 0.858 | 1.209 | 9804.7 | 4324 | 2783 | 8.24 |
| FFR-M-groups-coarse | 0.426 | 0.352 | 0.598 | 0.749 | 0.852 | 1.201 | 2077.7 | 2972 | 1419 | 3.33 |
| FFR-M-groups-fine | 0.439 | 0.353 | 0.594 | 0.747 | 0.865 | 1.258 | 2154.2 | 2880 | 1431 | 3.37 |
| FFR-M minus historical_numeric | 0.433 | 0.353 | 0.596 | 0.748 | 0.855 | 1.209 | 2135.9 | 2807 | 1423 | 3.33 |
| FFR-M minus temporal_numeric | 0.463 | 0.375 | 0.623 | 0.798 | 0.858 | 1.325 | 1817.6 | 2867 | 1423 | 3.22 |
| FFR-M minus static_categorical | 0.423 | 0.353 | 0.596 | 0.743 | 0.851 | 1.187 | 2081.3 | 2715 | 1422 | 2.94 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds. Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
