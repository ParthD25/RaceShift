# RaceShift experiment: f1_2025h2

Generated 2026-09-11T09:17:37+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: season_round · train ≤ 2024 · validation 2025 · test 2025 · split round 12
Rows: train 119182 · validation 9751 · test 10519

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.388 | 0.336 | 0.551 | 0.768 | 81.8% | 93.5% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.448 | 0.366 | 0.576 | 0.825 | 78.6% | 93.0% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.435 | 0.357 | 0.531 | 0.744 | 78.4% | 94.3% | — | — | 0.4 | 1859 | 340 | — |
| hist_gradient_boosting | 0.359 | 0.303 | 0.493 | 0.656 | 84.7% | 95.1% | — | — | 42.6 | 2214 | 439 | — |
| FFR-M | 0.397 | 0.331 | 0.505 | 0.693 | 82.1% | 95.0% | 0.865 | 1.169 | 2572.7 | 2654 | 1325 | 3.31 |
| FFR-S | 0.393 | 0.328 | 0.501 | 0.694 | 82.2% | 95.1% | 0.859 | 1.145 | 494.6 | 1933 | 593 | 1.82 |
| FFR-L | 0.390 | 0.327 | 0.505 | 0.691 | 82.4% | 94.8% | 0.862 | 1.141 | 11276.4 | 3928 | 2591 | 8.18 |
| FFR-M-groups-coarse | 0.385 | 0.332 | 0.510 | 0.704 | 81.5% | 94.7% | 0.852 | 1.120 | 2415.2 | 2670 | 1321 | 3.27 |
| FFR-M-groups-fine | 0.396 | 0.332 | 0.505 | 0.692 | 81.7% | 94.9% | 0.860 | 1.162 | 2603.6 | 2670 | 1332 | 3.31 |
| FFR-M minus historical_numeric | 0.402 | 0.330 | 0.504 | 0.698 | 81.9% | 95.1% | 0.869 | 1.185 | 2540.1 | 2665 | 1325 | 3.27 |
| FFR-M minus temporal_numeric | 0.416 | 0.351 | 0.534 | 0.747 | 79.7% | 94.3% | 0.855 | 1.224 | 2331.8 | 2660 | 1324 | 3.16 |
| FFR-M minus static_categorical | 0.380 | 0.329 | 0.504 | 0.699 | 82.1% | 95.0% | 0.851 | 1.109 | 2685.5 | 2574 | 1324 | 2.89 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
