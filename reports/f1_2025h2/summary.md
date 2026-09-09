# RaceShift experiment: f1_2025h2

Generated 2026-09-09T14:32:44+00:00 from `f1_laps_fastf1.parquet` (data source: fastf1_timing).
Split: season_round · train ≤ 2024 · validation 2025 · test 2025 · split round 12
Rows: train 127861 · validation 10248 · test 10994

| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | Laps within 0.5 s | Laps within 1 s | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.434 | 0.357 | 0.645 | 0.822 | 80.6% | 92.7% | — | — | 0.0 | 0 | 0 | — |
| rolling_median_5 | 0.524 | 0.394 | 0.675 | 0.889 | 76.9% | 91.9% | — | — | 0.0 | 0 | 0 | — |
| ridge | 0.503 | 0.392 | 0.622 | 0.801 | 74.8% | 93.7% | — | — | 22.7 | 2478 | 803 | — |
| hist_gradient_boosting | 0.389 | 0.316 | 0.569 | 0.684 | 83.8% | 94.8% | — | — | 49.9 | 2887 | 830 | — |
| FFR-M | 0.430 | 0.350 | 0.593 | 0.729 | 80.6% | 94.4% | 0.862 | 1.211 | 2234.1 | 2868 | 1421 | 3.37 |
| FFR-S | 0.431 | 0.350 | 0.590 | 0.729 | 80.5% | 94.4% | 0.858 | 1.197 | 390.3 | 2024 | 636 | 1.88 |
| FFR-L | 0.429 | 0.348 | 0.595 | 0.731 | 80.5% | 94.2% | 0.861 | 1.202 | 9985.4 | 4154 | 2778 | 8.24 |
| FFR-M-groups-coarse | 0.424 | 0.350 | 0.598 | 0.740 | 80.4% | 94.2% | 0.854 | 1.187 | 2088.2 | 2811 | 1417 | 3.33 |
| FFR-M-groups-fine | 0.436 | 0.351 | 0.593 | 0.740 | 80.1% | 94.3% | 0.863 | 1.235 | 2190.8 | 2980 | 1429 | 3.37 |
| FFR-M minus historical_numeric | 0.436 | 0.351 | 0.595 | 0.739 | 80.2% | 94.3% | 0.860 | 1.219 | 2208.3 | 2816 | 1421 | 3.33 |
| FFR-M minus temporal_numeric | 0.463 | 0.375 | 0.623 | 0.799 | 77.8% | 93.6% | 0.860 | 1.323 | 1968.1 | 2814 | 1420 | 3.22 |
| FFR-M minus static_categorical | 0.423 | 0.352 | 0.595 | 0.736 | 80.1% | 94.4% | 0.852 | 1.175 | 2218.5 | 2714 | 1420 | 2.94 |

MAE, RMSE and p90 are absolute errors on the true next lap time in seconds (lower is better); the within-tolerance columns are the share of test laps predicted within 0.5 s and 1 s of the true lap (higher is better). Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.
