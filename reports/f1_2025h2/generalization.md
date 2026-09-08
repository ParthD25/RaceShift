# Generalisation gap: f1_2025h2

Rows: train 128071, validation 10248, test 10994. MAE in seconds on the true next lap. Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).

| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFR-M | 0.498 | 0.430 | 0.352 | -0.146 | 1.504 | 0.595 |
| FFR-S | 0.502 | 0.436 | 0.357 | -0.145 | 1.516 | 0.596 |
| FFR-L | 0.501 | 0.428 | 0.349 | -0.152 | 1.518 | 0.595 |
| ridge | 0.512 | 0.493 | 0.400 | -0.112 | 1.408 | 0.629 |
| hist_gradient_boosting | 0.439 | 0.391 | 0.316 | -0.123 | 1.418 | 0.568 |

Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.

## FFR-M error by season (training seasons are fit, later seasons are unseen)

| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2018 | train | 16374 | 0.510 | 0.960 | 0.41% |
| 2019 | train | 19520 | 0.491 | 0.915 | 0.26% |
| 2020 | train | 14056 | 0.536 | 1.953 | 0.60% |
| 2021 | train | 19264 | 0.540 | 1.852 | 0.71% |
| 2022 | train | 17778 | 0.496 | 1.620 | 0.55% |
| 2023 | train | 19349 | 0.500 | 1.921 | 0.72% |
| 2024 | train | 21730 | 0.436 | 0.970 | 0.30% |
| 2025 | validation | 10248 | 0.430 | 0.708 | 0.14% |
| 2025 | test | 10994 | 0.352 | 0.595 | 0.01% |
