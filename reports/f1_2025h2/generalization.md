# Generalisation gap: f1_2025h2

Rows: train 127861, validation 10248, test 10994. MAE in seconds on the true next lap. Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).

| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFR-S | 0.461 | 0.431 | 0.350 | -0.110 | 1.055 | 0.590 |
| FFR-M | 0.459 | 0.430 | 0.350 | -0.108 | 1.045 | 0.593 |
| FFR-L | 0.461 | 0.429 | 0.348 | -0.113 | 1.049 | 0.595 |
| ridge | 0.475 | 0.503 | 0.402 | -0.072 | 1.009 | 0.628 |
| hist_gradient_boosting | 0.405 | 0.389 | 0.318 | -0.087 | 0.995 | 0.571 |

Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.

## FFR-S error by season (training seasons are fit, later seasons are unseen)

| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2018 | train | 16374 | 0.512 | 0.963 | 0.40% |
| 2019 | train | 19520 | 0.491 | 0.914 | 0.25% |
| 2020 | train | 14018 | 0.479 | 1.491 | 0.40% |
| 2021 | train | 19205 | 0.469 | 1.108 | 0.41% |
| 2022 | train | 17753 | 0.444 | 1.057 | 0.33% |
| 2023 | train | 19298 | 0.428 | 1.130 | 0.46% |
| 2024 | train | 21693 | 0.416 | 0.731 | 0.23% |
| 2025 | validation | 10248 | 0.431 | 0.716 | 0.18% |
| 2025 | test | 10994 | 0.350 | 0.590 | 0.01% |
