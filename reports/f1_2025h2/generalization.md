# Generalisation gap: f1_2025h2

Rows: train 119182, validation 9751, test 10519. MAE in seconds on the true next lap. Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).

| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFR-S | 0.421 | 0.393 | 0.328 | -0.093 | 0.904 | 0.501 |
| FFR-M | 0.423 | 0.397 | 0.331 | -0.092 | 0.899 | 0.505 |
| FFR-L | 0.425 | 0.390 | 0.327 | -0.098 | 0.902 | 0.505 |
| ridge | 0.434 | 0.435 | 0.365 | -0.069 | 0.859 | 0.535 |
| hist_gradient_boosting | 0.378 | 0.359 | 0.303 | -0.075 | 0.864 | 0.494 |

Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.

## FFR-S error by season (training seasons are fit, later seasons are unseen)

| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2018 | train | 14849 | 0.463 | 0.776 | 0.14% |
| 2019 | train | 17943 | 0.451 | 0.736 | 0.14% |
| 2020 | train | 13095 | 0.442 | 1.374 | 0.29% |
| 2021 | train | 18197 | 0.419 | 0.868 | 0.13% |
| 2022 | train | 16325 | 0.413 | 0.998 | 0.23% |
| 2023 | train | 18033 | 0.388 | 0.888 | 0.38% |
| 2024 | train | 20740 | 0.390 | 0.685 | 0.18% |
| 2025 | validation | 9751 | 0.393 | 0.584 | 0.02% |
| 2025 | test | 10519 | 0.328 | 0.501 | 0.00% |
