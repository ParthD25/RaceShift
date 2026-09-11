# Generalisation gap: f1_2025h2_legacy_ext

Rows: train 420867, validation 9751, test 10519. MAE in seconds on the true next lap. Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).

| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFR-S | 0.583 | 0.385 | 0.325 | -0.259 | 1.268 | 0.503 |
| FFR-M | 0.585 | 0.382 | 0.328 | -0.257 | 1.264 | 0.508 |
| ridge | 0.604 | 0.420 | 0.354 | -0.250 | 1.247 | 0.530 |
| hist_gradient_boosting | 0.535 | 0.357 | 0.303 | -0.232 | 1.252 | 0.493 |

Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.

## FFR-S error by season (training seasons are fit, later seasons are unseen)

| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2000 | train | 15351 | 0.704 | 1.679 | 1.32% |
| 2001 | train | 15018 | 0.657 | 1.136 | 0.98% |
| 2002 | train | 15244 | 0.688 | 1.392 | 1.23% |
| 2003 | train | 13440 | 0.731 | 1.406 | 1.37% |
| 2004 | train | 15267 | 0.777 | 1.482 | 1.00% |
| 2005 | train | 16610 | 0.692 | 1.567 | 0.92% |
| 2006 | train | 17167 | 0.673 | 1.264 | 0.80% |
| 2007 | train | 16567 | 0.663 | 1.669 | 1.38% |
| 2008 | train | 16549 | 0.719 | 1.457 | 1.59% |
| 2009 | train | 14588 | 0.600 | 1.277 | 0.88% |
| 2010 | train | 19037 | 0.810 | 1.846 | 2.10% |
| 2011 | train | 18539 | 0.750 | 1.736 | 1.47% |
| 2012 | train | 20676 | 0.578 | 1.061 | 0.62% |
| 2013 | train | 18129 | 0.500 | 0.828 | 0.26% |
| 2014 | train | 16941 | 0.474 | 0.852 | 0.32% |
| 2015 | train | 16083 | 0.540 | 1.261 | 0.93% |
| 2016 | train | 19484 | 0.600 | 1.448 | 1.09% |
| 2017 | train | 16995 | 0.481 | 1.039 | 0.45% |
| 2018 | train | 14849 | 0.469 | 0.780 | 0.14% |
| 2019 | train | 17943 | 0.456 | 0.748 | 0.16% |
| 2020 | train | 13095 | 0.451 | 1.357 | 0.27% |
| 2021 | train | 18197 | 0.427 | 0.877 | 0.13% |
| 2022 | train | 16325 | 0.418 | 0.988 | 0.24% |
| 2023 | train | 18033 | 0.395 | 0.878 | 0.39% |
| 2024 | train | 20740 | 0.398 | 0.699 | 0.19% |
| 2025 | validation | 9751 | 0.385 | 0.585 | 0.02% |
| 2025 | test | 10519 | 0.325 | 0.503 | 0.00% |
