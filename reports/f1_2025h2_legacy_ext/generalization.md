# Generalisation gap: f1_2025h2_legacy_ext

Rows: train 429546, validation 10248, test 10994. MAE in seconds on the true next lap. Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).

| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFR-S | 0.591 | 0.425 | 0.348 | -0.243 | 1.293 | 0.592 |
| FFR-M | 0.592 | 0.425 | 0.351 | -0.241 | 1.289 | 0.595 |
| ridge | 0.614 | 0.463 | 0.379 | -0.235 | 1.277 | 0.619 |
| hist_gradient_boosting | 0.541 | 0.395 | 0.315 | -0.226 | 1.276 | 0.570 |

Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.

## FFR-S error by season (training seasons are fit, later seasons are unseen)

| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2000 | train | 15351 | 0.704 | 1.681 | 1.33% |
| 2001 | train | 15018 | 0.658 | 1.137 | 0.98% |
| 2002 | train | 15244 | 0.687 | 1.391 | 1.23% |
| 2003 | train | 13440 | 0.730 | 1.404 | 1.34% |
| 2004 | train | 15267 | 0.776 | 1.477 | 0.99% |
| 2005 | train | 16610 | 0.694 | 1.564 | 0.91% |
| 2006 | train | 17167 | 0.675 | 1.263 | 0.79% |
| 2007 | train | 16567 | 0.665 | 1.667 | 1.38% |
| 2008 | train | 16549 | 0.721 | 1.458 | 1.55% |
| 2009 | train | 14588 | 0.604 | 1.278 | 0.88% |
| 2010 | train | 19037 | 0.808 | 1.842 | 2.08% |
| 2011 | train | 18539 | 0.754 | 1.735 | 1.46% |
| 2012 | train | 20676 | 0.581 | 1.064 | 0.63% |
| 2013 | train | 18129 | 0.502 | 0.829 | 0.26% |
| 2014 | train | 16941 | 0.473 | 0.851 | 0.33% |
| 2015 | train | 16083 | 0.540 | 1.259 | 0.93% |
| 2016 | train | 19484 | 0.599 | 1.445 | 1.06% |
| 2017 | train | 16995 | 0.482 | 1.039 | 0.45% |
| 2018 | train | 16374 | 0.510 | 0.958 | 0.40% |
| 2019 | train | 19520 | 0.492 | 0.920 | 0.27% |
| 2020 | train | 14018 | 0.483 | 1.467 | 0.39% |
| 2021 | train | 19205 | 0.474 | 1.100 | 0.40% |
| 2022 | train | 17753 | 0.448 | 1.035 | 0.34% |
| 2023 | train | 19298 | 0.430 | 1.114 | 0.45% |
| 2024 | train | 21693 | 0.419 | 0.735 | 0.23% |
| 2025 | validation | 10248 | 0.425 | 0.707 | 0.16% |
| 2025 | test | 10994 | 0.348 | 0.592 | 0.02% |
