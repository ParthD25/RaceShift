# Generalisation gap: f1_2025h2_legacy_ext

Rows: train 429756, validation 10248, test 10994. MAE in seconds on the true next lap. Gap = test MAE minus train MAE; a large positive gap means the model fits its training laps much better than unseen laps (memorisation).

| Model | Train MAE | Validation MAE | Test MAE | Gap (test − train) | Train RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFR-S | 0.603 | 0.418 | 0.342 | -0.261 | 1.425 | 0.588 |
| FFR-M | 0.604 | 0.420 | 0.347 | -0.256 | 1.420 | 0.592 |
| ridge | 0.627 | 0.468 | 0.383 | -0.244 | 1.408 | 0.621 |
| hist_gradient_boosting | 0.551 | 0.394 | 0.315 | -0.235 | 1.390 | 0.572 |

Baselines here are refit on the training split only (the experiment tables refit learned baselines on train+validation before scoring test), so their test MAE can differ slightly from `summary.md`.

## FFR-S error by season (training seasons are fit, later seasons are unseen)

| Season | Split | Laps | MAE (s) | RMSE (s) | Share of laps with error > 5 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2000 | train | 15351 | 0.703 | 1.679 | 1.33% |
| 2001 | train | 15018 | 0.657 | 1.137 | 0.99% |
| 2002 | train | 15244 | 0.684 | 1.391 | 1.25% |
| 2003 | train | 13440 | 0.729 | 1.404 | 1.34% |
| 2004 | train | 15267 | 0.774 | 1.476 | 0.96% |
| 2005 | train | 16610 | 0.695 | 1.563 | 0.90% |
| 2006 | train | 17167 | 0.676 | 1.262 | 0.80% |
| 2007 | train | 16567 | 0.668 | 1.666 | 1.37% |
| 2008 | train | 16549 | 0.723 | 1.458 | 1.56% |
| 2009 | train | 14588 | 0.607 | 1.279 | 0.88% |
| 2010 | train | 19037 | 0.811 | 1.842 | 2.06% |
| 2011 | train | 18539 | 0.753 | 1.733 | 1.47% |
| 2012 | train | 20676 | 0.582 | 1.064 | 0.63% |
| 2013 | train | 18129 | 0.503 | 0.830 | 0.26% |
| 2014 | train | 16941 | 0.474 | 0.851 | 0.33% |
| 2015 | train | 16083 | 0.540 | 1.258 | 0.93% |
| 2016 | train | 19484 | 0.599 | 1.443 | 1.06% |
| 2017 | train | 16995 | 0.480 | 1.037 | 0.45% |
| 2018 | train | 16374 | 0.509 | 0.955 | 0.39% |
| 2019 | train | 19520 | 0.494 | 0.922 | 0.27% |
| 2020 | train | 14056 | 0.538 | 1.973 | 0.61% |
| 2021 | train | 19264 | 0.548 | 1.870 | 0.70% |
| 2022 | train | 17778 | 0.499 | 1.611 | 0.55% |
| 2023 | train | 19349 | 0.504 | 1.964 | 0.71% |
| 2024 | train | 21730 | 0.434 | 0.980 | 0.30% |
| 2025 | validation | 10248 | 0.418 | 0.699 | 0.15% |
| 2025 | test | 10994 | 0.342 | 0.588 | 0.02% |
