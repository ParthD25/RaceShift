# Seed sensitivity: f1_2025h2 / ffr_small.json

Same data, split and hyperparameters; only the random seed differs (weight initialisation and batch order).

| Run | Test MAE (s) | Test RMSE (s) | p90 (s) | Within 0.5 s | 80% coverage | Train (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 42 (headline) | 0.350 | 0.590 | 0.729 | 80.5% | 0.858 | 390 |
| seed 1 | 0.352 | 0.591 | 0.735 | 80.2% | 0.858 | 537 |
| seed 2 | 0.351 | 0.590 | 0.729 | 80.3% | 0.860 | 377 |
| seed 3 | 0.348 | 0.589 | 0.732 | 80.7% | 0.861 | 390 |

**Spread across the swept seeds** (mean ± population std):

- Test MAE (s): 0.350 ± 0.002 (n = 3)
- Test RMSE (s): 0.590 ± 0.001 (n = 3)
- p90 (s): 0.732 ± 0.003 (n = 3)
- Within 0.5 s: 0.804 ± 0.002 (n = 3)
- 80% coverage: 0.859 ± 0.001 (n = 3)

A difference between two FFR variants smaller than about two standard deviations of the test MAE here should be read as noise, not as an effect.
