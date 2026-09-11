# Seed sensitivity: f1_2025h2 / ffr_small.json

Same data, split and hyperparameters; only the random seed differs (weight initialisation and batch order).

| Run | Test MAE (s) | Test RMSE (s) | p90 (s) | Within 0.5 s | 80% coverage | Train (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 42 (headline) | 0.328 | 0.501 | 0.694 | 82.2% | 0.859 | 495 |
| seed 1 | 0.330 | 0.502 | 0.694 | 82.2% | 0.862 | 486 |
| seed 2 | 0.327 | 0.501 | 0.690 | 82.2% | 0.859 | 496 |
| seed 3 | 0.327 | 0.501 | 0.693 | 82.3% | 0.863 | 491 |

**Spread across the swept seeds** (mean ± population std):

- Test MAE (s): 0.328 ± 0.001 (n = 3)
- Test RMSE (s): 0.501 ± 0.001 (n = 3)
- p90 (s): 0.692 ± 0.002 (n = 3)
- Within 0.5 s: 0.822 ± 0.001 (n = 3)
- 80% coverage: 0.861 ± 0.002 (n = 3)

A difference between two FFR variants smaller than about two standard deviations of the test MAE here should be read as noise, not as an effect.
