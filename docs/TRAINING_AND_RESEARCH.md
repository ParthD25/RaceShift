# Training and Research Standard

## Research question

> Can Forward-Forward regression, trained with local layer-wise updates and no global
> backpropagation, approach the next-lap forecasting accuracy of conventional baselines
> while reducing training-memory requirements?

Forward-Forward is not assumed to be better than backpropagation. The FFR literature reports
recovering most but not all of backprop accuracy on regression benchmarks with lower peak
training memory, and the broader Forward-Forward literature acknowledges a gap to
state-of-the-art backprop on standard benchmarks. RaceShift tests the trade-off, so every run
records accuracy **and** resources: training wall time, peak RSS, traced peak memory,
inference latency and artifact size.

## Primary architecture

RaceShift FFR is a deep Forward-Forward regression model.

Production ladder:

```text
encoded input
   -> Local FF layer 1: 512 nodes, 8 target groups
   -> Local FF layer 2: 384 nodes, 16 target groups
   -> Local FF layer 3: 256 nodes, 32 target groups
   -> Local FF layer 4: 192 nodes, 64 target groups
   -> concatenate goodness vectors + local layer predictions
   -> closed-form ridge readout
```

Large Colab ladder:

```text
1024 -> 768 -> 512 -> 384 -> 256 nodes
```

The large ladder is an experiment, not a guaranteed improvement.

## Local learning

Each layer:

1. normalizes its input rows
2. computes ReLU activations
3. partitions hidden nodes into ordinal target groups
4. computes mean-squared activation per group as goodness
5. compares the goodness distribution with a soft ordinal target distribution
6. computes the derivative of the **local** objective analytically
7. updates only that layer with a local Adam state
8. emits normalized activations to the next layer

There is no end-to-end gradient chain.

The source file is checked by a test for `.backward(` and `autograd.grad(`; the blind reviewers ran the same scan.

## Regression target

The model predicts:

```text
next_lap_time - rolling_median_5_at_current_lap
```

The final next-lap estimate is:

```text
rolling_median_5 + predicted_residual
```

This reduces circuit-scale differences while preserving a prediction measured in seconds.

## Required baselines

- previous lap
- rolling-five median
- ridge regression
- gradient/tree baseline
- frozen zero-shot time-series foundation models when practical

Frozen open models are benchmarks only. Standard LoRA/QLoRA fine-tuning is excluded from the primary project because it would reintroduce global backpropagation.

`scripts/train_baselines.py` takes the same raw lap table as `scripts/train_ffr.py`, builds the same
full-context feature table, applies the same chronological split and the same train-only
preprocessing (`raceshift.features.preprocessing.make_preprocessor`). Ridge and gradient boosting
predict the same residual target. Its `metrics.json` is listed by `GET /api/experiments` next to FFR
runs, so the comparison is always visible.

On the synthetic fixture the tree baseline beats the demo FFR artifact. That is reported as-is: the
fixture is a pipeline check, and the real question is answered only on unseen Formula 1 seasons.

## Evaluation

Headline metric: MAE in seconds, the average distance between the predicted and the true
next lap. This is regression, so there is no classification accuracy; a 0.43 s MAE on a 90 s
lap is a 0.4% relative error, and lower is better.

Also report:

- RMSE
- median absolute error
- p90 absolute error
- signed bias
- share of laps predicted within 0.5 s and within 1 s (the accuracy-style view; higher is better)
- MAPE and R² on the true lap time
- 80% interval coverage
- interval width
- artifact size
- CPU inference latency

Every metric is recorded for the training split as well as validation and test.

Break down by circuit, driver, team, compound, tyre age, weather regime, traffic regime and track status.

## Chronological validation

Headline protocol (FastF1 tier, every round, every team):

```text
2018-2024              train
2025 rounds 1-12       validation (interval calibration, config selection)
2025 rounds 13-24      test
2026                   domain-shift evaluation with the 2024-trained model, no retraining
```

Extension experiment: the same validation and test rows, with training extended back to
2000 using the legacy Ergast tier (lap times and positions only). This asks whether eighteen
extra seasons of low-detail history help or hurt, and is reported next to the headline
table, never merged into it. Do not use a random row split as the headline result.

## Experiment protocol

`scripts/run_experiments.py` runs the baselines and any list of FFR configs on the same table
and split, as separate processes so peak memory is measured per run, and writes
`reports/<name>/summary.{json,md}`.

```bash
python scripts/run_experiments.py --input data/processed/f1_laps.parquet --name f1_2025h2 \
  --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12 \
  --ffr configs/ffr_small.json configs/ffr_production.json configs/ffr_colab_large.json \
      configs/ffr_m_groups_coarse.json configs/ffr_m_groups_fine.json \
  --ablate historical_numeric temporal_numeric static_categorical
```

`scripts/full_pipeline.sh` chains collection, table building, the matrix above, the Monza
circuit holdout, the 2026 domain-shift run and the legacy extension, and is resumable at
every stage.

Depth ladder: FFR-S (256 → 128), FFR-M (512 → 384 → 256 → 192), FFR-L (1024 → 768 → 512 →
384 → 256). Group ladders: 4/8/16/32, 8/16/32/64, 16/32/64/64. Feature ablations drop one
taxonomy group at a time. Splits: season-forward, season-round (early/late holdout season),
circuit holdout (`--holdout-event`), and the 2026 domain-shift holdout once 2026 rounds are
collected.

## Overfitting and memorisation checks

A model that memorises its training laps fits them far better than unseen laps. RaceShift
makes that visible rather than assuming it away:

- Every run records **train, validation and test** metrics in `metrics.json` (FFR and every
  baseline), so the generalisation gap (test MAE minus train MAE) is a first-class number.
- `scripts/generalization_gap.py` rebuilds an artifact's exact table and split, scores the
  saved weights on all three splits, refits ridge and gradient-boosted trees on the same
  training rows, and writes `reports/<name>/generalization.md` with a per-season table.
- Preprocessing (imputation, scaling, one-hot vocabularies) is fitted on training rows only;
  validation and test are transformed with training statistics.
- The FFR readout is a closed-form ridge over layer features (`ridge_alpha`), hidden layers
  use weight decay, update-norm clipping and a small learning rate; interval calibration uses
  validation residuals, never training residuals.
- `train_ffr.py --wandb` logs per-layer local losses and all split metrics to Weights &
  Biases when `wandb` is installed (`pip install -e ".[research]"`; `WANDB_MODE=offline` works
  without an account). It is optional and never required for a run.

Measured on the 2018-2024 → 2025 split (`reports/f1_2025h2/generalization.md`): every model,
FFR and baselines alike, has a **higher** error on its training laps (about 0.50 s) than on
validation (0.43 s) or test (0.35 s). There is no memorisation; the training seasons simply
contain more disrupted laps (training RMSE 1.5 s vs 0.6 s on test), and the per-season table
attributes that to specific years. If anything FFR under-fits: its train and validation errors
are close and both trail the tree model.

## Data hygiene decisions that changed the results

- Pit-in, pit-out, yellow-flag, safety-car, VSC, red-flag, deleted and inaccurate laps are
  never training rows or targets, and every lag or rolling statistic is scoped to the current run
  of consecutive valid laps (`docs/FEATURE_CONTRACT.md`).
- Lap-validity rules v3 exclude yellow-flag laps and the first lap after a safety-car
  period. A blind test on the 2026 races (`reports/blind_2026/REPORT.md`) found
  yellow laps that had passed as "clean" carrying 1.9 s mean error with a +1.5 s bias, and
  the lap after a safety car entering the rolling-5 baseline 20 s slow. Measured over
  2018-2026, the two rules remove 4.6% of previously valid laps and cut the naive
  previous-lap error on the 2025 test rounds from 0.357 s to 0.336 s: the targets are cleaner,
  so every model's error falls and version 2 and version 3 numbers must not be compared.
- Lap-validity rules v2 also exclude the **restart lap** after a red flag. Under v1 the
  first timed lap after a stoppage (pit-lane exit, formation lap, standing restart) carried a
  clear track status and an "accurate" marker, passed every rule, and produced 40-56 s
  errors that dominated RMSE on the Monza holdout. The 370 such laps in 2018-2026 are 0.2%
  of laps but were the largest single source of error. Every result in the README was
  re-run under v2; the version is recorded in every `metrics.json`.
- Lap-time-scale features are relative to the current rolling pace; the first real run with
  absolute features had ridge extrapolating to −9 s residuals on unseen seasons.
- Features observed in under 5% of training rows are dropped; weather-matched priors were
  producing standardized shifts of 8+ between train and test after imputation.
- The training residual target is winsorized to ±6 s; the raw target ranges from −54 s to
  +18 s on real races and least-squares fits were dominated by the tails.

## What "Forward-Forward" means here, and what it does not

RaceShift FFR trains each layer with its own objective: hidden units are split into ordered
groups, the mean squared activation of each group is its "goodness", and a softmax over group
goodness is trained by cross entropy against a soft ordinal target derived from the scaled
residual. The gradient of that loss with respect to the layer's weight and bias is derived by
hand (`_FFLocalLayer.local_gradient`) and applied with a local Adam step. Layer *k*+1 receives
the normalised output of layer *k* as a plain array; no quantity computed in layer *k*+1 ever
reaches layer *k*. Two tests make this concrete (`tests/test_forward_forward.py`), and the blind
reviewers reproduced both from the public code: the analytic gradient compared with finite differences entry by entry
(max difference 2e-10), and a layer's gradient and update bit-identical when every later
layer's weights are replaced with random values.

This is **greedy layer-wise supervised training with local objectives**. It shares with
Hinton's Forward-Forward algorithm the absence of a backward pass across layers and the use of
a per-layer goodness measure, but it does **not** use positive and negative data passes. Calling
it "Forward-Forward regression" is a shorthand for the family (local goodness objectives,
forward-only training), not a claim to reproduce the original algorithm.

The 80% interval is the 80th percentile of |validation residual|, computed on the clipped (±6 s)
validation target. Clipping only affects residuals beyond ±6 s, which are far outside the
80th percentile (about 0.6 s), so the interval is unaffected; the coverage reported in the tables
is always evaluated on the raw, unclipped target.

## Seed sensitivity

Every headline row is one run with seed 42. Where `scripts/seed_sweep.py` has been run, the
resulting `reports/<name>/seeds.md` gives the mean and standard deviation of test MAE across
seeds for FFR-S, which is the yardstick for deciding whether a difference between two FFR
variants is an effect or noise.

## Known gaps

- The 2018 Italian Grand Prix race is missing from the FastF1 tier: the live-timing archive
  fails to load timing data for that session (`Failed to load timing data!`), so the
  collector records it as a failure and every other 2018-2026 round is present.
- Race-control messages are not yet used to flag laps affected by incidents that are not
  encoded in the track status string. Red-flag and safety-car restart laps and yellow-flag
  laps are handled by the lap-validity rules (v3), but a slow lap caused by debris or a
  driver pitting for damage under green still passes every rule.
  Across 2018-2026, 41 laps out of 174k valid ones are more than 1.4× their driver's race
  median without any flag; most are 2020 Austrian Grand Prix laps where the status string
  lags the safety-car deployment.
- Historical priors are medians over earlier events; a nearest-neighbour similarity
  retrieval over normalised conditions is the planned replacement.
- `scripts/eval_chronos2_zeroshot.py` expects the light table from `scripts/build_lap_dataset.py`
  and an optional `chronos` install; it has not been run on real data yet.
