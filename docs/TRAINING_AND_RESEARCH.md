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

The source file is checked by tests for `.backward(` and `autograd.grad(`.

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

Headline metric: MAE in seconds.

Also report:

- RMSE
- median absolute error
- p90 absolute error
- signed bias
- 80% interval coverage
- interval width
- artifact size
- CPU inference latency

Break down by circuit, driver, team, compound, tyre age, weather regime, traffic regime and track status.

## Chronological validation

Recommended:

```text
2019-2023 train
2024 validation
2025 test
2026 future domain-shift holdout
```

Do not use a random row split as the headline result.

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

Depth ladder: FFR-S (256 → 128), FFR-M (512 → 384 → 256 → 192), FFR-L (1024 → 768 → 512 →
384 → 256). Group ladders: 4/8/16/32, 8/16/32/64, 16/32/64/64. Feature ablations drop one
taxonomy group at a time. Splits: season-forward, season-round (early/late holdout season),
circuit holdout (`--holdout-event`), and the 2026 domain-shift holdout once 2026 rounds are
collected.

## Data hygiene decisions that changed the results

- Pit-in, pit-out, safety-car, VSC, red-flag, deleted and inaccurate laps are never
  training rows or targets, and every lag or rolling statistic is scoped to the current run
  of consecutive valid laps (`docs/FEATURE_CONTRACT.md`).
- Lap-time-scale features are relative to the current rolling pace; the first real run with
  absolute features had ridge extrapolating to −9 s residuals on unseen seasons.
- Features observed in under 5% of training rows are dropped; weather-matched priors were
  producing standardized shifts of 8+ between train and test after imputation.
- The training residual target is winsorized to ±6 s; the raw target ranges from −54 s to
  +18 s on real races and least-squares fits were dominated by the tails.

## Known gaps

- Race-control messages are not yet used to flag laps affected by incidents that are not
  encoded in the track status string.
- Historical priors are medians over earlier events; a nearest-neighbour similarity
  retrieval over normalised conditions is the planned replacement.
- `scripts/eval_chronos2_zeroshot.py` expects the light table from `scripts/build_lap_dataset.py`
  and an optional `chronos` install; it has not been run on real data yet.
