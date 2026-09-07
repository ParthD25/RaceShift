# Training and Research Standard

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

## Required ablations

- 1 lap vs 3 laps vs 5 laps context
- no historical priors vs historical priors
- weather removed
- tyre features removed
- driver/team categorical state removed
- 2 vs 3 vs 4 vs 5 Forward-Forward layers
- node-width ladder comparison
- ordinal group count comparison
- local optimizer/learning-rate comparison

## Research references

- Geoffrey Hinton, *The Forward-Forward Algorithm: Some Preliminary Investigations*, arXiv:2212.13345
- *FFR: Forward-Forward Learning for Regression* (2026 preprint), used as inspiration for ordinal/coarse-to-fine regression design
- Self-Contrastive Forward-Forward work, used as a reference for sequential/local representation learning

RaceShift must describe itself as FFR-inspired unless and until the implementation is formally reproduced against the exact paper protocol.
