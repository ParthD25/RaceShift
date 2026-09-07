# RaceShift Master Specification v0.4

## Product thesis
RaceShift is a local-first motorsport ML system that forecasts a driver's **next lap** using only information legitimately available through the current completed lap. It is not an F1 chatbot and not a generic dashboard. The research focus is whether a deep Forward-Forward regression architecture can produce competitive next-lap forecasts without a global backward pass.

## Primary goal
Predict `lap N+1` from full context available at the end of `lap N`:

- driver and team/constructor
- circuit/event/session
- current and recent lap pace
- sector times
- tyre compound, tyre age, fresh/used state, stint
- position and traffic gaps
- air and track temperature
- humidity, pressure, rainfall
- wind speed and wind direction
- track status
- optional completed-lap telemetry summaries
- matched historical pace from earlier races only

The model predicts the residual:

```text
next_lap_time - rolling_median_5
```

Final forecast:

```text
rolling_median_5 + predicted_residual
```

## Training policy
The primary trainable model is **RaceShift FFR**.

No global backpropagation is allowed in the primary training path:

```text
Tensor.backward()      prohibited
autograd.grad()        prohibited
end-to-end BP          prohibited
LoRA/QLoRA fine-tune   prohibited for the primary model
```

Each layer learns independently from a local ordinal-goodness objective with a manually derived local update. The final numeric readout is solved in closed form with ridge regression.

## Deep multi-layer architecture
Production default:

```text
encoded input
  -> 512 hidden nodes, 8 ordinal groups
  -> 384 hidden nodes, 16 ordinal groups
  -> 256 hidden nodes, 32 ordinal groups
  -> 192 hidden nodes, 64 ordinal groups
  -> goodness + local predictions from all layers
  -> closed-form ridge readout
```

Large Colab research configuration:

```text
1024 -> 768 -> 512 -> 384 -> 256 hidden nodes
```

The larger model is an ablation, not an assumed improvement.

## Why multiple layers and nodes matter
Each local layer is expected to learn a different granularity of the continuous target. Earlier layers are coarser, later layers have more ordinal groups and can represent finer lap-time residual structure. Layer disagreement is also used as an uncertainty signal.

## Feature engineering rules
- 5-lap temporal context through explicit lagged features
- circular wind encoding using sine/cosine
- historical priors computed from earlier events only
- train-only preprocessing
- target adjacency check: `next_lap_number - lap_number == 1`
- no target-lap telemetry, sectors or elapsed-time-revealing fields

## Historical context features
RaceShift calculates earlier-event medians for:

- driver + circuit
- team + circuit
- compound + circuit
- driver + circuit + compound
- team + circuit + compound
- driver global
- team global
- circuit + compound + matched weather bins
- driver + matched weather
- team + matched weather

## Local-first runtime

```text
local CSV/Parquet / FastF1 / OpenF1
                  |
                  v
         leakage-safe features
                  |
                  v
       RaceShift FFR artifact
                  |
                  v
         FastAPI localhost
                  |
                  v
           React/Vite UI
```

Default use needs no paid API and no secret key.

## Data modes
1. Local CSV/Parquet import
2. Free historical FastF1/OpenF1 data
3. Optional authenticated live adapter in the Python backend only
4. Hugging Face/Kaggle research datasets
5. Future IMSA/WEC adapters

## Required baselines
- previous lap
- rolling-five median
- ridge regression
- tree boosting
- frozen zero-shot time-series foundation models where practical

Open models from Hugging Face are comparison models only unless a non-backprop adaptation method is implemented.

## Validation
Headline split:

```text
2019-2023 train
2024 validation
2025 test
2026 future domain-shift holdout
```

Headline metric: MAE in seconds.

Also report RMSE, median AE, p90 AE, bias, interval coverage, interval width, artifact size and inference latency.

## Required ablations
- 1/3/5-lap context
- 2/3/4/5 local layers
- hidden-node width ladders
- 8/16/32/64 goodness-group ladder variants
- remove weather
- remove tyre features
- remove driver/team categorical state
- remove matched historical priors
- compare local optimizer settings

## Definition of done
RaceShift is portfolio-ready only when:

- real F1 data has been collected reproducibly
- leakage tests pass
- FFR trains without global backprop
- model beats at least the rolling-five baseline on unseen future-season data, or the report transparently explains why it does not
- UI reads real artifact metrics rather than invented numbers
- local setup works from the README
- no secret is required for the default workflow
- all claims on the resume are supported by saved experiment results

## Files to read
- `README.md`
- `docs/LOCAL_SETUP.md`
- `docs/FEATURE_CONTRACT.md`
- `docs/TRAINING_AND_RESEARCH.md`
- `docs/DATA_AND_MODEL_FINDINGS.md`
- `docs/SECURITY.md`
