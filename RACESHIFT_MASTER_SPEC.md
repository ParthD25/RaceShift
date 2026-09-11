# RaceShift Master Project Standard

## Motorsport Next-Lap Forecasting with Forward-Forward Regression

**Version:** 0.4.0  
**Status date:** 2026-09-07  
**Primary series:** Formula 1  
**Planned extensions:** IMSA, FIA WEC / Le Mans  
**Training environment:** Google Colab  
**Product runtime:** local machine, React/Vite + FastAPI  
**Primary trainable model:** RaceShift FFR, Forward-Forward regression inspired by FFR 2026  

---

# 1. Product thesis

RaceShift is a local-first motorsport machine-learning system that forecasts a driver's **next lap before it happens** from the complete race context available at the end of the current lap.

The core research question is:

> Given only information legitimately known through lap N, how accurately can a backpropagation-free Forward-Forward regression model forecast lap N+1 across drivers, circuits, weather, tyre states, teams and future seasons?

RaceShift is not a generic Formula 1 dashboard and is not an LLM wrapper. The project is valuable only if the prediction is leakage-safe, evaluated chronologically, and demonstrably useful beyond a trivial previous-lap baseline.

---

# 2. Exact project goal

For every completed lap `N`, RaceShift should build a state containing:

- driver
- team/constructor
- circuit/event/session
- current and recent lap pace
- sector pace
- tyre compound
- tyre age
- fresh/used tyre state
- stint
- position
- air temperature
- track temperature
- humidity
- pressure
- rain state
- wind speed
- wind direction
- traffic gap when available
- race/track status when available
- compact telemetry summary from completed lap N when available
- historical pace under comparable conditions from previous events only

The model then predicts:

```text
lap_time(N + 1)
```

The implemented training target is the residual from a local five-lap pace baseline:

```text
target_delta_vs_rolling5_s = lap_time(N + 1) - rolling_median_5_at_N
```

Final prediction:

```text
predicted_next_lap = rolling_median_5_at_N + predicted_residual
```

---

# 3. Why this prediction problem is legitimate

A model that receives the full speed trace of the same lap it is asked to predict can obtain misleadingly strong results because speed over distance already contains most of the elapsed-time information.

RaceShift avoids that trap:

```text
completed information through lap N
               |
               v
         RaceShift FFR
               |
               v
         forecast lap N+1
```

No measurement from lap N+1 is permitted in the input.

---

# 4. Product user

Primary users:

- motorsport data-science students
- ML engineers interested in racing telemetry
- technical motorsport fans
- analysts exploring historical pace
- researchers comparing backpropagation-free learning methods

The portfolio demo should be understandable by both technical and nontechnical recruiters.

---

# 5. Product workflow

## Default offline workflow

1. User installs the project.
2. User runs `npm run dev`.
3. RaceShift launches FastAPI on localhost and the Vite UI.
4. User imports a local CSV/Parquet dataset or uses the included synthetic fixture.
5. User selects a trained RaceShift FFR artifact.
6. User runs a next-lap forecast.
7. UI displays predicted pace, calibrated interval, model disagreement, tyre/weather context and prior-race context.

## Training workflow

1. User opens the Google Colab notebook.
2. User mounts Drive.
3. User collects/downloads real motorsport data.
4. RaceShift normalizes it to the common schema.
5. Leakage-safe features are generated.
6. Data is split chronologically.
7. RaceShift FFR trains with local Forward-Forward updates.
8. Baselines and frozen foundation models are evaluated.
9. The real artifact is exported to the local project.

## Optional live workflow

Live data is not required.

OpenF1 currently offers authenticated real-time data through REST, MQTT and WebSocket for sponsor users. RaceShift may record that feed in the local Python backend and then use completed-lap updates for forecasting.

Credentials never enter React/Vite.

---

# 6. Non-goals

Version 1 will not:

- predict the final race result as its core task
- pretend to reproduce proprietary F1 team simulation systems
- claim private fuel load as known ground truth
- use target-lap telemetry to predict the same target lap
- rely on OpenAI/Claude/Gemini APIs
- require a paid live timing subscription
- scrape an unlicensed live F1 source
- let an LLM invent pit-strategy recommendations
- mix F1, IMSA and WEC data into one model without series-aware validation
- use standard global backpropagation in the primary trainable model
- claim that Forward-Forward is superior before real evaluation proves it

---

# 7. Local-first architecture

```text
                DATA SOURCES
  Local files | FastF1 | OpenF1 | HF/Kaggle
                     |
                     v
             normalization layer
                     |
                     v
            RaceShift lap schema
                     |
                     v
          leakage-safe features
                     |
          +----------+----------+
          |                     |
          v                     v
   Google Colab training    local data files
          |                     |
          v                     |
 RaceShift FFR artifact         |
          |                     |
          +----------+----------+
                     |
                     v
              FastAPI localhost
                     |
                     v
               React/Vite UI
```

The browser is a presentation/control surface. It is not a secret store, training environment or third-party API credential holder.

---

# 8. Technology standard

## Local application

- React
- TypeScript
- Vite
- Recharts
- FastAPI
- Uvicorn

## Data / ML

- Python
- pandas
- NumPy
- PyArrow
- scikit-learn
- PyTorch tensor operations
- FastF1
- requests
- Hugging Face Hub/Datasets for optional research sources

TensorFlow is not part of the base stack because the project has one primary training framework and does not benefit from maintaining two CUDA/deep-learning stacks.

PyTorch is used for tensor/GPU operations. RaceShift FFR does not call PyTorch's global backward/autograd training path.

---

# 9. Forward-Forward-only training policy

The primary trainable model must satisfy all of these conditions:

```text
Tensor.backward()        = prohibited
autograd.grad()          = prohibited
global backward graph   = prohibited
standard LoRA/QLoRA     = prohibited for primary training
end-to-end BP fine-tune = prohibited for primary training
```

The current implementation:

```text
src/raceshift/models/forward_forward_regressor.py
```

is inspired by:

1. Hinton, The Forward-Forward Algorithm: Some Preliminary Investigations  
   https://arxiv.org/abs/2212.13345

2. FFR: Forward-Forward Learning for Regression, 2026  
   https://arxiv.org/abs/2606.03927

3. Self-Contrastive Forward-Forward algorithm, Nature Communications 2025  
   https://www.nature.com/articles/s41467-025-61037-0

FFR is the most directly relevant recent work because it targets continuous regression with ordinal competitive goodness and a coarse-to-fine ladder.

The project must state that its architecture is **FFR-inspired**, not a verified reproduction of the paper.

---

# 10. RaceShift FFR architecture

## Input

A train-only transformed feature vector containing:

- current lap context
- lags 1 through 4 for important numeric context
- categorical one-hot state
- historical priors

## Coarse-to-fine local layers

Suggested production research ladder:

```text
Layer 1:  8 ordinal target groups
Layer 2: 16 ordinal target groups
Layer 3: 32 ordinal target groups
Layer 4: 64 ordinal target groups
```

Each local layer:

1. row-normalizes its input
2. computes hidden activations
3. partitions hidden units into ordered target groups
4. computes group goodness as mean squared activation
5. compares goodness to a soft target distribution based on target distance
6. analytically computes only the current layer's local parameter update
7. updates weights using an explicit local Adam implementation
8. normalizes its hidden output for the next layer

No gradient is sent through earlier or later layers.

## Readout

After local training:

- concatenate goodness vectors across layers
- append each layer's local prediction
- solve a ridge regression head in closed form

## Uncertainty

Current uncertainty signals:

- validation-residual quantiles for an 80% interval
- standard deviation/disagreement across local layer predictions

---

# 11. Full feature contract

## Current-lap numeric state

- lap number
- lap time
- sector 1/2/3
- tyre life
- fresh tyre indicator
- stint
- position
- air temperature
- track temperature
- humidity
- pressure
- rainfall
- wind speed
- wind direction sine
- wind direction cosine
- gap ahead
- gap behind

## Completed-lap telemetry summary, optional

- mean speed
- maximum speed
- mean throttle
- braking-active fraction
- mean RPM
- maximum RPM
- gear changes
- DRS-active fraction

## Derived pace state

- rolling median 3
- rolling median 5
- rolling mean sector 1/2/3 over 3 laps
- recent pace trend
- current lap delta vs rolling-five baseline

## Five-lap flattened context

For selected core timing/weather/telemetry fields, lag 1 through lag 4 are included in addition to the current lap.

This supplies temporal context without recurrent or attention backpropagation.

## Historical features from prior events only

- driver/circuit median pace
- team/circuit median pace
- compound/circuit median pace
- driver/circuit/compound median pace
- team/circuit/compound median pace
- driver global median
- team global median
- weather-matched median
- driver weather-matched median
- team weather-matched median

## Categorical state

- series
- event
- circuit
- session
- driver
- team
- manufacturer
- car class
- compound
- tyre manufacturer
- track status

---

# 12. Tyre manufacturer rule

In Formula 1, tyre manufacturer is effectively constant within the championship, so it does not provide useful F1 training variance.

RaceShift therefore prioritizes:

- compound
- tyre life
- fresh/used state
- stint
- historical compound behavior

The normalized schema still keeps `tyre_manufacturer` because IMSA/WEC/general motorsport can contain meaningful manufacturer differences.

---

# 13. Weather and wind rule

Wind direction is circular.

Store:

```text
wind_dir_sin = sin(direction)
wind_dir_cos = cos(direction)
```

Do not feed raw direction alone as a linear number because 359 degrees and 1 degree are physically close.

Weather-matched historical priors currently use condition bins for:

- track temperature
- air temperature
- humidity
- wind speed
- rain state
- same circuit
- same compound

Alternative matching methods may be tested later, but must use earlier events only.

---

# 14. Leakage standard

Leakage is a release-blocking defect.

Forbidden:

- `target_next_lap_time_s` in features
- target-lap sectors
- target-lap speed/throttle/brake/RPM
- target-lap timestamps that reveal duration
- current-event rows inside historical priors
- post-race final result when predicting an earlier lap
- fitting preprocessing on validation/test data
- random lap-level split for the headline result

Required protections:

- groupwise `shift(-1)` target
- adjacency check `next_lap_number - lap_number == 1`
- train-only preprocessing
- target blocklist
- prior-event shift before historical expanding median
- chronological split
- automated tests

---

# 15. Data source standard

## FastF1

Primary reproducible historical F1 source.

Use for lap timing, tyres, weather, track status and optional completed-lap telemetry summaries.

## OpenF1

Historical from 2023 onward is free and keyless.

Current useful public fields include:

- lap timing
- car speed/throttle/brake/RPM/gear/DRS
- weather
- stints
- pit
- position
- intervals
- race control

Real-time is optional sponsor access and must be backend-only from a credential perspective.

## Hugging Face datasets

Curated in `dataset_manifest.json` and `docs/DATA_SOURCES.md`.

Important examples:

- F1 StratLab Strategy Dataset, 2023-2025, ~16.1 GB
- F1 Corner Telemetry 2024-2025, 7.13 GB raw CSV
- Renumics Montreal 2023 telemetry, 1,317 rows
- IMSA endurance dataset, ~264 MB repository

## Kaggle

Use only as benchmark/fallback datasets with license verification.

## WEC / Le Mans

Build adapters for user-acquired timing files. Do not redistribute rights-sensitive raw archives without permission.

---

# 16. Live-data standard

Live data is an optional adapter, not a product dependency.

Supported conceptual modes:

```text
MODE A  offline local CSV/Parquet
MODE B  free historical OpenF1
MODE C  FastF1 historical/cache
MODE D  optional authenticated OpenF1 live snapshot/recorder
MODE E  post-session live-timing recording imported as local data
```

RaceShift should never claim a live connection when it is rendering fixture/historical data.

---

# 17. Hugging Face model strategy

The project reviewed current time-series foundation models including:

- Amazon Chronos-2
- Chronos-2 Small
- IBM Granite Tiny Time Mixer R3
- Datadog Toto 2.0 4M and 22M
- NX-AI TiRex-2
- Google TimesFM 2.5
- Salesforce Moirai 2.0 R Small
- MOMENT-1 Small
- Lag-Llama
- Timer-S1

Current policy:

> use them as frozen zero-shot/reference benchmarks, not fine-tuning bases, because standard fine-tuning would violate the no-global-backprop project rule.

General LLMs such as Qwen/Kimi are not primary numerical forecasters.

---

# 18. Baseline standard

Required comparisons:

1. previous lap
2. rolling-five median
3. ridge/linear regression
4. tree boosting
5. RaceShift FFR
6. frozen TS foundation models where practical

Forward-Forward is the research focus, not a reason to ignore simpler baselines.

---

# 19. Evaluation standard

Primary metric:

```text
MAE in seconds
```

Secondary:

- RMSE
- median absolute error
- p90 absolute error
- signed bias
- interval coverage
- interval width
- artifact size
- CPU inference latency
- peak training memory when measurable

Break down errors by:

- circuit
- driver
- team
- compound
- tyre-age bucket
- dry/wet
- temperature regime
- traffic regime
- track status

---

# 20. Chronological validation standard

Do not randomly split laps from the same races for the main result.

Recommended initial real split if training through 2025:

```text
train       2019-2023
validation  2024
test        2025
```

Later:

```text
train       <= 2025
holdout     2026
```

The 2026 holdout is valuable for domain-shift analysis under the new technical-regulation era.

---

# 21. Required ablation studies

A real portfolio report should include:

1. current lap only vs five-lap context
2. weather removed vs included
3. wind removed vs included
4. historical priors removed vs included
5. telemetry summaries removed vs included
6. driver/team/circuit categories removed vs included
7. different FFR ladder depths
8. different ordinal group schedules
9. tree baseline vs FFR
10. frozen foundation model vs FFR

This lets RaceShift answer not only “what scored best?” but “what information actually helped?”

---

# 22. Data schema standard

Every normalized lap should use compatible fields from this family:

```text
series
season
round_number
event
event_date
circuit
session
driver
driver_number
team
manufacturer
car_model
car_class
lap_number
lap_time_s
sector1_s
sector2_s
sector3_s
compound
tyre_manufacturer
tyre_life
fresh_tyre
stint
position
track_status
pit_in
pit_out
is_accurate
deleted
air_temp_c
track_temp_c
humidity_pct
pressure_mbar
rainfall
wind_speed_ms
wind_direction_deg
gap_ahead_s
gap_behind_s
mean_speed_kph
max_speed_kph
mean_throttle_pct
brake_active_pct
mean_rpm
max_rpm
gear_changes
drs_active_pct
```

Adapters may leave unsupported optional fields null.

---

# 23. UI standard

The UI is a dark motorsport analysis workspace with clear separation between:

- fixture/example visuals
- local imported data
- real model results

Routes:

- Overview
- Forecast
- Telemetry
- Experiments
- Datasets
- Models
- Compare Drivers
- Strategy Insights
- Settings

The Forecast page is the core real workflow and must remain connected to the local API.

The UI must not fabricate real F1 model accuracy.

---

# 24. Local API standard

FastAPI endpoints currently include:

```text
GET  /api/health
GET  /api/runtime
GET  /api/setup
GET  /api/models
GET  /api/datasets
POST /api/import
POST /api/forecast/latest
GET  /api/experiments
GET  /api/imports/{file}/summary
GET  /api/forecast            (query-string alias of POST /api/forecast/latest)
```

Security:

- bind localhost
- local CORS only
- imported file type/size checks
- source path restricted to project data
- artifact path restricted to project artifacts
- secrets never returned

---

# 25. Security standard

Never commit:

- `.env`
- source-service passwords
- OAuth tokens
- Kaggle keys
- Hugging Face private tokens
- F1TV authentication data
- raw licensed motorsport datasets unless redistribution is clearly permitted

Never load untrusted `model.pt` or `preprocessor.joblib` artifacts. Serialized Python objects are a deserialization security boundary.

---

# 26. Reproducibility standard

Every real run should save:

```text
model.pt
preprocessor.joblib
metrics.json
training_history.json
test_predictions.csv
```

`metrics.json` should record:

- model name/version
- training method
- global backprop false
- train/validation/test seasons
- feature contract version
- FFR group ladder
- hyperparameters
- test metrics
- source dataset identifiers

A future improvement should add git commit hash and environment/package versions.

---

# 27. Synthetic demo policy

The repository includes a synthetic fixture and small FFR artifact solely to prove:

```text
import -> features -> training -> artifact -> local inference -> UI
```

Synthetic metrics may never be placed on a resume or described as Formula 1 performance.

---

# 28. Current implementation status

Implemented:

- local React/Vite UI
- local FastAPI
- local import workflow
- FastF1 ingestion
- OpenF1 ingestion/normalization
- optional completed-lap telemetry summaries
- dataset/model research registries
- full-context feature engineering
- prior-event matched-condition features
- five-lap flattened context
- Forward-Forward ordinal regression layers
- explicit local Adam
- closed-form readout
- interval calibration
- local model save/load
- local forecast endpoint
- synthetic demo artifact
- Verified checks for leakage and feature availability, lap-state adjacency incl. yellow-flag laps and red-flag/safety-car restart laps, chronological and round splits, finite-difference gradients and layer locality, artifact forecasting and backtesting, ONNX export and the local API
- real FastF1 collection and experiment runner with baselines, FFR depth/group ladders, feature ablations, circuit holdout and resource metrics

Validated on real F1 so far: see `reports/` for the measured tables. Not yet validated:

- wet-race performance in isolation
- 2026 domain shift beyond the single no-retraining test in `reports/domain_shift_2026`
- live-stream reliability
- endurance-series generalization
- corner telemetry Forward-Forward learning

---

# 29. Staged completion plan

## Milestone 1 - local product foundation

Status: implemented.

Definition:

- one-command local launch
- working import and forecast flow
- offline demo
- documented setup

## Milestone 2 - real F1 training

Use Colab to train/evaluate on historical F1.

Definition of done:

- chronological real split
- baseline comparison
- FFR metrics
- calibrated interval result
- no leakage violations
- real artifact loads in local UI

## Milestone 3 - research report

Definition:

- ablation table
- per-circuit/per-weather error analysis
- memory/latency measurement
- frozen HF model benchmark
- documented limitations

## Milestone 4 - 2026 domain shift

Definition:

- train through 2025
- untouched 2026 holdout
- quantify degradation under regulation change
- optionally test local adaptation methods that still satisfy no-backprop rule

## Milestone 5 - telemetry/corner research

Investigate FFR/SCFF-style local learning for corner telemetry.

## Milestone 6 - endurance adapter

Add IMSA first, then WEC/Le Mans user-acquired data.

---

# 30. Definition of done for the portfolio version

RaceShift is portfolio-ready when all of the following are true:

- project installs from documented instructions
- `npm run dev` starts UI + API locally
- local import/forecast workflow works
- real F1 artifact is trained and stored separately from synthetic demo
- no global backprop is used in primary training
- no target leakage is found
- future-season test is reported
- FFR is compared against simple baselines
- uncertainty is calibrated and reported
- at least one feature ablation is completed
- UI clearly distinguishes fixture from real data
- secrets are not committed
- raw data licenses are documented
- README includes reproducible training/runtime instructions
- resume language matches measured results

---

# 31. Resume positioning after real validation

Do not use numeric claims before real evaluation.

A safe non-numeric line once the real pipeline is completed:

> Built RaceShift, a local-first Formula 1 next-lap forecasting system using a Forward-Forward regression architecture with no global backpropagation, integrating tyre, weather, wind, traffic, telemetry and prior-race context through leakage-safe chronological evaluation.

After real metrics are validated, add one measured result such as future-season MAE and clearly name the holdout period.

---

# 32. Project rule for future agents

Before adding any feature or model, answer:

1. Does this improve next-lap forecasting, explainability, evaluation or data reliability?
2. Is the required information actually available before the target lap?
3. Does it preserve the no-global-backprop training rule?
4. Can it run in the Colab/local architecture?
5. Does its license permit our intended use?
6. Can we measure whether it helped?

If not, do not add it merely because it is fashionable or “AI.”
