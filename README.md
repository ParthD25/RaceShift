# RaceShift

**Forward-only motorsport pace forecasting.** RaceShift predicts a Formula 1 driver's next lap
from what is known at the end of the current lap:

- driver, constructor and circuit identity
- tyre compound, age and stint
- race context: position, gaps, consecutive clean laps
- weather: track and air temperature, humidity, pressure, rain, wind as sine/cosine
- recent pace: rolling medians, sector shares, four lagged laps (scoped to clean racing laps)
- historical matched conditions from earlier events only

**Research question**

> Can Forward-Forward regression, trained layer by layer with no global backpropagation,
> approach conventional forecasting accuracy while reducing training-memory requirements?

RaceShift does not assume the answer. Every run reports accuracy, calibration, training time,
peak memory, inference latency and artifact size next to naive, linear and tree baselines.

![RaceShift forecast page](docs/images/forecast.png)

## Run it

Python 3.11+ and Node.js 20+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[local]"
npm install
npm run dev
```

UI at <http://127.0.0.1:5173>, API at <http://127.0.0.1:8000/api/health>. Open **Forecast** and
press **Run forecast**: the repository ships the whole 2025 season of FastF1 race timing
(`data/imports/f1_2025_season.parquet`, 640 KB) and the real-data FFR-M model, so the first
forecast is a real driver's next lap at the 2025 Abu Dhabi Grand Prix, followed by a backtest of
that driver's last ten laps (predicted vs actual). Nothing needs to be trained or downloaded.
Also included: a synthetic fixture and a small demo model trained on it, both badged
*Synthetic* wherever they appear.

```bash
npm run test:py     # Python tests: leakage, feature availability, lap adjacency, gradients, splits, artifact, API
npm run build       # TypeScript check + Vite build
API_PORT=8010 WEB_PORT=5180 npm run dev                # if 8000 or 5173 is taken
python scripts/fetch_fastf1.py --year 2025 --event "Abu Dhabi" --session R \
    --output data/imports/abu_dhabi_2025.parquet       # one more race in about 15 s; the UI lists it
npm run demo:data && npm run demo:model                # regenerate the synthetic fixture and demo model (untracked output)
```

A step-by-step walkthrough with expected output is in `docs/LOCAL_SETUP.md`. There is no
live-timing connection; RaceShift reads local files only.

**How to read the numbers.** This is regression, not classification, so there is no
"accuracy" percentage. MAE is the average distance in seconds between the predicted and the
true next lap, and lower is better. Read the results as: the model is typically within about a
third of a second of the real next lap on laps of about 90 s, predicts 8 in 10 laps within
half a second and 19 in 20 within one second, and its "80% interval" contains the true lap
about 86% of the time. Every table also reports the two zero-parameter baselines (repeat the
last lap, take the five-lap median): a model that does not clearly beat them has not learned
anything a fan could not do with a stopwatch.

## Results

<!-- RESULTS:BEGIN -->
**Season-round split, 2018-2025 FastF1 tier** — train ≤ 2024 · validation 2025 rounds ≤ 12 · test 2025 rounds > 12. Rows: train 127861, validation 10248, test 10994. Data: fastf1_timing.

| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | Laps within 0.5 s | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.357 | 0.645 | 0.822 | 80.6% | — | — | — | — | — |
| rolling_median_5 | 0.394 | 0.675 | 0.889 | 76.9% | — | — | — | — | — |
| ridge | 0.392 | 0.622 | 0.801 | 74.8% | — | 22.7 | 2478 | 803 | — |
| **hist_gradient_boosting** | 0.316 | 0.569 | 0.684 | 83.8% | — | 49.9 | 2887 | 830 | — |
| FFR-M | 0.350 | 0.593 | 0.729 | 80.6% | 0.862 | 2234.1 | 2868 | 1421 | 3.37 |
| FFR-S | 0.350 | 0.590 | 0.729 | 80.5% | 0.858 | 390.3 | 2024 | 636 | 1.88 |
| FFR-L | 0.348 | 0.595 | 0.731 | 80.5% | 0.861 | 9985.4 | 4154 | 2778 | 8.24 |
| FFR-M-groups-coarse | 0.350 | 0.598 | 0.740 | 80.4% | 0.854 | 2088.2 | 2811 | 1417 | 3.33 |
| FFR-M-groups-fine | 0.351 | 0.593 | 0.740 | 80.1% | 0.863 | 2190.8 | 2980 | 1429 | 3.37 |
| FFR-M minus historical_numeric | 0.351 | 0.595 | 0.739 | 80.2% | 0.860 | 2208.3 | 2816 | 1421 | 3.33 |
| FFR-M minus temporal_numeric | 0.375 | 0.623 | 0.799 | 77.8% | 0.860 | 1968.1 | 2814 | 1420 | 3.22 |
| FFR-M minus static_categorical | 0.352 | 0.595 | 0.736 | 80.1% | 0.852 | 2218.5 | 2714 | 1420 | 2.94 |

Full table with validation metrics, interval widths and latency: `reports/f1_2025h2/summary.md`. Error by circuit, constructor, compound, tyre age, conditions, race phase and position: `reports/f1_2025h2/breakdowns.md`.

**Circuit holdout (Monza)** — every season of **Italian Grand Prix** held out; train ≤ 2024, validation 2025. Rows: train 123362, validation 20404, test 6133. Data: fastf1_timing.

| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | Laps within 0.5 s | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.346 | 0.615 | 0.809 | 81.0% | — | — | — | — | — |
| rolling_median_5 | 0.356 | 0.572 | 0.806 | 78.7% | — | — | — | — | — |
| ridge | 0.345 | 0.552 | 0.735 | 80.2% | — | 23.4 | 2575 | 830 | — |
| **hist_gradient_boosting** | 0.298 | 0.512 | 0.664 | 84.7% | — | 50.4 | 2921 | 854 | — |
| FFR-M | 0.347 | 0.538 | 0.694 | 80.0% | 0.838 | 2237.8 | 2761 | 1371 | 2.72 |
| FFR-S | 0.350 | 0.540 | 0.707 | 79.6% | 0.830 | 369.6 | 1996 | 614 | 1.24 |

Full table with validation metrics, interval widths and latency: `reports/holdout_monza/summary.md`.

**2026 domain shift, no retraining** — train ≤ 2024 · validation 2025 · test 2026. Rows: train 127861, validation 21242, test 11445. Data: fastf1_timing.

| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | Laps within 0.5 s | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.472 | 0.858 | 1.129 | 72.7% | — | — | — | — | — |
| rolling_median_5 | 0.501 | 0.882 | 1.168 | 69.3% | — | — | — | — | — |
| ridge | 0.457 | 0.778 | 1.003 | 71.5% | — | 25.1 | 2575 | 867 | — |
| **hist_gradient_boosting** | 0.432 | 0.755 | 0.972 | 74.5% | — | 51.5 | 2960 | 891 | — |
| FFR-M | 0.433 | 0.768 | 0.957 | 74.4% | 0.769 | 2233.3 | 2814 | 1421 | 3.39 |
| FFR-S | 0.434 | 0.767 | 0.970 | 74.3% | 0.768 | 390.6 | 1995 | 636 | 1.91 |

Full table with validation metrics, interval widths and latency: `reports/domain_shift_2026/summary.md`.
<!-- RESULTS:END -->

Every number above is reproducible from `scripts/run_experiments.py` on FastF1 data. Synthetic
fixture numbers are never reported as Formula 1 results.

**Protocol notes that matter when reading the tables.**

- Every row is a single run with seed 42. Differences of a few thousandths of a second between
  FFR variants (depth, group ladders, ablations that "change nothing") are within what a
  different seed could produce and are not claimed as effects; `reports/<name>/seeds.md`
  records the seed spread where it has been measured.
- The learned baselines (ridge, gradient-boosted trees) report validation metrics from a
  train-only fit and test metrics from a refit on train + validation, the usual practice for
  models with no early stopping. FFR is trained on the training seasons only and uses the
  validation rounds to calibrate its interval. The asymmetry favours the baselines slightly;
  `reports/f1_2025h2/generalization.md` shows the train-only tree at 0.316 s vs 0.317 s.
- "Forward-Forward" here means greedy layer-wise training with a local ordinal-goodness
  objective and no gradient flowing between layers. There is no positive/negative data pass as
  in Hinton's original formulation; `docs/TRAINING_AND_RESEARCH.md` spells out the difference.
- Lap-validity rules are versioned (`lap_validity_version` in every metrics file). Version 2
  excludes the restart lap after a red flag. Every table above was produced under version 2;
  the legacy-tier extension is being regenerated and returns to the tables when it finishes.

**What the numbers say so far** (2018-2024 training, 128k clean laps; test = 2025 rounds 13-24; lap-validity rules v2):

- Gradient-boosted trees are the most accurate model and train in under a minute. Forward-Forward
  regression does not beat them on this task (0.316 s vs 0.348-0.350 s test MAE).
- FFR beats the linear and rolling-median baselines and edges the naive previous-lap baseline by
  0.007-0.009 s. That gap is real but small: repeating the last lap already gets 80.6% of laps
  within half a second, FFR-M 80.6%, the tree 83.8%.
- Depth does not pay: FFR-S, FFR-M and FFR-L land at 0.350, 0.350 and 0.348 s for 6.5, 37 and
  166 minutes of training. The differences are inside seed noise (see the protocol notes).
- Interval calibration works: the 80% intervals cover 85-86% of test laps.
- Memory: FFR-S trains within 636 MB of traced allocations, below the tree's 830 MB, at equal
  accuracy to FFR-M; FFR-M and FFR-L need more, not less. The training-memory advantage argued
  for Forward-Forward appears only for the smallest network, and it does not come with an
  accuracy advantage.
- Ablations: removing the temporal pace features costs 0.025 s MAE; removing historical priors
  or driver/team/circuit identity changes nothing measurable. Recent pace carries the signal.
- Group-ladder variants (4/8/16/32, 8/16/32/64, 16/32/64/64) are indistinguishable.
- **Memorisation check.** Every model's error on its own training laps (FFR about 0.46 s, tree
  0.41 s) is higher than on validation (0.43 s) and test (0.35 s) because the training seasons
  contain more disrupted laps; the 2025 test rounds are the cleanest laps in the data. A negative
  train-to-test gap rules out gross overfitting but does not by itself prove generalisation;
  the circuit holdout and the 2026 shift below are the real tests. Train, validation and test
  metrics are recorded for every run, and `reports/f1_2025h2/generalization.md` has the
  per-season table.
- **Unseen circuit (every Italian Grand Prix held out).** With the red-flag restart lap excluded,
  Monza is no harder than the season split: trees 0.298 s, ridge 0.345 s, previous lap 0.346 s,
  FFR-M 0.347 s, FFR-S 0.350 s, RMSE 0.51-0.62 s for every model. FFR no longer loses more than
  the tree on an unseen circuit; it simply ties the naive baseline there. Under the version 1
  rules the same experiment reported RMSE of 2.5-3.1 s and FFR-M at 0.558 s, all from the
  handful of restart laps.
- **2026 domain shift (new regulations, model trained through 2024, never retrained).** Every
  model degrades by about 0.08 s MAE: trees 0.432 s, FFR-M 0.433 s, FFR-S 0.434 s, ridge
  0.457 s, previous lap 0.472 s. FFR-M has the best p90 (0.96 s) and degrades no worse than the
  tree, but its 80% intervals, calibrated on 2025, cover only 77% of 2026 laps: the shift is
  visible in calibration before it is visible in MAE.
- **Training extended to 2000 with the legacy tier (same 2025 test rows).** Being regenerated
  under the version 2 rules; the version 1 result was a small gain for every learned model
  (ridge 0.393 → 0.372 s, trees 0.317 → 0.314 s, FFR-M 0.352 → 0.347 s, FFR-S 0.357 → 0.342 s)
  at three to five times the training cost.

## Architecture

```text
FastF1 (2018→) · Jolpica/Ergast (2000-2017) · local CSV
        │
        ▼
lap-state flags ──► valid racing laps ──► segments of consecutive clean laps
        │
        ▼
leakage-safe features: dynamic state · temporal pace (relative) · historical priors (earlier events)
        │
        ▼
chronological split (season-forward, season-round, circuit holdout, 2026 domain shift)
        │
        ├─► baselines: previous lap · rolling-5 median · ridge · gradient-boosted trees
        │
        └─► RaceShift FFR: local Forward-Forward layers ─► ridge readout ─► forecast + 80% interval
                │
                ▼
        artifact (weights, preprocessor, contract, metrics) ─► FastAPI (localhost) ─► React UI
```

**RaceShift FFR.** Each layer is trained on its own ordinal-goodness objective with an
explicit local Adam update. Hidden units are partitioned into ordered target groups (8 → 16 →
32 → 64 for FFR-M); layer *k* trains on the frozen, normalised output of layer *k−1*. A
closed-form ridge readout over all layers' goodness vectors and local predictions produces
the residual forecast; the 80% interval is calibrated on validation residuals and widened
by cross-layer disagreement. No `Tensor.backward()`, no `autograd.grad()`, and a test fails if
either appears in the model source.

## Methodology

- **Target.** Residual of lap N+1 against the rolling five-lap median at lap N; forecast =
  baseline + residual. Training winsorizes the residual to ±6 s; evaluation never does.
- **Valid laps.** Pit-in, pit-out, safety-car, VSC, red-flag, red-flag restart, deleted and
  inaccurate laps are never rows or targets, and every lag or rolling statistic is scoped to
  the current run of consecutive clean laps, so a pit stop resets the temporal context. The
  rule set is versioned (`lap_validity_version` in every metrics file).
- **Relative pace features.** Lap-time-scale inputs are relative to the current rolling pace
  so they transfer across circuits; the one absolute anchor is the rolling median itself.
- **Historical priors.** Medians of per-event medians from strictly earlier events (driver ×
  circuit, team × circuit, compound × circuit, matched weather bins, …), relative to current pace.
- **Leakage tests.** A perturbation test changes everything after a cutoff lap and asserts no
  feature at or before it moves; another asserts priors ignore the current event.
- **Splits.** Season-round (train 2018-2024, validation early 2025, test late 2025), circuit
  holdout, train ≤ 2024 → 2026 domain shift without retraining, and a 2000-2024 legacy
  training extension on the same test rows.
- **Baselines first.** Previous lap, rolling-five median, ridge, gradient-boosted trees on the
  same table, split, sparse-feature filter and clipped target (learned baselines refit on
  train + validation for the test score; see the protocol notes under Results).
- **Local learning, tested.** Each layer's analytic gradient is checked against finite
  differences, and a test asserts that a layer's update does not change when the weights of
  any later layer change: nothing flows backwards between layers.
- **Resources.** Training wall time, peak RSS, traced peak, single-row latency, artifact bytes.

Details: `docs/FEATURE_CONTRACT.md`, `docs/TRAINING_AND_RESEARCH.md`, `RACESHIFT_MASTER_SPEC.md`.

## Data: every team, every round, two tiers

| Tier | Seasons | Source | What a lap carries |
| --- | --- | --- | --- |
| `fastf1_timing` | 2018 → today | FastF1 live-timing archive | lap and sector times, tyre compound/age/stint, position, track status, pit markers, weather |
| `legacy_timing` | 2000 → 2017 | Jolpica (Ergast schema) | lap time, position, constructor, pit stops from 2011; no sectors, tyres, track status or weather |

Timing data older than 2018 exists only in the Ergast schema, so the legacy tier is
deliberately thinner: missing columns stay missing, lap validity uses a documented heuristic
(opening lap, recorded pit laps and any lap slower than 1.12 × the driver's race median are
excluded), and every row is tagged with its tier so the model and the reports can tell them
apart. Headline results use the FastF1 tier; the legacy tier is evaluated as a training-set
extension on the same 2025 test split.

```bash
scripts/full_pipeline.sh fastf1 build-fastf1 experiments        # 2018→today, every round, main matrix + holdouts
scripts/full_pipeline.sh legacy build-all legacy-experiments    # 2000→2017 extension on the same test split
```

Both collectors are resumable and stay under the public API budgets (FastF1 and Jolpica each
allow about 500 requests per hour), so a full collection takes several hours unattended. The
Colab notebook `notebooks/RaceShift_FFR_Colab.ipynb` runs the same pipeline with data in
Drive. Copy any finished artifact folder into `artifacts/` and the UI lists it.

## Export a model

```bash
python scripts/export_model.py artifacts/f1_2025h2_ffr-m --verify-input data/imports/f1_2025_season.parquet
```

Writes `artifacts/<name>/export/` with the Forward-Forward core as ONNX (verified against the
NumPy implementation with onnxruntime before it is saved), the fitted preprocessor as plain
JSON with a pure-NumPy implementation (`raceshift.models.export.apply_preprocessor_spec`, no
pickle needed), a generated model card (data, split, metrics, resources, limitations,
inference snippet) and an export manifest, plus `exports/<name>.zip` bundling the whole
artifact. The same bundle is served by `GET /api/models/{id}/export` (staged under
`exports/<name>/`, never inside the tracked artifact) and linked from the Models page.

What is in git for the real FFR-M artifact: the NumPy weights, preprocessor, feature contract,
metrics, test predictions, the JSON preprocessor spec, the model card and the export manifest.
The `.onnx` file itself is not committed (`*.onnx` is ignored); the command above regenerates
and re-verifies it in a few seconds.

## Local API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health`, `/api/runtime`, `/api/setup` | Liveness, runtime versions and offline-local mode, setup checklist |
| GET | `/api/models`, `/api/datasets`, `/api/experiments` | Artifacts and baselines, local imports and sources, every `metrics.json` |
| GET | `/api/imports/{file}/summary` | Rows, seasons, drivers and latest session of one import |
| GET | `/api/models/{id}/export` | Zip bundle of an artifact: weights, preprocessor, ONNX core, JSON preprocessor spec, model card |
| POST | `/api/import` | Upload a CSV/Parquet lap table (200 MB, 2M rows, 250 columns) into `data/imports/` |
| POST | `/api/forecast/latest` | `{file, driver?, artifact?}` → next lap, 80% interval, input and historical context |
| POST | `/api/forecast/backtest` | `{file, driver?, artifact?, laps?}` → predicted vs actual for the driver's last completed laps, MAE, naive-baseline MAE, interval coverage |
| GET | `/api/reports`, `/api/reports/drivers` | Committed breakdown reports (MAE by circuit, team, compound, ...) and per-driver test error of every artifact |

Localhost only, no credentials, path-restricted file access. See `apps/api/README.md` and
`docs/SECURITY.md`.

## Project map

```text
apps/web/                  React/Vite UI (Overview, Forecast + backtest, Compare Drivers, Experiments, Datasets, Models; Telemetry/Strategy are marked Planned)
apps/api/                  local FastAPI backend
src/raceshift/data/        schema, provenance, splits, FastF1 / Jolpica-Ergast / OpenF1 adapters
src/raceshift/features/    lap-state flags, segments, leakage-safe features, selection
src/raceshift/models/      Forward-Forward regressor and artifact runtime
src/raceshift/train/       metrics and resource measurement
src/raceshift/foundation/  walk-forward examples for frozen time-series foundation models
scripts/                   collectors (FastF1, Jolpica), training, baselines, experiment runner, full_pipeline.sh
configs/                   FFR-S/M/L, group-ladder ablations, baseline settings
reports/                   measured experiment tables (committed)
notebooks/                 Colab workflow
data/imports/              local datasets (2025 season FastF1 timing and the synthetic fixture included)
artifacts/                 model artifacts (real FFR-M and legacy FFR-S, baseline metrics and the synthetic demo committed)
docs/                      feature contract, research standard, sources, security, local setup walkthrough
tests/                     Python tests (see `npm run test:py`)
```

## Status

Foundation and real-data pipeline complete; Forward-Forward is evaluated against baselines on
real races with chronological splits. Not yet done: wet-weather analysis in isolation,
similarity-retrieval priors, endurance-series adapters, corner-telemetry Forward-Forward.
