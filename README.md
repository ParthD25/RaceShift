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

UI at <http://127.0.0.1:5173>, API at <http://127.0.0.1:8000/api/health>. Open **Forecast**,
pick a dataset and an artifact, press **Run forecast**. The packaged demo artifact is trained on
a synthetic fixture and is badged *Synthetic model*; real artifacts are badged *Real artifact*.

```bash
npm run test:py     # 44 tests: leakage, feature availability, lap adjacency, splits, no-backprop policy, artifact, API
npm run build       # TypeScript check + Vite build
```

## Results

<!-- RESULTS:BEGIN -->
**Season-round split, 2018-2025 FastF1 tier** — train ≤ 2024 · validation 2025 rounds ≤ 12 · test 2025 rounds > 12. Rows: train 128071, validation 10248, test 10994. Data: fastf1_timing.

| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.357 | 0.645 | 0.822 | — | — | — | — | — |
| rolling_median_5 | 0.394 | 0.675 | 0.889 | — | — | — | — | — |
| ridge | 0.393 | 0.624 | 0.811 | — | 21.4 | 2478 | 804 | — |
| **hist_gradient_boosting** | 0.317 | 0.570 | 0.680 | — | 43.4 | 2888 | 830 | — |
| FFR-M | 0.352 | 0.595 | 0.740 | 0.857 | 2064.1 | 2973 | 1423 | 3.37 |
| FFR-S | 0.357 | 0.596 | 0.747 | 0.856 | 375.2 | 1990 | 637 | 1.88 |
| FFR-L | 0.349 | 0.595 | 0.735 | 0.858 | 9804.7 | 4324 | 2783 | 8.24 |
| FFR-M-groups-coarse | 0.352 | 0.598 | 0.749 | 0.852 | 2077.7 | 2972 | 1419 | 3.33 |
| FFR-M-groups-fine | 0.353 | 0.594 | 0.747 | 0.865 | 2154.2 | 2880 | 1431 | 3.37 |
| FFR-M minus historical_numeric | 0.353 | 0.596 | 0.748 | 0.855 | 2135.9 | 2807 | 1423 | 3.33 |
| FFR-M minus temporal_numeric | 0.375 | 0.623 | 0.798 | 0.858 | 1817.6 | 2867 | 1423 | 3.22 |
| FFR-M minus static_categorical | 0.353 | 0.596 | 0.743 | 0.851 | 2081.3 | 2715 | 1422 | 2.94 |

Full table with validation metrics, interval widths and latency: `reports/f1_2025h2/summary.md`.

**Circuit holdout (Monza)** — every season of **Italian Grand Prix** held out; train ≤ 2024, validation 2025. Rows: train 123559, validation 20404, test 6166. Data: fastf1_timing.

| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.509 | 2.793 | 0.826 | — | — | — | — | — |
| rolling_median_5 | 0.600 | 3.103 | 0.848 | — | — | — | — | — |
| ridge | 0.576 | 2.550 | 0.790 | — | 25.1 | 2494 | 831 | — |
| **hist_gradient_boosting** | 0.498 | 2.619 | 0.692 | — | 49.4 | 2915 | 855 | — |
| FFR-M | 0.558 | 2.783 | 0.726 | 0.837 | 2051.9 | 2767 | 1373 | 2.72 |
| FFR-S | 0.569 | 2.798 | 0.744 | 0.826 | 355.8 | 2087 | 615 | 1.24 |

Full table with validation metrics, interval widths and latency: `reports/holdout_monza/summary.md`.

**2026 domain shift, no retraining** — train ≤ 2024 · validation 2025 · test 2026. Rows: train 128071, validation 21242, test 11483. Data: fastf1_timing.

| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_lap | 0.556 | 2.250 | 1.151 | — | — | — | — | — |
| rolling_median_5 | 0.628 | 2.501 | 1.210 | — | — | — | — | — |
| ridge | 0.553 | 2.065 | 1.026 | — | 23.5 | 2583 | 868 | — |
| **hist_gradient_boosting** | 0.544 | 2.145 | 1.018 | — | 48.0 | 2970 | 892 | — |
| FFR-M | 0.545 | 2.238 | 0.993 | 0.766 | 2124.6 | 2876 | 1423 | 3.40 |
| FFR-S | 0.553 | 2.267 | 1.004 | 0.765 | 362.1 | 1996 | 637 | 1.91 |

Full table with validation metrics, interval widths and latency: `reports/domain_shift_2026/summary.md`.
<!-- RESULTS:END -->

Every number above is reproducible from `scripts/run_experiments.py` on FastF1 data. Synthetic
fixture numbers are never reported as Formula 1 results.

**What the numbers say so far** (2018-2024 training, 128k clean laps; test = 2025 rounds 13-24):

- Gradient-boosted trees are the most accurate model and train in under a minute. Forward-Forward
  regression does not beat them on this task.
- FFR beats the linear and rolling-median baselines and, at M and L depth, edges the naive
  previous-lap baseline; FFR-S ties it. Depth helps a little (0.357 → 0.352 → 0.349 s) at a
  large cost in training time (6 → 34 → 163 minutes).
- Interval calibration works: the 80% intervals cover 85-87% of test laps.
- Memory: FFR-S trains within 637 MB of traced allocations, below the tree's 830 MB, but is
  less accurate; FFR-M and FFR-L need more, not less. The training-memory advantage argued for
  Forward-Forward does not appear in this NumPy implementation at this scale.
- Ablations: removing the temporal pace features costs 0.02 s MAE; removing historical priors
  or driver/team/circuit identity changes nothing measurable. Recent pace carries the signal.
- Group-ladder variants (4/8/16/32, 8/16/32/64, 16/32/64/64) are indistinguishable.
- **No memorisation.** Every model's error on its own training laps (about 0.50 s) is higher
  than on validation (0.43 s) and test (0.35 s); the training seasons contain more disrupted
  laps. Train, validation and test metrics are recorded for every run, and
  `reports/f1_2025h2/generalization.md` has the per-season table.
- **Unseen circuit (every Italian Grand Prix held out).** Errors rise for every model and the
  ranking holds: trees 0.498 s, previous lap 0.509 s, FFR-M 0.558 s, FFR-S 0.569 s. FFR loses
  more than the tree when the circuit has never been seen. RMSE jumps to 2.5-3.1 s for every
  model because of red-flag stoppages in the 2020 and 2026 races: the laps around the stoppage
  pass the validity rules yet are 40-56 s off. They are 0.8% of test laps; without them FFR-M's
  RMSE is 0.50 s. Red-flag-adjacent laps are a documented gap in the lap-state rules.
- **2026 domain shift (new regulations, model trained through 2024, never retrained).** All
  models degrade by about 0.2 s MAE and the gap between them closes: trees 0.544 s, FFR-M
  0.545 s, previous lap 0.556 s, ridge 0.553 s. FFR-M has the best p90 (0.99 s) and degrades no
  worse than the tree, but its 80% intervals, calibrated on 2025, cover only 77% of 2026 laps:
  the shift is visible in calibration before it is visible in MAE.

The 2000-2024 legacy training extension is reported when it finishes; see `reports/`.

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
- **Valid laps.** Pit-in, pit-out, safety-car, VSC, red-flag, deleted and inaccurate laps are
  never rows or targets, and every lag or rolling statistic is scoped to the current run of
  consecutive clean laps, so a pit stop resets the temporal context.
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
  same table, split, sparse-feature filter and clipped target.
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

## Local API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health`, `/api/runtime`, `/api/setup` | Liveness, runtime versions and offline-local mode, setup checklist |
| GET | `/api/models`, `/api/datasets`, `/api/experiments` | Artifacts and baselines, local imports and sources, every `metrics.json` |
| GET | `/api/imports/{file}/summary` | Rows, seasons, drivers and latest session of one import |
| POST | `/api/import` | Upload a CSV/Parquet lap table (200 MB, 2M rows, 250 columns) into `data/imports/` |
| POST | `/api/forecast/latest` | `{file, driver?, artifact?}` → next lap, 80% interval, input and historical context |

Localhost only, no credentials, path-restricted file access. See `apps/api/README.md` and
`docs/SECURITY.md`.

## Project map

```text
apps/web/                  React/Vite UI (live API data; fixture visuals are badged)
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
data/imports/              local datasets (synthetic fixture included)
artifacts/                 model artifacts (synthetic demo committed; real runs listed when present)
docs/                      feature contract, research standard, sources, security
tests/                     44 tests
```

## Status

Foundation and real-data pipeline complete; Forward-Forward is evaluated against baselines on
real races with chronological splits. Not yet done: wet-weather analysis in isolation,
similarity-retrieval priors, endurance-series adapters, corner-telemetry Forward-Forward.
