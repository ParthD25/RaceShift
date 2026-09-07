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
npm run test:py     # 39 tests: leakage, feature availability, lap adjacency, splits, no-backprop policy, artifact, API
npm run build       # TypeScript check + Vite build
```

## Results

<!-- RESULTS:BEGIN -->
Real Formula 1 results are being generated; see `reports/` for every measured table.
<!-- RESULTS:END -->

Every number above is reproducible from `scripts/run_experiments.py` on FastF1 data. Synthetic
fixture numbers are never reported as Formula 1 results.

## Architecture

```text
FastF1 / OpenF1 / local CSV
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
- **Splits.** Season-forward (train ≤ 2024, test 2025), season-round (early/late 2025),
  circuit holdout, and train ≤ 2025 → 2026 domain shift.
- **Baselines first.** Previous lap, rolling-five median, ridge, gradient-boosted trees on the
  same table, split, sparse-feature filter and clipped target.
- **Resources.** Training wall time, peak RSS, traced peak, single-row latency, artifact bytes.

Details: `docs/FEATURE_CONTRACT.md`, `docs/TRAINING_AND_RESEARCH.md`, `RACESHIFT_MASTER_SPEC.md`.

## Train on real data

Locally:

```bash
python scripts/fetch_fastf1_seasons.py --years 2022-2025 --session R \
  --events Bahrain Jeddah Suzuka Monaco Silverstone Spa Monza "Marina Bay" Austin "Mexico City" "São Paulo" "Yas Island"
python -c "import glob,pandas as pd; pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/raw/fastf1/*.parquet'))]).to_parquet('data/processed/f1_laps.parquet', index=False)"
python scripts/run_experiments.py --input data/processed/f1_laps.parquet --name f1_2025h2 \
  --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12 \
  --ffr configs/ffr_small.json configs/ffr_production.json configs/ffr_colab_large.json
```

Or in Google Colab with `notebooks/RaceShift_FFR_Colab.ipynb`, which clones this repository,
keeps data and artifacts in Drive, and runs the same matrix. Copy any finished artifact folder
into `artifacts/` and the UI lists it.

## Local API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health`, `/api/runtime`, `/api/setup` | Liveness, runtime versions and offline-local mode, setup checklist |
| GET | `/api/models`, `/api/datasets`, `/api/experiments` | Artifacts and baselines, local imports and sources, every `metrics.json` |
| GET | `/api/imports/{file}/summary` | Rows, seasons, drivers and latest session of one import |
| POST | `/api/import` | Upload a CSV/Parquet lap table (200 MB, 2M rows) into `data/imports/` |
| POST | `/api/forecast/latest` | `{file, driver?, artifact?}` → next lap, 80% interval, input and historical context |

Localhost only, no credentials, path-restricted file access. See `apps/api/README.md` and
`docs/SECURITY.md`.

## Project map

```text
apps/web/                  React/Vite UI (live API data; fixture visuals are badged)
apps/api/                  local FastAPI backend
src/raceshift/data/        schema, provenance, splits, FastF1/OpenF1 adapters
src/raceshift/features/    lap-state flags, segments, leakage-safe features, selection
src/raceshift/models/      Forward-Forward regressor and artifact runtime
src/raceshift/train/       metrics and resource measurement
src/raceshift/foundation/  walk-forward examples for frozen time-series foundation models
scripts/                   collection, training, baselines, experiment runner
configs/                   FFR-S/M/L, group-ladder ablations, baseline settings
reports/                   measured experiment tables (committed)
notebooks/                 Colab workflow
data/imports/              local datasets (synthetic fixture included)
artifacts/                 model artifacts (synthetic demo committed; real runs listed when present)
docs/                      feature contract, research standard, sources, security
tests/                     39 tests
```

## Status

Foundation and real-data pipeline complete; Forward-Forward is evaluated against baselines on
real races with chronological splits. Not yet done: wet-weather analysis in isolation,
similarity-retrieval priors, endurance-series adapters, corner-telemetry Forward-Forward.
