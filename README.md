# RaceShift

Local-first Formula 1 **next-lap forecasting**. RaceShift predicts a driver's lap N+1 from
information legitimately known at the end of lap N: tyres, stint, weather, wind, traffic, sectors,
recent pace, telemetry summaries and prior-race context. The primary trainable model is a deep,
multi-layer **Forward-Forward regressor** with local layer updates and **no global backpropagation**.

**Status: foundation scaffold (v0.4).** The local product, training pipeline, baselines and tests
work end to end on a synthetic fixture. Real Formula 1 training has not been run yet, so this
repository makes **no Formula 1 accuracy claims**. Every number the UI shows is labelled
*Fixture visual*, *Synthetic model* or *Real artifact*.

## What is included

- React + Vite workspace UI wired to a localhost FastAPI backend
- Leakage-safe full-context feature engineering (driver, constructor, circuit, tyres, stint,
  weather, wind as sin/cos, traffic, sectors, five-lap flattened history, prior-event matched-condition pace)
- Chronological season-forward splits; the headline metric is always a future season
- RaceShift FFR: local Forward-Forward layers with ordinal goodness groups, explicit local Adam,
  closed-form ridge readout, 80% validation-calibrated interval plus layer disagreement
- Required baselines on the same table and split: previous lap, rolling-five median, ridge,
  gradient-boosted trees
- FastF1 and OpenF1 collectors, Hugging Face / IMSA source registry, Colab notebook
- Synthetic demo dataset and demo artifact so the app runs before any download
- 26 Python tests covering leakage rules, splits, the no-backprop policy, artifact forecasting and the API

## Quick start

Python 3.11+ and Node.js 20+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[local]"
npm install
npm run dev
```

- UI: <http://127.0.0.1:5173>
- API: <http://127.0.0.1:8000/api/health>

Open **Forecast**, keep `synthetic_fixture.csv` and `raceshift_ffr_demo` selected, and press
**Run forecast**. The result is badged *Synthetic model* because the packaged artifact was trained
on `data/imports/synthetic_fixture.csv`.

Verify the install:

```bash
npm run test:py     # pytest, 26 tests
npm run build       # TypeScript check + Vite build
```

## Connect your own data

Drop a `.csv` or `.parquet` lap table into `data/imports/` or upload it on the **Datasets** page.
Required columns: `season, event, session, driver, lap_number, lap_time_s`; the full schema is in
`docs/FEATURE_CONTRACT.md`. The API resolves filenames strictly inside `data/imports/`.

## Train the real model

Use Google Colab and `notebooks/RaceShift_FFR_Colab.ipynb`. It clones this repository, keeps data
and artifacts in Google Drive, runs the synthetic smoke test, collects 2022-2025 races at Bahrain,
Silverstone and Monza, runs the baselines first, then trains the production ladder:

```text
input -> 512 -> 384 -> 256 -> 192 hidden nodes, 8/16/32/64 ordinal groups
train <= 2023, validate 2024, test 2025
```

The same commands work locally:

```bash
python scripts/train_baselines.py --input data/processed/f1_laps.parquet --output artifacts/baselines_2025test
python scripts/train_ffr.py --input data/processed/f1_laps.parquet --config configs/ffr_production.json \
  --output artifacts/raceshift_ffr_2025test --train-end 2023 --val-year 2024 --test-year 2025
```

Copy a finished artifact folder into `artifacts/` and the UI lists it as *Real artifact*.

The packaged demo (`artifacts/raceshift_ffr_demo`) uses the small `configs/ffr_demo.json` ladder
(172 encoded inputs -> 128 -> 96 -> 64 -> 48, 8 groups per layer) so the app runs instantly.

## Model rule

RaceShift FFR performs no end-to-end backpropagation. Each layer learns from its own local
ordinal-goodness objective; the implementation computes that derivative explicitly and updates only
that layer. A test fails if `.backward(` or `autograd.grad(` appears in the model source.

Forward-Forward is a research direction, not a proven replacement for backpropagation. On the
synthetic fixture the tree baseline beats the demo FFR artifact, and that is reported as-is. The
real question is answered only on unseen Formula 1 seasons.

## Local API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health`, `/api/runtime`, `/api/setup` | Liveness, runtime versions and offline-local mode, setup checklist |
| GET | `/api/models`, `/api/datasets`, `/api/experiments` | Artifacts and baselines, local imports and sources, every `metrics.json` |
| GET | `/api/imports/{file}/summary` | Rows, seasons, drivers and latest session of one import |
| POST | `/api/import` | Upload a CSV/Parquet lap table into `data/imports/` |
| POST | `/api/forecast/latest` | `{file, driver?, artifact?}` -> next-lap forecast with 80% interval |

Details in `apps/api/README.md`.

## Project map

```text
apps/web/                  React/Vite UI (pages read the local API; fixture visuals are badged)
apps/api/                  local FastAPI backend
src/raceshift/data/        schema, provenance, chronological splits, FastF1/OpenF1 adapters
src/raceshift/features/    leakage-safe full-context features and shared preprocessing
src/raceshift/models/      Forward-Forward regressor and artifact runtime
src/raceshift/train/       shared regression and interval metrics
src/raceshift/foundation/  walk-forward examples for frozen time-series foundation models
scripts/                   collection, training, baselines and demo utilities
configs/                   demo, production and Colab FFR ladders; baseline settings
notebooks/                 Colab training workflow
data/imports/              user-connected local datasets (synthetic fixture included)
artifacts/                 trained models and metrics (only the synthetic demo is committed)
design/                    concept renders (not screenshots; see design/README.md)
docs/                      setup, feature contract, research, sources, security
tests/                     leakage, split, policy, artifact and API tests
```

## Read next

1. `RACESHIFT_MASTER_SPEC.md`
2. `docs/LOCAL_SETUP.md`
3. `docs/FEATURE_CONTRACT.md`
4. `docs/TRAINING_AND_RESEARCH.md`
5. `docs/DATA_AND_MODEL_FINDINGS.md`
6. `docs/SECURITY.md`
