# RaceShift v0.4

RaceShift is a local-first motorsport ML project that predicts a driver's **next lap** from information available through the current completed lap. The primary trainable model is a deep, multi-layer Forward-Forward regressor with local layer updates and **no global backpropagation**.

## What is included

- React + Vite local UI
- FastAPI localhost API
- full F1 feature contract: driver, constructor, circuit, tyres, stint, weather, wind, traffic, sectors, recent pace and historical matched-condition pace
- four-layer production Forward-Forward architecture: `512 -> 384 -> 256 -> 192` hidden nodes
- larger Colab research ladder: `1024 -> 768 -> 512 -> 384 -> 256`
- 8/16/32/64 ordinal goodness groups for the production model
- manual local Adam updates, no `Tensor.backward()` or `autograd.grad()`
- closed-form ridge readout
- 80% residual-calibrated prediction interval plus layer disagreement
- FastF1 and OpenF1 collectors
- Hugging Face / Kaggle / endurance source registry
- synthetic local demo dataset and demo model artifact
- Colab notebook for real training
- tests for leakage, deep architecture and no-backprop policy

## Quick start

### 1. Install Python

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows PowerShell
pip install -e ".[local]"
```

### 2. Install Node dependencies

Install Node.js 20+ and npm, then from the project root:

```bash
npm install
```

The project uses an npm workspace, so this installs the root dev tooling and the React app dependencies.

### 3. Start RaceShift

```bash
npm run dev
```

This launches:

- UI: `http://127.0.0.1:5173`
- local API: `http://127.0.0.1:8000`
- API health: `http://127.0.0.1:8000/api/health`

The included synthetic dataset and demo artifact let the app run before you train on real F1 data.

## Connect your own data

Place a `.csv` or `.parquet` file in:

```text
data/imports/
```

The canonical schema and feature requirements are documented in `docs/FEATURE_CONTRACT.md`.

The local API only resolves filenames inside `data/imports/`. It does not expose arbitrary filesystem paths.

## Train the real model

For real training, use Google Colab and the notebook:

```text
notebooks/RaceShift_FFR_Colab.ipynb
```

The recommended first real split is:

```text
train:       2019-2023
validation:  2024
test:        2025
```

Then reserve 2026 for domain-shift evaluation when enough data is available.

The production training command is:

```bash
python scripts/train_ffr.py \
  --input data/processed/f1_laps.parquet \
  --config configs/ffr_production.json \
  --output artifacts/raceshift_ffr \
  --train-end 2023 \
  --val-year 2024 \
  --test-year 2025
```

## Important model rule

RaceShift FFR does not perform end-to-end backpropagation. Each layer learns independently from its own local ordinal-goodness objective. The implementation computes the local derivative explicitly and updates only that layer's parameters.

Forward-Forward is a research direction, not a proven universal replacement for backpropagation. RaceShift must be judged against simple and strong baselines on unseen future seasons.

## Project map

```text
apps/web/                  React/Vite UI
apps/api/                  local FastAPI backend
src/raceshift/data/        source adapters and split rules
src/raceshift/features/    leakage-safe full-context feature engineering
src/raceshift/models/      Forward-Forward model + artifact runtime
scripts/                   collection, training and demo utilities
configs/                   production and Colab FFR architectures
notebooks/                 Colab training workflow
data/imports/              user-connected local datasets
artifacts/                 trained models and metrics
docs/                      setup, feature, research and source documentation
tests/                     leakage and training-policy tests
```

## Read next

1. `RACESHIFT_MASTER_SPEC.md`
2. `docs/LOCAL_SETUP.md`
3. `docs/FEATURE_CONTRACT.md`
4. `docs/TRAINING_AND_RESEARCH.md`
5. `docs/DATA_AND_MODEL_FINDINGS.md`
6. `docs/SECURITY.md`
