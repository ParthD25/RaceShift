# Local Setup

## Required software

- Python 3.11+
- Node.js 20+
- npm
- Git, optional but recommended
- Google account only if you want to train in Colab

## Install

From the RaceShift root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"
npm install
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Launch

```bash
npm run dev
```

`concurrently` starts the local FastAPI server and the React/Vite development server together.

## Local data connection

Copy CSV or Parquet into `data/imports/`, or upload it from the Datasets page (the UI calls
`POST /api/import`). The backend resolves filenames strictly inside this directory and rejects
path traversal.

Required columns: `season, event, session, driver, lap_number, lap_time_s`. The full schema is in
`docs/FEATURE_CONTRACT.md`.

Then run a forecast from the Forecast page, or from the shell:

```bash
curl -s -X POST http://127.0.0.1:8000/api/forecast/latest \
  -H 'Content-Type: application/json' \
  -d '{"file": "my_2025_race_laps.parquet", "driver": "VER"}'

# equivalent query-string alias
curl -s "http://127.0.0.1:8000/api/forecast?file=my_2025_race_laps.parquet&driver=VER"
```

The forecast uses the chronologically latest session in the file. Omit `driver` to use the driver
with the most completed laps in that session.

## Real model artifact

After Colab training, download the artifact folder and copy it into `artifacts/`, for example:

```text
artifacts/raceshift_ffr_2025test/
```

Every complete artifact is listed by `GET /api/models` and selectable on the Forecast page (or via
the `artifact` field of the forecast request). The packaged `artifacts/raceshift_ffr_demo/` stays
labelled synthetic; do not overwrite it with a real model.

An artifact contains:

```text
model_weights.npz
model_config.json
training_history.json
preprocessor.joblib
feature_contract.json
metrics.json
test_predictions.csv   (written by train_ffr.py, not required at load time)
```

`metrics.json` carries `data_source` and `is_synthetic`, which the UI uses to badge results.

## Run the experiment matrix

```bash
python scripts/fetch_fastf1_seasons.py --years 2022-2025 --session R --events Bahrain Silverstone Monza
python scripts/build_lap_dataset.py --input data/raw/fastf1 --output data/processed/next_lap.parquet   # light table
python - <<'PY'
import glob, pandas as pd
pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/raw/fastf1/*.parquet'))]).to_parquet('data/processed/f1_laps.parquet', index=False)
PY
python scripts/run_experiments.py --input data/processed/f1_laps.parquet --name f1_2025 \
  --train-end 2023 --val-year 2024 --test-year 2025 --ffr configs/ffr_small.json configs/ffr_production.json
```

Results land in `reports/f1_2025/summary.md` and every run's `metrics.json` is listed by the API.

## Verify the install

```bash
npm run test:py        # 39 Python tests: leakage, availability, adjacency, splits, no-backprop policy, artifact, API
npm run build          # TypeScript check + Vite production build
```

## Optional historical data collection

FastF1:

```bash
python scripts/fetch_fastf1.py --help
```

OpenF1:

```bash
python scripts/fetch_openf1.py --help
```

Historical data should be cached locally and not committed to Git.
