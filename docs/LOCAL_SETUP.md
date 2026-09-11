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

`concurrently` starts the local FastAPI server (127.0.0.1:8000) and the React/Vite development
server (127.0.0.1:5173) together. If the API port is taken, the API prints `RaceShift API:
127.0.0.1:8000 is already in use` followed by two ports that are free at that moment and the
exact command to run; if only the web port is taken, Vite moves to the next free one and
prints the URL it chose. The API, the Vite proxy and the sidebar status all read the same
variables:

```bash
API_PORT=8010 WEB_PORT=5180 npm run dev
```

Expected output: `[API] Uvicorn running on http://127.0.0.1:8000` and `[WEB] Local:
http://127.0.0.1:5173/`. The Settings page shows a setup checklist with every dependency,
artifact and data file it found.

## Local data connection

The repository ships three tables in `data/imports/`: `f1_2025_season.parquet` (every 2025
race, FastF1 timing, 640 KB), `f1_2026_races.parquet` (the 13 races run so far in 2026, 417 KB,
never used to train any committed model) and `synthetic_fixture.csv` (generated engineering
data). The Forecast page selects the 2025 season by default, so the first **Run forecast**
predicts a real driver's next lap at the 2025 Abu Dhabi Grand Prix and backtests that driver's
last ten laps; pick the 2026 file to score the model on races it has never seen.

To add a race, fetch it (about 15 s per race, cached afterwards) or copy CSV/Parquet into
`data/imports/`, or upload it from the Datasets page (the UI calls `POST /api/import`). The
backend resolves filenames strictly inside this directory and rejects path traversal.

```bash
python scripts/fetch_fastf1.py --year 2025 --event "Abu Dhabi" --session R --output data/imports/abu_dhabi_2025.parquet
```

Required columns: `season, event, session, driver, lap_number, lap_time_s`, plus `event_date` or `round_number` so events can be ordered chronologically (historical priors must come from strictly earlier events; the import summary flags a table without either). Uploads missing a required column are rejected; out-of-range values (a negative lap time, 999 °C) are accepted but reported as warnings. The full schema is in
`docs/FEATURE_CONTRACT.md`.

Then run a forecast from the Forecast page, or from the shell:

```bash
curl -s -X POST http://127.0.0.1:8000/api/forecast/latest \
  -H 'Content-Type: application/json' \
  -d '{"file": "my_2025_race_laps.parquet", "driver": "VER"}'

# equivalent query-string alias
curl -s "http://127.0.0.1:8000/api/forecast?file=my_2025_race_laps.parquet&driver=VER"

# how did the model do on the laps that were actually driven? (predicted vs actual, last 10 lap pairs)
curl -s -X POST http://127.0.0.1:8000/api/forecast/backtest \
  -H 'Content-Type: application/json' \
  -d '{"file": "f1_2025_season.parquet", "driver": "VER", "laps": 10}'
```

The forecast uses the chronologically latest session in the file. Omit `driver` to use the
best-placed driver with the most completed laps in that session (the race winner when the
whole field finished). The response carries the lap's input context
(tyre, weather, gaps) and the historical priors from earlier events that fed the model.

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
pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/raw/fastf1/*.parquet'))]).to_parquet('data/processed/f1_laps_fastf1.parquet', index=False)
PY
python scripts/run_experiments.py --input data/processed/f1_laps_fastf1.parquet --name f1_2025 \
  --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12 --ffr configs/ffr_small.json configs/ffr_production.json
```

Results land in `reports/f1_2025h2/summary.md` and every run's `metrics.json` is listed by the API.

## Verify the install

```bash
npm run test:py        # Python tests: leakage, availability, lap rules, gradients, splits, artifact, API
npm run build          # TypeScript check + Vite production build
npm run demo:data && npm run demo:model   # synthetic fixture -> demo FFR artifact (about a minute)
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
