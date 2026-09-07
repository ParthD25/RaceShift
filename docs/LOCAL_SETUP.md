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

Copy CSV or Parquet into `data/imports/`. The backend intentionally prevents path traversal outside this directory.

Example:

```text
data/imports/my_2025_race_laps.parquet
```

Then request:

```text
GET http://127.0.0.1:8000/api/forecast?file=my_2025_race_laps.parquet&driver=VER
```

## Real model artifact

After Colab training, download the artifact folder and place it at:

```text
artifacts/raceshift_ffr_demo/
```

or change `DEFAULT_ARTIFACT` in `apps/api/main.py` to the new artifact directory.

An artifact contains:

```text
model_weights.npz
model_config.json
training_history.json
preprocessor.joblib
feature_contract.json
metrics.json
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
