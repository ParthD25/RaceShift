# RaceShift local API

Localhost-only FastAPI service used by the React UI. It reads local files and local model
artifacts. It holds no credentials, never trains, and reports `live_connected: false` until an
optional live adapter exists.

```bash
pip install -e ".[local]"          # installs the raceshift package plus API extras
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

`npm run dev` at the project root starts this server together with the Vite UI.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness, version, whether the demo artifact is complete |
| GET | `/api/runtime` | Python/package versions, `data_mode`, `live_connected`, default artifact provenance |
| GET | `/api/setup` | Setup checklist: dependencies, demo artifact, real artifacts, local imports |
| GET | `/api/models` | Every artifact under `artifacts/` (ready flag, architecture, metrics, provenance) plus required baselines |
| GET | `/api/datasets` | `dataset_manifest.json` sources, files in `data/imports/`, processed tables |
| GET | `/api/imports/{file}/summary` | Rows, columns, seasons, drivers and the latest session of one import |
| POST | `/api/import` | Multipart upload of a `.csv`/`.parquet` lap table into `data/imports/` (200 MB and 2,000,000-row caps, `?overwrite=true` to replace) |
| POST | `/api/forecast/latest` | JSON `{file, driver?, artifact?}` → next-lap forecast with 80% interval, `context` (tyre, weather, gaps at the end of the lap) and `historical_context` (medians from earlier events) |
| GET | `/api/forecast` | Query-string alias of the forecast endpoint for `curl` |
| GET | `/api/experiments` | Every `metrics.json` under `artifacts/`, baseline reports flattened per model |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/api/forecast/latest \
  -H 'Content-Type: application/json' \
  -d '{"file": "synthetic_fixture.csv", "driver": "BBB"}'
```

## Security posture

- Binds `127.0.0.1`; CORS allows only the Vite origins on port 5173.
- Import filenames and artifact ids are resolved strictly inside `data/imports/` and `artifacts/`;
  traversal, absolute paths and symlink escapes are rejected.
- Uploads are limited to CSV/Parquet, 200 MB and 2,000,000 rows, sanitised to a basename, and parsed before being kept; partial files are removed on any failure.
- Responses contain relative paths only and never echo internal exception text.
- Every forecast response carries `data_source` and `is_synthetic` so the UI can label it.
