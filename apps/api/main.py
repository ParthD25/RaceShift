"""RaceShift local API.

Localhost-only FastAPI service that the React/Vite UI talks to. It reads local files and
local model artifacts. It never holds credentials, never trains, and never claims a live
connection when it is serving fixture or historical data.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import threading
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from raceshift import __version__
from raceshift.data.provenance import infer_data_source, is_synthetic_source
from raceshift.data.schema import REQUIRED_FORECAST_COLUMNS
from raceshift.features.full_context import LAP_VALIDITY_VERSION
from raceshift.models.artifact import ARTIFACT_FILES, RaceShiftArtifact, blank_mask, inference_table_for, integer_season, missing_artifact_files

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
IMPORTS = DATA / "imports"
PROCESSED = DATA / "processed"
DEMO_ARTIFACT_ID = "raceshift_ffr_demo"           # synthetic fixture model, always present (CI smoke path)
REAL_ARTIFACT_ID = "f1_2025h2_ffr-m"             # committed real-data FFR-M; preferred default when complete


def default_artifact_id() -> str:
    """The real-data artifact when its files are all present, otherwise the synthetic demo."""
    return REAL_ARTIFACT_ID if not missing_artifact_files(ARTIFACTS / REAL_ARTIFACT_ID) else DEMO_ARTIFACT_ID
ALLOWED_SUFFIXES = {".csv", ".parquet"}
MAX_IMPORT_BYTES = 200 * 1024 * 1024
MAX_IMPORT_ROWS = 2_000_000
MAX_IMPORT_COLUMNS = 250
def _web_port() -> int:
    """WEB_PORT from the environment, falling back to Vite's default when unset or invalid."""
    raw = os.environ.get("WEB_PORT", "5173") or "5173"
    try:
        port = int(raw)
    except ValueError:
        return 5173
    return port if 1 <= port <= 65535 else 5173


_WEB_PORTS = sorted({5173, _web_port()})
UI_ORIGINS = [f"http://{host}:{port}" for port in _WEB_PORTS for host in ("localhost", "127.0.0.1")]
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

app = FastAPI(title="RaceShift Local API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=UI_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- helpers


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def _safe_child(base: Path, name: str, what: str) -> Path:
    """Resolve `name` strictly inside `base`. Rejects traversal, absolute paths and symlink escapes."""
    if not name or "\x00" in name or any(ord(ch) < 32 for ch in name) or name != Path(name).name or name in {".", ".."}:
        raise HTTPException(400, f"Invalid {what} name")
    candidate = (base / name).resolve()
    root = base.resolve()
    if root not in candidate.parents:
        raise HTTPException(400, f"Invalid {what} name")
    return candidate


def _safe_import_path(filename: str) -> Path:
    candidate = _safe_child(IMPORTS, filename, "import file")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Only CSV and Parquet are supported")
    if not candidate.is_file():
        raise HTTPException(404, f"Import file not found: {filename}")
    return candidate


def _safe_artifact_path(artifact_id: str) -> Path:
    candidate = _safe_child(ARTIFACTS, artifact_id, "artifact")
    if not candidate.is_dir():
        raise HTTPException(404, f"Artifact not found: {artifact_id}")
    missing = missing_artifact_files(candidate)
    if missing:
        raise HTTPException(503, f"Artifact {artifact_id} is incomplete, missing: {missing}")
    return candidate


def _load_table(path: Path, suffix: str | None = None) -> pd.DataFrame:
    kind = (suffix or path.suffix).lower()
    return pd.read_parquet(path) if kind == ".parquet" else pd.read_csv(path)


@lru_cache(maxsize=8)
def _load_artifact(directory: str) -> RaceShiftArtifact:
    return RaceShiftArtifact(Path(directory))


@lru_cache(maxsize=16)
def _cached_inference_table(history: int, path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Feature table for one import file, built once per file version and history length.
    Building the table is the slow part of a forecast (about 10 s for a season); every
    forecast, backtest and comparison on the same file reuses it, whichever artifact runs."""
    return inference_table_for(_load_table(Path(path_str)), history=history)


_TABLE_LOCKS: dict[tuple, threading.Lock] = {}
_TABLE_LOCKS_GUARD = threading.Lock()


def _inference_table(artifact_dir: Path, path: Path) -> pd.DataFrame:
    """Cached feature table with single-flight building: concurrent first requests for the
    same file wait for one build instead of each rebuilding the table."""
    stat = path.stat()
    key = (_load_artifact(str(artifact_dir)).history_laps, str(path), stat.st_mtime_ns, stat.st_size)
    with _TABLE_LOCKS_GUARD:
        lock = _TABLE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        return _cached_inference_table(*key)


def _data_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    source = infer_data_source(frame)
    return {"data_file_source": source, "data_is_synthetic": is_synthetic_source(source)}


def _artifact_dirs() -> list[Path]:
    if not ARTIFACTS.is_dir():
        return []
    return sorted(p for p in ARTIFACTS.iterdir() if p.is_dir())


def _artifact_entry(path: Path) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json") or {}
    config = _read_json(path / "model_config.json") or {}
    is_report = isinstance(metrics.get("models"), dict)  # a baselines folder: metrics only, no weights
    missing = [] if is_report else missing_artifact_files(path)
    split = metrics.get("split") or {}
    label = metrics.get("name") or ("Baseline report" if is_report else path.name)
    if split.get("train_end"):
        label = f"{label} (trained \u2264 {split['train_end']})"
    return {
        "id": path.name,
        "name": "RaceShift FFR" if config.get("model_type") == "RaceShiftFFR" else metrics.get("model", path.name),
        "label": label,
        "role": "primary-trainable" if config.get("model_type") == "RaceShiftFFR" else "baseline-report",
        "training": config.get("training_policy", metrics.get("training_policy")),
        "path": _relative(path),
        "ready": not missing and bool(metrics),
        "missing_files": missing,
        "is_default": path.name == default_artifact_id(),
        "is_synthetic": bool(metrics.get("is_synthetic", False)),
        "data_source": metrics.get("data_source", "unknown"),
        "architecture": config.get("architecture") or metrics.get("architecture"),
        "split": metrics.get("split"),
        "validation": metrics.get("validation"),
        "test": metrics.get("test"),
        "created_utc": metrics.get("created_utc"),
        "lap_validity_version": metrics.get("lap_validity_version"),
        "validity_rules_match": metrics.get("lap_validity_version") == LAP_VALIDITY_VERSION,
    }


def _lap_validity(artifact_dir: Path) -> dict[str, Any]:
    """Which lap-validity rules the runtime applies versus the rules the artifact was trained
    and scored under. A mismatch means the forecast's inputs are built differently from the
    laps the model learned on, so its published metrics do not describe this run."""
    metrics = _read_json(artifact_dir / "metrics.json") or {}
    trained = metrics.get("lap_validity_version")
    return {"runtime": LAP_VALIDITY_VERSION, "artifact": trained, "match": trained == LAP_VALIDITY_VERSION}


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _default_artifact_ready() -> bool:
    return not missing_artifact_files(ARTIFACTS / DEMO_ARTIFACT_ID)


# Files that ship with the repository; deleting them from the UI would break the documented
# first run, so the API refuses (they can still be removed with git or the shell).
SHIPPED_IMPORTS = {"f1_2025_season.parquet", "f1_2026_races.parquet", "synthetic_fixture.csv"}


def _import_files() -> list[dict[str, Any]]:
    if not IMPORTS.is_dir():
        return []
    rows = []
    for p in sorted(IMPORTS.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
            stat = p.stat()
            rows.append({"name": p.name, "bytes": stat.st_size, "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()})
    return rows


def _check_frame_types(frame: pd.DataFrame) -> None:
    """Reject tables whose identity columns cannot be interpreted, with a message that names
    the column instead of a stack trace from deep inside pandas."""
    for column in ("season", "lap_number"):
        if column in frame.columns:
            values = frame[column]
            numeric = pd.to_numeric(values, errors="coerce")
            bad = numeric.isna() & values.notna()
            if bad.any():
                example = str(values[bad].iloc[0])
                raise HTTPException(400, f"Column '{column}' must be numeric; found {example!r}")


def _table_summary(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in frame.columns]
    _check_frame_types(frame)
    summary: dict[str, Any] = {
        "file": name,
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "missing_required_columns": missing,
        "is_synthetic": bool("event" in frame and frame["event"].astype(str).str.startswith("Synthetic_").all()) if len(frame) else False,
        "data_source": infer_data_source(frame),
    }
    if not missing:
        summary.update({
            "seasons": sorted({integer_season(s) for s in frame["season"].unique()} - {None}),
            "events": sorted(frame["event"].astype(str).unique().tolist()),
            "drivers": sorted(frame["driver"].astype(str).unique().tolist()),
            "sessions": RaceShiftArtifact.list_sessions(frame),
        })
        try:
            latest_scope, key = RaceShiftArtifact.latest_session(frame)
        except ValueError:
            # Every row has a blank or fractional identity: nothing the selector can address.
            summary.update({"latest_session": None, "latest_session_drivers": []})
        else:
            summary.update({
                "latest_session": {"season": int(key["season"]), "event": str(key["event"]), "session": str(key["session"])},
                "latest_session_drivers": sorted(latest_scope["driver"].astype(str).unique().tolist()),
            })
    summary["has_chronology"] = bool({"event_date", "round_number"} & set(frame.columns))
    summary["data_warnings"] = _value_warnings(frame)
    if not summary["has_chronology"]:
        summary["data_warnings"].insert(
            0,
            "No 'event_date' or 'round_number' column: the forecaster needs one of them to order events "
            "(historical priors must come from strictly earlier events), so forecasts on this file will be refused.",
        )
    return summary


# Plausible ranges for a Formula 1 lap table. Values outside are not rejected (a user's own
# series may differ) but they are reported, because the model was never shown such values and
# silently imputing or scaling them would produce a confident number from garbage.
_VALUE_RANGES = {
    "lap_time_s": (30.0, 600.0, "s"),
    "tyre_life": (0.0, 100.0, "laps"),
    "position": (1.0, 40.0, ""),
    "track_temp_c": (-20.0, 80.0, "°C"),
    "air_temp_c": (-20.0, 60.0, "°C"),
    "humidity_pct": (0.0, 100.0, "%"),
    "wind_speed_ms": (0.0, 40.0, "m/s"),
    "lap_number": (1.0, 200.0, ""),
}


def _value_warnings(frame: pd.DataFrame) -> list[str]:
    warnings_out: list[str] = []
    for column, (low, high, unit) in _VALUE_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        if bad.any():
            worst = values[bad]
            example = worst.iloc[worst.sub((low + high) / 2).abs().argmax()]
            warnings_out.append(
                f"{int(bad.sum())} rows have {column} outside {low:g}-{high:g} {unit}".rstrip()
                + f" (for example {example:g}); the model was never trained on such values."
            )
    if "driver" in frame.columns:
        drivers = frame["driver"]
        if drivers.isna().any() or drivers.astype(str).str.strip().isin({"", "nan", "None"}).any():
            warnings_out.append("Some rows have an empty driver code; they will appear as a driver named 'nan'.")
    for column in ("season", "event", "session"):
        # Null or blank (whitespace-only) identity values: neither can be a selector key.
        if column in frame.columns and blank_mask(frame[column]).any():
            warnings_out.append(
                f"{int(blank_mask(frame[column]).sum())} rows have an empty {column}; they are listed under a placeholder "
                "session and cannot be forecast."
            )
    key_cols = [c for c in ("season", "event", "session", "driver", "lap_number") if c in frame.columns]
    if len(key_cols) == 5 and frame.duplicated(key_cols).any():
        warnings_out.append(
            f"{int(frame.duplicated(key_cols).sum())} rows duplicate another row's season/event/session/driver/lap_number; "
            "duplicates break lap adjacency and collapse every segment to one lap."
        )
    return warnings_out


def _forecast(
    file: str,
    driver: str | None,
    artifact_id: str | None,
    season: int | None = None,
    event: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    artifact_dir = _safe_artifact_path(artifact_id or default_artifact_id())
    path = _safe_import_path(file)
    try:
        frame = _load_table(path)
        _check_frame_types(frame)
        artifact = _load_artifact(str(artifact_dir))
        result = artifact.forecast_last_available(
            frame, driver=driver, table=_inference_table(artifact_dir, path), season=season, event=event, session=session
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive, keeps internals out of the response
        raise HTTPException(500, f"Local forecast failed: {type(exc).__name__}") from exc
    result["file"] = path.name
    result["session_warning"] = _session_warning(result.get("session"))
    result["lap_validity"] = _lap_validity(artifact_dir)
    result.update(_data_provenance(frame))
    return result


def _session_warning(session: object) -> str | None:
    """The models are trained on race laps only. Qualifying and practice laps (push laps,
    cool-down laps, fuel runs) follow a different process, so a forecast there is not
    evidence of anything; say so rather than returning a bare number."""
    code = str(session or "").strip().upper()
    if code in {"R", "RACE", "S", "SPRINT"}:
        return None
    return (
        f"This is a {code or 'non-race'} session. The model was trained on race laps only; "
        "qualifying and practice laps alternate push and cool-down laps, so this forecast is not meaningful."
    )


# --------------------------------------------------------------------------- models


class ForecastRequest(BaseModel):
    file: str = Field(default="synthetic_fixture.csv", description="Filename inside data/imports")
    driver: str | None = Field(default=None, description="Driver code. Defaults to the driver with most completed laps.")
    artifact: str | None = Field(default=None, description="Artifact directory name inside artifacts/")
    season: int | None = Field(default=None, description="Season of the session to use; defaults to the latest session in the file")
    event: str | None = Field(default=None, description="Event name of the session to use (see /api/imports/{file}/summary sessions)")
    session: str | None = Field(default=None, description="Session code (R, S, Q, ...) when the file holds several for one event")


# --------------------------------------------------------------------------- routes


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "local_first": True,
        "training_policy": "forward-forward-no-global-backprop",
        "demo_artifact_ready": _default_artifact_ready(),
    }


@app.get("/api/runtime")
def runtime() -> dict[str, Any]:
    default_dir = ARTIFACTS / default_artifact_id()
    default_metrics = _read_json(default_dir / "metrics.json") or {}
    return {
        "raceshift_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in ["numpy", "pandas", "scikit-learn", "fastapi", "joblib", "pyarrow"]},
        "data_mode": "offline-local",
        "lap_validity_version": LAP_VALIDITY_VERSION,
        "live_connected": False,
        "live_note": "RaceShift renders local files and historical data. No live timing connection is configured.",
        "default_artifact": {
            "id": default_artifact_id(),
            "ready": _default_artifact_ready(),
            "is_synthetic": bool(default_metrics.get("is_synthetic", False)),
            "data_source": default_metrics.get("data_source", "unknown"),
        },
        "project_root": ROOT.name,
        "api_bind": f"{os.environ.get('API_HOST', '127.0.0.1')}:{os.environ.get('API_PORT', '8000')}",
        "ui_origins": UI_ORIGINS,
    }


@app.get("/api/setup")
def setup() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing_pkgs = [n for n in ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "fastapi"] if _package_version(n) is None]
    checks.append({"item": "Python dependencies", "ok": not missing_pkgs, "detail": "all installed" if not missing_pkgs else f"missing: {missing_pkgs}"})
    default_missing = missing_artifact_files(ARTIFACTS / DEMO_ARTIFACT_ID)
    checks.append({"item": "Demo model artifact", "ok": not default_missing, "detail": f"artifacts/{DEMO_ARTIFACT_ID}" + ("" if not default_missing else f" missing {default_missing}")})
    real = [p.name for p in _artifact_dirs() if not missing_artifact_files(p) and not (_read_json(p / "metrics.json") or {}).get("is_synthetic", False) and (_read_json(p / "model_config.json") or {}).get("model_type") == "RaceShiftFFR"]
    checks.append({"item": "Real (non-synthetic) FFR artifact", "ok": bool(real), "detail": ", ".join(real) if real else "none yet. Train in Colab and copy the folder into artifacts/"})
    imports = _import_files()
    checks.append({"item": "Local datasets in data/imports", "ok": bool(imports), "detail": ", ".join(i["name"] for i in imports) if imports else "none"})
    processed = sorted(p.name for p in PROCESSED.glob("*") if p.is_file() and not p.name.startswith(".")) if PROCESSED.is_dir() else []
    checks.append({"item": "Processed tables in data/processed", "ok": True, "detail": ", ".join(processed) if processed else "none (optional)"})
    checks.append({"item": "Secrets", "ok": True, "detail": "The API reads no credentials and returns none. Live data is not configured."})
    return {"checks": checks, "all_required_ok": all(c["ok"] for c in checks if c["item"] in {"Python dependencies", "Demo model artifact"})}


@app.get("/api/models")
def models() -> dict[str, Any]:
    artifacts = [_artifact_entry(p) for p in _artifact_dirs()]
    registry = [
        {"id": "previous_lap", "name": "Previous lap", "role": "required-baseline", "training": "none", "ready": True},
        {"id": "rolling_median_5", "name": "Rolling-five median", "role": "required-baseline", "training": "none", "ready": True},
        {"id": "ridge", "name": "Ridge regression", "role": "required-baseline", "training": "closed-form on the residual target", "ready": True},
        {"id": "hist_gradient_boosting", "name": "Gradient-boosted trees", "role": "required-baseline", "training": "scikit-learn HistGradientBoosting", "ready": True},
        {"id": "frozen-tsfm", "name": "Frozen time-series foundation models", "role": "optional-zero-shot-benchmark", "training": "none", "ready": False},
    ]
    return {"default_artifact": default_artifact_id(), "artifacts": artifacts, "baselines": registry}


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    manifest = _read_json(ROOT / "dataset_manifest.json") or {"sources": []}
    processed = sorted(_relative(p) for p in PROCESSED.glob("*") if p.is_file() and not p.name.startswith(".")) if PROCESSED.is_dir() else []
    return {
        "sources": manifest.get("sources", []),
        "imports": _import_files(),
        "processed_files": processed,
        "import_dir": _relative(IMPORTS),
        "shipped_imports": sorted(SHIPPED_IMPORTS),
    }


@app.get("/api/imports/{filename}/summary")
def import_summary(filename: str) -> dict[str, Any]:
    path = _safe_import_path(filename)
    try:
        frame = _load_table(path)
    except Exception as exc:
        raise HTTPException(400, f"Could not read {filename}: {type(exc).__name__}") from exc
    return _table_summary(frame, path.name)


@app.post("/api/import")
async def import_file(file: UploadFile = File(...), overwrite: bool = Query(False)) -> dict[str, Any]:
    original = Path(file.filename or "").name
    safe_name = _SAFE_NAME.sub("_", original).strip("._")
    if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Only .csv and .parquet uploads are supported")
    target = _safe_child(IMPORTS, safe_name, "import file")
    if target.exists() and not overwrite:
        raise HTTPException(409, f"{safe_name} already exists. Pass overwrite=true to replace it.")
    IMPORTS.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.uploading"
    written = 0
    try:
        with tmp.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_IMPORT_BYTES:
                    raise HTTPException(413, f"Import exceeds {MAX_IMPORT_BYTES // (1024 * 1024)} MB limit")
                handle.write(chunk)
        frame = _load_table(tmp, suffix=target.suffix)
        if len(frame) > MAX_IMPORT_ROWS:
            raise HTTPException(413, f"Import has {len(frame)} rows; the limit is {MAX_IMPORT_ROWS}")
        if len(frame.columns) > MAX_IMPORT_COLUMNS:
            raise HTTPException(413, f"Import has {len(frame.columns)} columns; the limit is {MAX_IMPORT_COLUMNS}")
        if frame.empty or len(frame.columns) < 2:
            raise HTTPException(400, "Upload parsed to an empty or single-column table")
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"Upload is not a readable {target.suffix} table: {type(exc).__name__}") from exc
    finally:
        await file.close()
    missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in frame.columns]
    if missing:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"Upload rejected: missing required columns {missing}. Required: {list(REQUIRED_FORECAST_COLUMNS)}; "
            "see docs/FEATURE_CONTRACT.md.",
        )
    shutil.move(str(tmp), str(target))
    summary = _table_summary(frame, target.name)
    summary["bytes"] = written
    summary["stored_as"] = _relative(target)
    return summary


@app.delete("/api/imports/{filename}")
def delete_import(filename: str) -> dict[str, Any]:
    path = _safe_import_path(filename)
    if path.name in SHIPPED_IMPORTS:
        raise HTTPException(403, f"{path.name} ships with the repository and cannot be deleted from the UI")
    path.unlink()
    return {"deleted": path.name, "imports": _import_files()}


EXPORTS = ROOT / "exports"


@app.get("/api/models/{artifact_id}/export")
def export_model(artifact_id: str) -> FileResponse:
    """Build (or reuse) the export bundle for a complete artifact and return it as a zip.

    The bundle holds the weights, preprocessor, feature contract, metrics, the verified ONNX
    core, the JSON preprocessor spec and the model card. It is rebuilt when metrics.json is
    newer than the last export.
    """
    artifact_dir = _safe_artifact_path(artifact_id)
    bundle = EXPORTS / f"{artifact_dir.name}.zip"
    # The API never writes inside artifacts/ (those files are tracked in git); the export is
    # staged under exports/<artifact>/ and zipped from there.
    export_dir = EXPORTS / artifact_dir.name
    manifest = export_dir / "export_manifest.json"
    stale = not bundle.is_file() or not manifest.is_file() or manifest.stat().st_mtime < (artifact_dir / "metrics.json").stat().st_mtime
    if stale:
        try:
            from raceshift.models.export import export_artifact

            export_artifact(artifact_dir, bundle_dir=EXPORTS, export_dir=export_dir)
        except ImportError as exc:
            raise HTTPException(503, f"Model export needs the export extras (pip install -e '.[export]'): {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Export failed: {type(exc).__name__}") from exc
    return FileResponse(bundle, media_type="application/zip", filename=bundle.name)


@app.post("/api/forecast/latest")
def forecast_latest(request: ForecastRequest) -> dict[str, Any]:
    return _forecast(request.file, request.driver, request.artifact, request.season, request.event, request.session)


@app.get("/api/forecast")
def forecast(
    file: str = Query("synthetic_fixture.csv", description="Filename inside data/imports"),
    driver: str | None = Query(None),
    artifact: str | None = Query(None),
    season: int | None = Query(None),
    event: str | None = Query(None),
    session: str | None = Query(None),
) -> dict[str, Any]:
    """Query-string alias of POST /api/forecast/latest kept for curl convenience."""
    return _forecast(file, driver, artifact, season, event, session)


class BacktestRequest(ForecastRequest):
    laps: int = Field(default=10, ge=1, le=100, description="How many of the driver's last completed lap pairs to score")


@app.post("/api/forecast/backtest")
def forecast_backtest(request: BacktestRequest) -> dict[str, Any]:
    """Predicted vs actual for the last N lap pairs of one driver in the file's latest session.

    This is how a user checks the model on laps that were really driven. The rows are the
    training-style rows (lap N valid, lap N+1 valid and adjacent); the model never sees lap N+1.
    """
    artifact_dir = _safe_artifact_path(request.artifact or default_artifact_id())
    path = _safe_import_path(request.file)
    try:
        frame = _load_table(path)
        artifact = _load_artifact(str(artifact_dir))
        _check_frame_types(frame)
        result = artifact.backtest_session(
            frame, driver=request.driver, laps=request.laps, table=_inference_table(artifact_dir, path),
            season=request.season, event=request.event, session=request.session,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"Local backtest failed: {type(exc).__name__}") from exc
    result["file"] = path.name
    result["session_warning"] = _session_warning(result.get("session"))
    result["lap_validity"] = _lap_validity(artifact_dir)
    result.update(_data_provenance(frame))
    return result


# --------------------------------------------------------------------------- reports

REPORTS = ROOT / "reports"
REPORT_TITLES = {
    "f1_2025h2": "Season-round split, 2018-2025 FastF1 tier",
    "f1_2025h2_legacy_ext": "Training extended to 2000 with the legacy tier",
    "holdout_monza": "Circuit holdout (Monza)",
    "domain_shift_2026": "2026 domain shift, no retraining",
}


@app.get("/api/reports")
def reports() -> dict[str, Any]:
    """Committed experiment reports (reports/<name>/breakdowns.json): MAE by circuit, team, compound, ..."""
    out: list[dict[str, Any]] = []
    if REPORTS.is_dir():
        for path in sorted(p for p in REPORTS.iterdir() if p.is_dir()):
            breakdowns = _read_json(path / "breakdowns.json") or {}
            out.append(
                {
                    "name": path.name,
                    "title": REPORT_TITLES.get(path.name),
                    "models": breakdowns.get("models", []),
                    "rows": breakdowns.get("rows"),
                    "breakdowns": breakdowns.get("breakdowns"),
                    "has_summary": (path / "summary.md").is_file(),
                }
            )
    return {"reports": out}


@app.get("/api/reports/drivers")
def driver_reports() -> dict[str, Any]:
    """Per-driver test error of every complete artifact, straight from its metrics.json."""
    out: list[dict[str, Any]] = []
    for path in _artifact_dirs():
        metrics = _read_json(path / "metrics.json")
        if not metrics or not isinstance(metrics.get("test_by_driver"), dict):
            continue
        out.append(
            {
                "artifact": path.name,
                "name": metrics.get("name", path.name),
                "is_synthetic": bool(metrics.get("is_synthetic", False)),
                "split": metrics.get("split"),
                "test_by_driver": metrics["test_by_driver"],
            }
        )
    return {"artifacts": out}


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for path in _artifact_dirs():
        metrics = _read_json(path / "metrics.json")
        if not metrics:
            continue
        common = {
            "artifact": path.name,
            "data_source": metrics.get("data_source", "unknown"),
            "is_synthetic": bool(metrics.get("is_synthetic", False)),
            "split": metrics.get("split"),
            "created_utc": metrics.get("created_utc"),
            "input_file": metrics.get("input_file"),
        }
        if isinstance(metrics.get("models"), dict):
            for name, result in metrics["models"].items():
                runs.append({"run": f"{path.name}/{name}", "model": name, "method": "baseline", "training_policy": metrics.get("training_policy"), "validation": result.get("validation"), "test": result.get("test"), "resources": result.get("resources"), **common})
        else:
            runs.append({"run": path.name, "model": metrics.get("name", metrics.get("model", path.name)), "method": "forward-forward" if "forward-forward" in str(metrics.get("training_policy", "")) else metrics.get("training_policy"), "training_policy": metrics.get("training_policy"), "architecture": metrics.get("architecture"), "validation": metrics.get("validation"), "test": metrics.get("test"), "resources": metrics.get("resources"), **common})
    return {"experiments": runs}
