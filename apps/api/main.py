"""RaceShift local API.

Localhost-only FastAPI service that the React/Vite UI talks to. It reads local files and
local model artifacts. It never holds credentials, never trains, and never claims a live
connection when it is serving fixture or historical data.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import sys
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from raceshift import __version__
from raceshift.data.provenance import infer_data_source
from raceshift.data.schema import REQUIRED_FORECAST_COLUMNS
from raceshift.models.artifact import ARTIFACT_FILES, RaceShiftArtifact, missing_artifact_files

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
IMPORTS = DATA / "imports"
PROCESSED = DATA / "processed"
DEFAULT_ARTIFACT_ID = "raceshift_ffr_demo"
ALLOWED_SUFFIXES = {".csv", ".parquet"}
MAX_IMPORT_BYTES = 200 * 1024 * 1024
MAX_IMPORT_ROWS = 2_000_000
MAX_IMPORT_COLUMNS = 250
UI_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

app = FastAPI(title="RaceShift Local API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=UI_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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
    if not name or name != Path(name).name or name in {".", ".."}:
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


def _artifact_dirs() -> list[Path]:
    if not ARTIFACTS.is_dir():
        return []
    return sorted(p for p in ARTIFACTS.iterdir() if p.is_dir())


def _artifact_entry(path: Path) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json") or {}
    config = _read_json(path / "model_config.json") or {}
    missing = missing_artifact_files(path)
    return {
        "id": path.name,
        "name": "RaceShift FFR" if config.get("model_type") == "RaceShiftFFR" else metrics.get("model", path.name),
        "role": "primary-trainable" if config.get("model_type") == "RaceShiftFFR" else "baseline-report",
        "training": config.get("training_policy", metrics.get("training_policy")),
        "path": _relative(path),
        "ready": not missing,
        "missing_files": missing,
        "is_default": path.name == DEFAULT_ARTIFACT_ID,
        "is_synthetic": bool(metrics.get("is_synthetic", False)),
        "data_source": metrics.get("data_source", "unknown"),
        "architecture": config.get("architecture") or metrics.get("architecture"),
        "split": metrics.get("split"),
        "validation": metrics.get("validation"),
        "test": metrics.get("test"),
        "created_utc": metrics.get("created_utc"),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _default_artifact_ready() -> bool:
    return not missing_artifact_files(ARTIFACTS / DEFAULT_ARTIFACT_ID)


def _import_files() -> list[dict[str, Any]]:
    if not IMPORTS.is_dir():
        return []
    rows = []
    for p in sorted(IMPORTS.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
            stat = p.stat()
            rows.append({"name": p.name, "bytes": stat.st_size, "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()})
    return rows


def _table_summary(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in frame.columns]
    summary: dict[str, Any] = {
        "file": name,
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "missing_required_columns": missing,
        "is_synthetic": bool("event" in frame and frame["event"].astype(str).str.startswith("Synthetic_").all()) if len(frame) else False,
        "data_source": infer_data_source(frame),
    }
    if not missing:
        latest_scope, key = RaceShiftArtifact.latest_session(frame)
        summary.update({
            "seasons": sorted(int(s) for s in pd.to_numeric(frame["season"], errors="coerce").dropna().unique()),
            "events": sorted(frame["event"].astype(str).unique().tolist()),
            "drivers": sorted(frame["driver"].astype(str).unique().tolist()),
            "latest_session": {"season": int(key["season"]), "event": str(key["event"]), "session": str(key["session"])},
            "latest_session_drivers": sorted(latest_scope["driver"].astype(str).unique().tolist()),
        })
    return summary


def _forecast(file: str, driver: str | None, artifact_id: str | None) -> dict[str, Any]:
    artifact_dir = _safe_artifact_path(artifact_id or DEFAULT_ARTIFACT_ID)
    path = _safe_import_path(file)
    try:
        frame = _load_table(path)
        artifact = _load_artifact(str(artifact_dir))
        result = artifact.forecast_last_available(frame, driver=driver)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive, keeps internals out of the response
        raise HTTPException(500, f"Local forecast failed: {type(exc).__name__}") from exc
    result["file"] = path.name
    return result


# --------------------------------------------------------------------------- models


class ForecastRequest(BaseModel):
    file: str = Field(default="synthetic_fixture.csv", description="Filename inside data/imports")
    driver: str | None = Field(default=None, description="Driver code. Defaults to the driver with most completed laps.")
    artifact: str | None = Field(default=None, description="Artifact directory name inside artifacts/")


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
    default_dir = ARTIFACTS / DEFAULT_ARTIFACT_ID
    default_metrics = _read_json(default_dir / "metrics.json") or {}
    return {
        "raceshift_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in ["numpy", "pandas", "scikit-learn", "fastapi", "joblib", "pyarrow"]},
        "data_mode": "offline-local",
        "live_connected": False,
        "live_note": "RaceShift renders local files and historical data. No live timing connection is configured.",
        "default_artifact": {
            "id": DEFAULT_ARTIFACT_ID,
            "ready": _default_artifact_ready(),
            "is_synthetic": bool(default_metrics.get("is_synthetic", False)),
            "data_source": default_metrics.get("data_source", "unknown"),
        },
        "project_root": ROOT.name,
        "api_bind": "127.0.0.1:8000",
        "ui_origins": UI_ORIGINS,
    }


@app.get("/api/setup")
def setup() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing_pkgs = [n for n in ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "fastapi"] if _package_version(n) is None]
    checks.append({"item": "Python dependencies", "ok": not missing_pkgs, "detail": "all installed" if not missing_pkgs else f"missing: {missing_pkgs}"})
    default_missing = missing_artifact_files(ARTIFACTS / DEFAULT_ARTIFACT_ID)
    checks.append({"item": "Demo model artifact", "ok": not default_missing, "detail": f"artifacts/{DEFAULT_ARTIFACT_ID}" + ("" if not default_missing else f" missing {default_missing}")})
    real = [p.name for p in _artifact_dirs() if not missing_artifact_files(p) and not (_read_json(p / "metrics.json") or {}).get("is_synthetic", False) and (_read_json(p / "model_config.json") or {}).get("model_type") == "RaceShiftFFR"]
    checks.append({"item": "Real (non-synthetic) FFR artifact", "ok": bool(real), "detail": ", ".join(real) if real else "none yet. Train in Colab and copy the folder into artifacts/"})
    imports = _import_files()
    checks.append({"item": "Local datasets in data/imports", "ok": bool(imports), "detail": ", ".join(i["name"] for i in imports) if imports else "none"})
    processed = sorted(p.name for p in PROCESSED.glob("*") if p.is_file()) if PROCESSED.is_dir() else []
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
    return {"default_artifact": DEFAULT_ARTIFACT_ID, "artifacts": artifacts, "baselines": registry}


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    manifest = _read_json(ROOT / "dataset_manifest.json") or {"sources": []}
    processed = sorted(_relative(p) for p in PROCESSED.glob("*") if p.is_file()) if PROCESSED.is_dir() else []
    return {"sources": manifest.get("sources", []), "imports": _import_files(), "processed_files": processed, "import_dir": _relative(IMPORTS)}


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
    shutil.move(str(tmp), str(target))
    summary = _table_summary(frame, target.name)
    summary["bytes"] = written
    summary["stored_as"] = _relative(target)
    return summary


@app.post("/api/forecast/latest")
def forecast_latest(request: ForecastRequest) -> dict[str, Any]:
    return _forecast(request.file, request.driver, request.artifact)


@app.get("/api/forecast")
def forecast(
    file: str = Query("synthetic_fixture.csv", description="Filename inside data/imports"),
    driver: str | None = Query(None),
    artifact: str | None = Query(None),
) -> dict[str, Any]:
    """Query-string alias of POST /api/forecast/latest kept for curl convenience."""
    return _forecast(file, driver, artifact)


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
