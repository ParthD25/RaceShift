from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from raceshift.models.artifact import RaceShiftArtifact

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
IMPORTS = DATA / "imports"
DEFAULT_ARTIFACT = ARTIFACTS / "raceshift_ffr_demo"

app = FastAPI(title="RaceShift Local API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _safe_import_path(filename: str) -> Path:
    candidate = (IMPORTS / filename).resolve()
    root = IMPORTS.resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(400, "Invalid import filename")
    if candidate.suffix.lower() not in {".csv", ".parquet"}:
        raise HTTPException(400, "Only CSV and Parquet are supported")
    if not candidate.exists():
        raise HTTPException(404, f"Import file not found: {filename}")
    return candidate


def _load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "0.4.0",
        "local_first": True,
        "training_policy": "forward-forward-no-global-backprop",
        "demo_artifact_ready": DEFAULT_ARTIFACT.exists(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    registry = [
        {
            "id": "raceshift-ffr",
            "name": "RaceShift FFR",
            "role": "primary-trainable",
            "training": "Forward-Forward local updates",
            "artifact": str(DEFAULT_ARTIFACT),
        },
        {"id": "rolling5", "name": "Rolling-five median", "role": "required-baseline", "training": "none"},
        {"id": "ridge", "name": "Ridge regression", "role": "required-baseline", "training": "closed-form / conventional baseline"},
        {"id": "frozen-tsfm", "name": "Frozen time-series foundation models", "role": "optional-zero-shot-benchmark", "training": "none"},
    ]
    for model in registry:
        artifact = model.get("artifact")
        model["ready"] = bool(artifact and Path(artifact).exists()) if artifact else True
        if artifact:
            model["metrics"] = _read_json(Path(artifact) / "metrics.json")
            model["config"] = _read_json(Path(artifact) / "model_config.json")
    return {"models": registry}


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    manifest = _read_json(ROOT / "dataset_manifest.json") or {"sources": []}
    imports = sorted(p.name for p in IMPORTS.glob("*") if p.suffix.lower() in {".csv", ".parquet"})
    processed = sorted(str(p.relative_to(ROOT)) for p in (DATA / "processed").glob("*") if p.is_file())
    return {"sources": manifest.get("sources", []), "imports": imports, "processed_files": processed}


@app.get("/api/forecast")
def forecast(
    file: str = Query("synthetic_fixture.csv", description="Filename inside data/imports"),
    driver: str | None = Query(None),
) -> dict[str, Any]:
    if not DEFAULT_ARTIFACT.exists():
        raise HTTPException(503, "RaceShift demo artifact is not installed. Train or copy an artifact first.")
    path = _safe_import_path(file)
    try:
        frame = _load_table(path)
        artifact = RaceShiftArtifact(DEFAULT_ARTIFACT)
        return artifact.forecast_last_available(frame, driver=driver)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Local forecast failed: {exc}") from exc


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    runs = []
    if ARTIFACTS.exists():
        for metrics_path in ARTIFACTS.glob("*/metrics.json"):
            metrics = _read_json(metrics_path)
            if metrics:
                runs.append({"run": metrics_path.parent.name, "metrics": metrics})
    return {"experiments": runs}
