from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from make_synthetic_fixture import build_fixture

import apps.api.main as api

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    imports = tmp_path / "imports"
    imports.mkdir()
    build_fixture(seasons=(2024, 2025), events_per_season=2, laps=14).to_csv(imports / "laps.csv", index=False)
    monkeypatch.setattr(api, "IMPORTS", imports)
    return TestClient(api.app)


def test_health_runtime_setup_report_offline_local_mode(client: TestClient):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["demo_artifact_ready"] is True
    runtime = client.get("/api/runtime").json()
    assert runtime["live_connected"] is False
    assert runtime["data_mode"] == "offline-local"
    # The default is the committed real-data artifact when it is complete, else the demo.
    assert runtime["default_artifact"]["id"] in {api.REAL_ARTIFACT_ID, api.DEMO_ARTIFACT_ID}
    assert runtime["default_artifact"]["ready"] is True
    assert "/" not in runtime["project_root"]
    setup = client.get("/api/setup").json()
    assert setup["all_required_ok"] is True
    assert {c["item"] for c in setup["checks"]} >= {"Python dependencies", "Demo model artifact"}


def test_models_and_datasets_list_local_resources(client: TestClient):
    models = client.get("/api/models").json()
    demo = next(a for a in models["artifacts"] if a["id"] == "raceshift_ffr_demo")
    assert demo["ready"] is True and demo["is_synthetic"] is True
    assert sum(a["is_default"] for a in models["artifacts"]) == 1
    assert models["default_artifact"] == api.default_artifact_id()
    assert {b["id"] for b in models["baselines"]} >= {"previous_lap", "rolling_median_5", "ridge", "hist_gradient_boosting"}
    datasets = client.get("/api/datasets").json()
    assert [i["name"] for i in datasets["imports"]] == ["laps.csv"]
    assert datasets["sources"], "dataset_manifest.json sources expected"


def test_import_summary_and_forecast_latest(client: TestClient):
    summary = client.get("/api/imports/laps.csv/summary").json()
    assert summary["rows"] == 2 * 2 * 4 * 14
    assert summary["missing_required_columns"] == []
    assert summary["latest_session"] == {"season": 2025, "event": "Synthetic_GP_2", "session": "R"}

    result = client.post("/api/forecast/latest", json={"file": "laps.csv", "driver": "AAA", "artifact": "raceshift_ffr_demo"})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["driver"] == "AAA" and body["season"] == 2025 and body["lap_number_completed"] == 14
    assert body["lower_80_s"] <= body["predicted_next_lap_s"] <= body["upper_80_s"]
    assert body["is_synthetic"] is True and body["artifact"] == "raceshift_ffr_demo"

    backtest = client.post("/api/forecast/backtest", json={"file": "laps.csv", "driver": "AAA", "artifact": "raceshift_ffr_demo", "laps": 6})
    assert backtest.status_code == 200, backtest.text
    bt = backtest.json()
    assert bt["driver"] == "AAA" and 1 <= len(bt["laps"]) <= 6
    assert all(lap["actual_next_lap_s"] > 0 and lap["abs_error_s"] >= 0 for lap in bt["laps"])
    assert bt["summary"]["mae_s"] >= 0 and 0 <= bt["summary"]["interval80_coverage"] <= 1
    assert bt["laps"][-1]["next_lap_number"] <= 14

    alias = client.get("/api/forecast", params={"file": "laps.csv", "driver": "AAA", "artifact": "raceshift_ffr_demo"}).json()
    assert alias["predicted_next_lap_s"] == pytest.approx(body["predicted_next_lap_s"])


def test_forecast_rejects_bad_inputs(client: TestClient):
    assert client.post("/api/forecast/latest", json={"file": "../pyproject.toml"}).status_code == 400
    assert client.post("/api/forecast/latest", json={"file": "missing.csv"}).status_code == 404
    assert client.post("/api/forecast/latest", json={"file": "laps.csv", "driver": "ZZZ"}).status_code == 400
    assert client.post("/api/forecast/latest", json={"file": "laps.csv", "artifact": "../data"}).status_code == 400
    assert client.post("/api/forecast/latest", json={"file": "laps.csv", "artifact": "nope"}).status_code == 404


def test_import_upload_validates_type_and_writes_file(client: TestClient):
    bad = client.post("/api/import", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
    csv_bytes = build_fixture(seasons=(2025,), events_per_season=1, laps=8).to_csv(index=False).encode()
    ok = client.post("/api/import", files={"file": ("../evil name.csv", csv_bytes, "text/csv")})
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["file"] == "evil_name.csv" and body["rows"] == 32 and body["missing_required_columns"] == []
    assert (api.IMPORTS / "evil_name.csv").is_file()
    dup = client.post("/api/import", files={"file": ("evil_name.csv", csv_bytes, "text/csv")})
    assert dup.status_code == 409
    garbage = client.post("/api/import", files={"file": ("broken.parquet", b"not parquet", "application/octet-stream")})
    assert garbage.status_code == 400
    assert not (api.IMPORTS / "broken.parquet").exists()


def test_experiments_flatten_baseline_reports(client: TestClient):
    runs = client.get("/api/experiments").json()["experiments"]
    demo = next(r for r in runs if r["run"] == "raceshift_ffr_demo")
    assert demo["is_synthetic"] is True and demo["test"]["mae_s"] > 0
    assert demo["method"] == "forward-forward"


def test_import_enforces_size_and_row_limits(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    csv_bytes = build_fixture(seasons=(2025,), events_per_season=1, laps=8).to_csv(index=False).encode()
    monkeypatch.setattr(api, "MAX_IMPORT_BYTES", 1024)
    too_big = client.post("/api/import", files={"file": ("big.csv", csv_bytes, "text/csv")})
    assert too_big.status_code == 413
    assert not (api.IMPORTS / "big.csv").exists() and not list(api.IMPORTS.glob(".*uploading"))
    monkeypatch.setattr(api, "MAX_IMPORT_BYTES", 200 * 1024 * 1024)
    monkeypatch.setattr(api, "MAX_IMPORT_ROWS", 10)
    too_many = client.post("/api/import", files={"file": ("rows.csv", csv_bytes, "text/csv")})
    assert too_many.status_code == 413
    assert not (api.IMPORTS / "rows.csv").exists()


def test_forecast_reports_input_and_historical_context(client: TestClient):
    body = client.post("/api/forecast/latest", json={"file": "laps.csv", "driver": "AAA", "artifact": "raceshift_ffr_demo"}).json()
    assert body["context"]["compound"] in {"MEDIUM", "HARD"}
    assert body["context"]["tyre_life"] >= 1
    assert body["context"]["track_temp_c"] is not None
    # Two seasons of fixture history exist, so the driver/circuit prior is available.
    assert body["historical_context"]["driver_circuit_pace_s"] is not None
    assert body["historical_context"]["driver_overall_pace_s"] is not None


def test_import_enforces_column_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    csv_bytes = build_fixture(seasons=(2025,), events_per_season=1, laps=8).to_csv(index=False).encode()
    monkeypatch.setattr(api, "MAX_IMPORT_COLUMNS", 5)
    too_wide = client.post("/api/import", files={"file": ("wide.csv", csv_bytes, "text/csv")})
    assert too_wide.status_code == 413
    assert not (api.IMPORTS / "wide.csv").exists()


def test_import_summary_reports_data_tier_provenance(client: TestClient):
    summary = client.get("/api/imports/laps.csv/summary").json()
    assert summary["is_synthetic"] and summary["data_source"] == "synthetic_fixture"


def test_model_export_endpoint_returns_bundle(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pytest.importorskip("onnxruntime")
    import shutil

    artifacts = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts" / "raceshift_ffr_demo", artifacts / "raceshift_ffr_demo", ignore=shutil.ignore_patterns("export"))
    monkeypatch.setattr(api, "ARTIFACTS", artifacts)
    monkeypatch.setattr(api, "EXPORTS", tmp_path / "exports")
    response = client.get("/api/models/raceshift_ffr_demo/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    assert len(response.content) > 10_000
    # A read-only download must never write inside artifacts/ (those files are tracked in git).
    assert not (artifacts / "raceshift_ffr_demo" / "export").exists()
    assert (tmp_path / "exports" / "raceshift_ffr_demo" / "MODEL_CARD.md").exists()
    assert client.get("/api/models/nope/export").status_code == 404
    assert client.get("/api/models/..%2Fetc/export").status_code in (400, 404)
