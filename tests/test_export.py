"""Model export: the ONNX core must reproduce the NumPy model, and the bundle must be complete."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from raceshift.models.export import export_artifact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "raceshift_ffr_demo"


@pytest.fixture()
def demo_copy(tmp_path: Path) -> Path:
    target = tmp_path / "raceshift_ffr_demo"
    shutil.copytree(DEMO, target, ignore=shutil.ignore_patterns("export"))
    return target


def test_export_writes_verified_onnx_card_and_bundle(demo_copy: Path, tmp_path: Path):
    result = export_artifact(demo_copy, bundle_dir=tmp_path / "bundles")
    assert result["ffr_onnx"].exists()
    assert max(result["verification"].values()) < 1e-3
    card = (demo_copy / "export" / "MODEL_CARD.md").read_text()
    assert "synthetic fixture, not Formula 1 results" in card
    assert "No global backpropagation" in card
    manifest = json.loads((demo_copy / "export" / "export_manifest.json").read_text())
    assert manifest["artifact"] == "raceshift_ffr_demo"
    with zipfile.ZipFile(result["bundle"]) as zf:
        names = set(zf.namelist())
    assert {"raceshift_ffr_demo/model_weights.npz", "raceshift_ffr_demo/preprocessor.joblib", "raceshift_ffr_demo/metrics.json",
            "raceshift_ffr_demo/export/MODEL_CARD.md", f"raceshift_ffr_demo/export/{result['ffr_onnx'].name}"} <= names


def test_onnx_core_matches_numpy_on_real_preprocessed_rows(demo_copy: Path):
    import joblib
    import onnxruntime as ort
    import pandas as pd

    from raceshift.features.full_context import build_full_context_table
    from raceshift.models.forward_forward_regressor import ForwardForwardRegressor

    contract = json.loads((demo_copy / "feature_contract.json").read_text())
    table = build_full_context_table(pd.read_csv(ROOT / "data" / "imports" / "synthetic_fixture.csv"), history=contract["history"])
    x = np.asarray(joblib.load(demo_copy / "preprocessor.joblib").transform(table[contract["numeric"] + contract["categorical"]]), dtype=np.float32)
    result = export_artifact(demo_copy, verify_rows=x)
    session = ort.InferenceSession(str(result["ffr_onnx"]), providers=["CPUExecutionProvider"])
    residual, lower, upper, disagreement = session.run(None, {"features": x})
    model = ForwardForwardRegressor.load(demo_copy)
    expected = model.predict_with_uncertainty(x)
    np.testing.assert_allclose(residual, expected["prediction"], atol=1e-4)
    np.testing.assert_allclose(lower, expected["lower_80"], atol=1e-4)
    np.testing.assert_allclose(upper, expected["upper_80"], atol=1e-4)
    assert np.all(lower <= residual) and np.all(residual <= upper)


def test_preprocessor_spec_keeps_none_distinct_from_the_string_none():
    """Legacy-tier rows carry a Python None compound; sklearn keeps it as its own category,
    distinct from the string "None" and from float NaN. The JSON spec must agree column for column."""
    import numpy as np
    import pandas as pd

    from raceshift.features.preprocessing import make_preprocessor
    from raceshift.models.export import apply_preprocessor_spec, preprocessor_spec

    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "compound": ["SOFT", None, "None", np.nan, "HARD", None],
            "team": ["A", "B", "A", "B", "A", "B"],
        }
    )
    contract = {"numeric": ["x"], "categorical": ["compound", "team"]}
    prep = make_preprocessor(contract["numeric"], contract["categorical"]).fit(frame)
    spec = preprocessor_spec(prep, contract)
    expected = np.asarray(prep.transform(frame), dtype=np.float32)
    got = apply_preprocessor_spec(spec, frame)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-6)
    compound = next(c for c in spec["categorical"] if c["column"] == "compound")
    vocabulary = compound["frequent"] + compound["infrequent"]
    assert None in vocabulary and "None" in vocabulary and "missing" in vocabulary


def test_committed_export_specs_match_their_preprocessors():
    """Every export spec tracked in git must agree with the joblib preprocessor next to it on
    real shipped laps, otherwise pickle-free inference silently drifts (this happened once)."""
    import glob
    import json

    import joblib
    import numpy as np
    import pandas as pd

    from raceshift.features.full_context import build_full_context_table
    from raceshift.models.export import apply_preprocessor_spec

    root = Path(__file__).resolve().parents[1]
    specs = sorted(glob.glob(str(root / "artifacts" / "*" / "export" / "*_preprocessor.json")))
    assert specs, "no committed export specs found"
    raw = pd.read_parquet(root / "data" / "imports" / "f1_2025_season.parquet")
    tables: dict[int, pd.DataFrame] = {}
    for spec_path in specs:
        artifact = Path(spec_path).parents[1]
        contract = json.loads((artifact / "feature_contract.json").read_text())
        history = int(contract["history"])
        if history not in tables:
            tables[history] = build_full_context_table(raw, history=history)
        cols = contract["numeric"] + contract["categorical"]
        frame = tables[history][cols].head(500)
        prep = joblib.load(artifact / "preprocessor.joblib")
        spec = json.loads(Path(spec_path).read_text())
        expected = np.asarray(prep.transform(frame), dtype=np.float32)
        got = apply_preprocessor_spec(spec, frame)
        assert got.shape == expected.shape, spec_path
        np.testing.assert_allclose(got, expected, atol=1e-4, err_msg=spec_path)
