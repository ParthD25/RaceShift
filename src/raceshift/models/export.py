"""Export a saved RaceShift FFR artifact to portable formats.

Three things come out of :func:`export_artifact`:

* ``<name>_ffr.onnx``            the Forward-Forward core: preprocessed features in, next-lap
                                 residual, 80% interval bounds and layer disagreement out.
                                 Built directly from the NumPy weights, verified against the
                                 NumPy implementation with onnxruntime before it is written.
* ``<name>_preprocessor.onnx``   the train-only sklearn preprocessor (imputers, scaler,
                                 one-hot vocabularies) converted with skl2onnx, when the
                                 converter supports every step; otherwise the joblib file
                                 remains the preprocessor and the model card says so.
* ``<name>_end_to_end.onnx``     preprocessor + core merged into one graph that takes the raw
                                 contract columns, when both halves exist.

Plus ``MODEL_CARD.md`` (data, split, metrics, resources, limitations, inference snippet) and a
zip bundle of the whole artifact for download. Everything is derived from the artifact files;
nothing is typed by hand.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from raceshift.models.forward_forward_regressor import ForwardForwardRegressor

OPSET = 13
RESIDUAL_OUTPUTS = ("residual_s", "lower_80_residual_s", "upper_80_residual_s", "layer_disagreement_s")


def _const(name: str, array: np.ndarray):
    from onnx import numpy_helper

    return numpy_helper.from_array(np.ascontiguousarray(array), name=name)


def build_ffr_onnx(model: ForwardForwardRegressor, input_name: str = "features"):
    """Translate the fitted NumPy Forward-Forward regressor into an ONNX graph."""
    from onnx import TensorProto, helper

    nodes, inits = [], []
    inits.append(_const("eps", np.array([1e-6], dtype=np.float32)))
    cur = input_name
    blocks, layer_preds = [], []
    for i, layer in enumerate(model.layers):
        p = f"l{i}_"
        nodes.append(helper.make_node("ReduceL2", [cur], [p + "norm"], axes=[1], keepdims=1))
        nodes.append(helper.make_node("Max", [p + "norm", "eps"], [p + "den"]))
        nodes.append(helper.make_node("Div", [cur, p + "den"], [p + "xn"]))
        inits.append(_const(p + "W", layer.weight.astype(np.float32)))
        inits.append(_const(p + "b", layer.bias.astype(np.float32)))
        nodes.append(helper.make_node("MatMul", [p + "xn", p + "W"], [p + "z0"]))
        nodes.append(helper.make_node("Add", [p + "z0", p + "b"], [p + "z"]))
        nodes.append(helper.make_node("Relu", [p + "z"], [p + "h"]))
        inits.append(_const(p + "shape", np.array([-1, layer.groups, layer.nodes_per_group], dtype=np.int64)))
        nodes.append(helper.make_node("Reshape", [p + "h", p + "shape"], [p + "h3"]))
        nodes.append(helper.make_node("Mul", [p + "h3", p + "h3"], [p + "sq"]))
        nodes.append(helper.make_node("ReduceMean", [p + "sq"], [p + "good"], axes=[2], keepdims=0))
        nodes.append(helper.make_node("Softmax", [p + "good"], [p + "soft"], axis=1))
        inits.append(_const(p + "centers", np.linspace(-1.0, 1.0, layer.groups, dtype=np.float32)))
        nodes.append(helper.make_node("Mul", [p + "soft", p + "centers"], [p + "weighted"]))
        inits.append(_const(p + "ax1", np.array([1], dtype=np.int64)))
        nodes.append(helper.make_node("ReduceSum", [p + "weighted", p + "ax1"], [p + "lp"], keepdims=1))
        nodes.append(helper.make_node("Concat", [p + "good", p + "lp"], [p + "block"], axis=1))
        blocks.append(p + "block")
        layer_preds.append(p + "lp")
        cur = p + "h"

    nodes.append(helper.make_node("Concat", blocks, ["ffr_features"], axis=1))
    coef = model.readout_coef.astype(np.float32).reshape(-1)
    inits.append(_const("readout_w", coef[1:].reshape(-1, 1)))
    inits.append(_const("readout_b", coef[:1]))
    nodes.append(helper.make_node("MatMul", ["ffr_features", "readout_w"], ["pred2d"]))
    nodes.append(helper.make_node("Add", ["pred2d", "readout_b"], ["pred2d_b"]))
    inits.append(_const("squeeze_ax", np.array([1], dtype=np.int64)))
    nodes.append(helper.make_node("Squeeze", ["pred2d_b", "squeeze_ax"], [RESIDUAL_OUTPUTS[0]]))

    # Layer disagreement: population std of the unscaled per-layer predictions.
    nodes.append(helper.make_node("Concat", layer_preds, ["lp_all"], axis=1))
    inits.append(_const("t_scale", np.array([model.target_scale], dtype=np.float32)))
    inits.append(_const("t_center", np.array([model.target_center], dtype=np.float32)))
    nodes.append(helper.make_node("Mul", ["lp_all", "t_scale"], ["lp_scaled"]))
    nodes.append(helper.make_node("Add", ["lp_scaled", "t_center"], ["lp_unscaled"]))
    nodes.append(helper.make_node("ReduceMean", ["lp_unscaled"], ["lp_mean"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["lp_unscaled", "lp_mean"], ["lp_diff"]))
    nodes.append(helper.make_node("Mul", ["lp_diff", "lp_diff"], ["lp_diff2"]))
    nodes.append(helper.make_node("ReduceMean", ["lp_diff2"], ["lp_var"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("Sqrt", ["lp_var"], [RESIDUAL_OUTPUTS[3]]))
    inits.append(_const("q80", np.array([model.interval_q80], dtype=np.float32)))
    nodes.append(helper.make_node("Max", [RESIDUAL_OUTPUTS[3], "q80"], ["half_width"]))
    nodes.append(helper.make_node("Sub", [RESIDUAL_OUTPUTS[0], "half_width"], [RESIDUAL_OUTPUTS[1]]))
    nodes.append(helper.make_node("Add", [RESIDUAL_OUTPUTS[0], "half_width"], [RESIDUAL_OUTPUTS[2]]))

    graph = helper.make_graph(
        nodes,
        "raceshift_ffr",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, model.input_dim])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [None]) for name in RESIDUAL_OUTPUTS],
        initializer=inits,
        doc_string="RaceShift Forward-Forward regressor. Outputs are residuals against rolling_median_5 in seconds: next_lap = rolling_median_5 + residual_s.",
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="raceshift")
    onnx_model.ir_version = 8
    import onnx

    onnx.checker.check_model(onnx_model)
    return onnx_model


def verify_ffr_onnx(onnx_model, model: ForwardForwardRegressor, x: np.ndarray, atol: float = 2e-3) -> dict[str, float]:
    """Run the ONNX graph with onnxruntime and compare against the NumPy model on ``x``."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: np.asarray(x, dtype=np.float32)})
    expected = model.predict_with_uncertainty(np.asarray(x, dtype=np.float32))
    pairs = zip(outputs, [expected["prediction"], expected["lower_80"], expected["upper_80"], expected["layer_disagreement"]])
    diffs = {name: float(np.max(np.abs(got - want))) for name, (got, want) in zip(RESIDUAL_OUTPUTS, pairs)}
    worst = max(diffs.values())
    if worst > atol:
        raise AssertionError(f"ONNX export disagrees with the NumPy model by up to {worst:.4g} s: {diffs}")
    return diffs


def try_export_preprocessor(prep, contract: dict, output: Path) -> str | None:
    """Convert the sklearn preprocessor with skl2onnx. Returns a reason string when unsupported."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError as exc:  # pragma: no cover - optional dependency
        return f"skl2onnx not installed ({exc})"
    initial_types = [(c, FloatTensorType([None, 1])) for c in contract["numeric"]]
    initial_types += [(c, StringTensorType([None, 1])) for c in contract["categorical"]]
    try:
        onnx_prep = convert_sklearn(prep, initial_types=initial_types, target_opset=OPSET, options={"zipmap": False})
    except Exception as exc:  # noqa: BLE001 - report, do not fail the export
        return f"{type(exc).__name__}: {exc}"
    output.write_bytes(onnx_prep.SerializeToString())
    return None


def merge_end_to_end(prep_path: Path, ffr_path: Path, output: Path) -> str | None:
    import onnx
    from onnx import compose

    prep = onnx.load(str(prep_path))
    core = onnx.load(str(ffr_path))
    prep_out = [o.name for o in prep.graph.output]
    if len(prep_out) != 1:
        return f"preprocessor graph has {len(prep_out)} outputs, expected 1"
    try:
        merged = compose.merge_models(prep, core, io_map=[(prep_out[0], core.graph.input[0].name)])
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    onnx.checker.check_model(merged)
    onnx.save(merged, str(output))
    return None


def preprocessor_spec(prep, contract: dict) -> dict:
    """Serialise the fitted sklearn preprocessor to plain JSON so inference needs no pickle.

    Numeric columns: median imputation, a missing indicator for the columns that had gaps in
    training, then standard scaling over values + indicators. Categorical columns: missing
    values become the literal ``missing`` category, then one-hot with sklearn's infrequent
    handling (rare training categories share one ``__infrequent__`` column; unknown values
    produce all zeros).
    """
    steps = {name: (pipe, cols) for name, pipe, cols in prep.transformers_ if name != "remainder"}
    num_pipe, num_cols = steps["num"]
    cat_pipe, cat_cols = steps["cat"]
    imputer = num_pipe.named_steps["impute"]
    scaler = num_pipe.named_steps["scale"]
    onehot = cat_pipe.named_steps["onehot"]
    indicator_features = [int(i) for i in imputer.indicator_.features_] if imputer.indicator_ is not None else []
    categorical = []
    infrequent_all = getattr(onehot, "infrequent_categories_", [None] * len(cat_cols))
    for j, col in enumerate(cat_cols):
        infrequent = set() if infrequent_all[j] is None else {str(v) for v in infrequent_all[j]}
        frequent = [str(v) for v in onehot.categories_[j] if str(v) not in infrequent]
        categorical.append({"column": col, "frequent": frequent, "infrequent": sorted(infrequent)})
    return {
        "format": "raceshift-preprocessor-v1",
        "numeric_columns": list(num_cols),
        "numeric_medians": [float(v) for v in imputer.statistics_],
        "missing_indicator_features": indicator_features,
        "scale_mean": [float(v) for v in scaler.mean_],
        "scale_std": [float(v) for v in scaler.scale_],
        "categorical_fill": str(cat_pipe.named_steps["impute"].fill_value),
        "categorical": categorical,
        "output_dim": int(len(scaler.mean_) + sum(len(c["frequent"]) + (1 if c["infrequent"] else 0) for c in categorical)),
    }


def apply_preprocessor_spec(spec: dict, frame) -> np.ndarray:
    """Pure-NumPy re-implementation of the fitted preprocessor from :func:`preprocessor_spec`."""
    import pandas as pd

    values = frame[spec["numeric_columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    missing = np.isnan(values)
    filled = np.where(missing, np.asarray(spec["numeric_medians"])[None, :], values)
    blocks = [filled]
    if spec["missing_indicator_features"]:
        blocks.append(missing[:, spec["missing_indicator_features"]].astype(np.float64))
    numeric = np.concatenate(blocks, axis=1)
    numeric = (numeric - np.asarray(spec["scale_mean"])[None, :]) / np.asarray(spec["scale_std"])[None, :]
    cat_blocks = []
    for entry in spec["categorical"]:
        # Mirror sklearn exactly: only float NaN counts as missing for object columns, so a
        # Python None is a category of its own (its string form "None"), as it was in training.
        fill = spec["categorical_fill"]
        text = np.array([fill if (isinstance(v, float) and np.isnan(v)) else str(v) for v in frame[entry["column"]].to_numpy()], dtype=object)
        cols = [(text == cat).astype(np.float64) for cat in entry["frequent"]]
        if entry["infrequent"]:
            cols.append(np.isin(text, entry["infrequent"]).astype(np.float64))
        if cols:
            cat_blocks.append(np.stack(cols, axis=1))
    out = np.concatenate([numeric] + cat_blocks, axis=1) if cat_blocks else numeric
    return out.astype(np.float32)


def _fmt(v, d=3):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))


def model_card(artifact: Path, metrics: dict, config: dict, contract: dict, export_notes: dict) -> str:
    name = metrics.get("name", artifact.name)
    arch = " → ".join(str(n) for n in config["config"]["layer_nodes"])
    groups = " / ".join(str(g) for g in config["config"]["ordinal_groups"])
    sp = metrics.get("split", {})
    rows = metrics.get("rows", {})
    res = metrics.get("resources", {})
    tr = res.get("training", {})
    lines = [
        f"# {name} model card", "",
        f"RaceShift Forward-Forward regressor exported from `artifacts/{artifact.name}` on {metrics.get('created_utc', 'unknown date')}.",
        "", "## What it predicts", "",
        "The next lap time of a Formula 1 driver from information known at the end of the current lap. The network predicts the",
        "residual against the driver's rolling five-lap median (`rolling_median_5`); `next_lap = rolling_median_5 + residual`. It",
        "also returns an 80% interval (validation-residual quantile widened by cross-layer disagreement).",
        "", "## Architecture", "",
        f"- Hidden layers: {arch} nodes, ordinal groups {groups}; each layer trained with a local ordinal-goodness objective and an explicit local Adam update. No global backpropagation.",
        f"- Readout: closed-form ridge over all layers' goodness vectors and local predictions (alpha {config['config'].get('ridge_alpha')}).",
        f"- Input: {len(contract['numeric'])} numeric and {len(contract['categorical'])} categorical contract columns after train-only imputation, scaling and one-hot encoding ({config['input_dim']} model inputs).",
        "", "## Training data and split", "",
        f"- Data source: `{metrics.get('data_source')}`" + (" (synthetic fixture, not Formula 1 results)" if metrics.get("is_synthetic") else ""),
        f"- Split mode: {sp.get('mode')}; train ≤ {sp.get('train_end')}, validation {sp.get('validation')}, test {sp.get('test')}"
        + (f", split round {sp.get('split_round')}" if sp.get("split_round") else "") + (f", held-out event {sp.get('holdout_event')}" if sp.get("holdout_event") else ""),
        f"- Rows: train {rows.get('train')}, validation {rows.get('validation')}, test {rows.get('test')}",
        f"- Dropped sparse features: {len(contract.get('dropped_sparse', []))}; dropped groups: {contract.get('dropped_groups') or 'none'}",
        "", "## Measured performance", "",
        "| Split | MAE (s) | RMSE (s) | p90 (s) | Laps within 0.5 s | 80% coverage |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "test"):
        m = metrics.get(split)
        if not m:
            continue
        share = m.get("within_0_5s_share")
        lines.append(f"| {split} | {_fmt(m.get('mae_s'))} | {_fmt(m.get('rmse_s'))} | {_fmt(m.get('p90_ae_s'))} | {('%.1f%%' % (100 * share)) if share is not None else '—'} | {_fmt(m.get('interval80_coverage'))} |")
    lines += [
        "", "## Resources", "",
        f"- Training wall time {_fmt(tr.get('wall_seconds'), 1)} s, peak RSS {_fmt(tr.get('peak_rss_mb'), 0)} MB, traced training peak {_fmt(tr.get('peak_traced_mb'), 0)} MB, device {res.get('device', 'cpu')}",
        f"- Inference {_fmt(res.get('inference_single_row_ms'), 3)} ms per single row, artifact {_fmt((res.get('artifact_bytes') or 0) / 1e6, 2)} MB",
        "", "## Files in this export", "",
    ]
    for key, note in export_notes.items():
        lines.append(f"- `{key}`: {note}")
    lines += [
        "", "## Limitations", "",
        "- Laps around red-flag stoppages can pass the validity rules and produce very large errors (documented gap).",
        "- Intervals are calibrated on the validation season; under regulation change (2026) coverage drops below the nominal 80%.",
        "- Historical priors need earlier events in the same table; a single-race file yields missing priors, which the model treats as their own category.",
        "- The model is a research artifact for comparing local Forward-Forward learning against baselines; the gradient-boosted tree baseline is more accurate on the same data.",
        "", "## Inference", "",
        "```python", "import json, joblib, numpy as np, onnxruntime as ort, pandas as pd",
        "from raceshift.features.full_context import build_full_context_table",
        f"art = 'artifacts/{artifact.name}'",
        "contract = json.load(open(f'{art}/feature_contract.json'))",
        "table = build_full_context_table(pd.read_parquet('laps.parquet'), history=contract['history'])",
        "x = joblib.load(f'{art}/preprocessor.joblib').transform(table[contract['numeric'] + contract['categorical']]).astype(np.float32)",
        f"sess = ort.InferenceSession(f'{{art}}/export/{name.lower().replace(' ', '_')}_ffr.onnx')",
        "residual, lower, upper, disagreement = sess.run(None, {'features': x})",
        "next_lap = table['rolling_median_5'].to_numpy() + residual",
        "```",
    ]
    return "\n".join(lines) + "\n"


def export_artifact(artifact_dir: str | Path, verify_rows: np.ndarray | None = None, bundle_dir: str | Path | None = None, verify_table=None) -> dict:
    """Export one artifact. Writes into ``<artifact>/export/`` and optionally a zip bundle.

    ``verify_rows`` are preprocessed feature rows used to check the ONNX core against NumPy;
    when omitted, random rows in the standardized range are used. ``verify_table`` is an
    optional raw frame of contract columns used to check the JSON preprocessor spec against
    the sklearn preprocessor.
    """
    import onnx

    artifact = Path(artifact_dir)
    metrics = json.loads((artifact / "metrics.json").read_text())
    config = json.loads((artifact / "model_config.json").read_text())
    contract = json.loads((artifact / "feature_contract.json").read_text())
    model = ForwardForwardRegressor.load(artifact)
    stem = str(metrics.get("name", artifact.name)).lower().replace(" ", "_")
    out = artifact / "export"
    out.mkdir(parents=True, exist_ok=True)

    core = build_ffr_onnx(model)
    if verify_rows is None:
        verify_rows = np.random.default_rng(0).normal(size=(64, model.input_dim)).astype(np.float32)
    diffs = verify_ffr_onnx(core, model, verify_rows)
    ffr_path = out / f"{stem}_ffr.onnx"
    onnx.save(core, str(ffr_path))
    notes = {ffr_path.name: f"Forward-Forward core, opset {OPSET}; verified against NumPy on {len(verify_rows)} rows, max |Δ| {max(diffs.values()):.2e} s"}

    import joblib

    prep = joblib.load(artifact / "preprocessor.joblib")
    prep_path = out / f"{stem}_preprocessor.onnx"
    reason = try_export_preprocessor(prep, contract, prep_path)
    end_to_end = None
    skl2onnx_reason = None
    if reason is None:
        notes[prep_path.name] = "train-only preprocessor (imputation, scaling, one-hot) converted with skl2onnx"
        e2e_path = out / f"{stem}_end_to_end.onnx"
        merge_reason = merge_end_to_end(prep_path, ffr_path, e2e_path)
        if merge_reason is None:
            notes[e2e_path.name] = "preprocessor + core in one graph: raw contract columns in, residual and interval out"
            end_to_end = e2e_path
        else:
            notes["end_to_end"] = f"not produced: {merge_reason}"
    else:
        notes["preprocessor.joblib"] = (
            "fitted sklearn preprocessor (joblib). skl2onnx cannot convert SimpleImputer(add_indicator=True), so there is no "
            f"preprocessor ONNX; use {stem}_preprocessor.json with raceshift.models.export.apply_preprocessor_spec for pickle-free inference"
        )
        skl2onnx_reason = reason
    spec = preprocessor_spec(prep, contract)
    spec_path = out / f"{stem}_preprocessor.json"
    spec_note = "fitted preprocessor as plain JSON (medians, indicators, scaling, one-hot vocabularies) for pickle-free inference via raceshift.models.export.apply_preprocessor_spec"
    if verify_table is not None:
        cols = contract["numeric"] + contract["categorical"]
        expected = np.asarray(prep.transform(verify_table[cols]), dtype=np.float32)
        got = apply_preprocessor_spec(spec, verify_table[cols])
        spec_diff = float(np.max(np.abs(got - expected))) if got.shape == expected.shape else float("inf")
        if spec_diff > 1e-4:
            raise AssertionError(f"JSON preprocessor spec disagrees with sklearn (shape {got.shape} vs {expected.shape}, max |Δ| {spec_diff:.3g})")
        spec_note += f"; verified against sklearn on {len(verify_table)} rows, max |Δ| {spec_diff:.2e}"
    spec_path.write_text(json.dumps(spec, indent=1))
    notes[spec_path.name] = spec_note
    notes["feature_contract.json"] = "exact input columns, history length, dropped features"
    notes["metrics.json"] = "all measured numbers for this run"

    card = model_card(artifact, metrics, config, contract, notes)
    (out / "MODEL_CARD.md").write_text(card)
    (out / "export_manifest.json").write_text(json.dumps({"artifact": artifact.name, "name": metrics.get("name"), "files": notes, "onnx_verification_max_abs_diff_s": diffs, "skl2onnx_error": skl2onnx_reason}, indent=2))

    bundle = None
    if bundle_dir is not None:
        bundle_root = Path(bundle_dir)
        bundle_root.mkdir(parents=True, exist_ok=True)
        bundle = bundle_root / f"{artifact.name}.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(artifact.rglob("*")):
                if path.is_file() and path.suffix != ".zip":
                    zf.write(path, arcname=str(Path(artifact.name) / path.relative_to(artifact)))
    return {"export_dir": out, "ffr_onnx": ffr_path, "preprocessor_onnx": prep_path if reason is None else None, "end_to_end_onnx": end_to_end, "bundle": bundle, "notes": notes, "verification": diffs}


def remove_export(artifact_dir: str | Path) -> None:
    shutil.rmtree(Path(artifact_dir) / "export", ignore_errors=True)
