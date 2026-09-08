#!/usr/bin/env python
"""Export a trained RaceShift artifact to ONNX plus a model card and a downloadable bundle.

Usage:
    python scripts/export_model.py artifacts/f1_2025h2_ffr-m --verify-input data/imports/f1_2025_season.parquet
    python scripts/export_model.py artifacts/raceshift_ffr_demo --bundle-dir exports

Writes <artifact>/export/{<name>_ffr.onnx, <name>_preprocessor.onnx, <name>_end_to_end.onnx,
MODEL_CARD.md, export_manifest.json} and, with --bundle-dir, <bundle-dir>/<artifact>.zip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.features.full_context import build_full_context_table  # noqa: E402
from raceshift.models.export import export_artifact  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("artifact", help="artifact directory, e.g. artifacts/f1_2025h2_ffr-m")
    p.add_argument("--verify-input", help="lap table used to verify the ONNX graph on real preprocessed rows (default: random rows)")
    p.add_argument("--bundle-dir", default=str(ROOT / "exports"), help="where to write <artifact>.zip; pass '' to skip")
    args = p.parse_args()

    artifact = Path(args.artifact)
    verify_rows = None
    verify_table = None
    if args.verify_input:
        contract = json.loads((artifact / "feature_contract.json").read_text())
        raw = pd.read_parquet(args.verify_input) if args.verify_input.endswith(".parquet") else pd.read_csv(args.verify_input)
        table = build_full_context_table(raw, history=int(contract["history"]))
        prep = joblib.load(artifact / "preprocessor.joblib")
        verify_table = table[contract["numeric"] + contract["categorical"]].head(2000)
        verify_rows = np.asarray(prep.transform(verify_table), dtype=np.float32)

    result = export_artifact(artifact, verify_rows=verify_rows, bundle_dir=args.bundle_dir or None, verify_table=verify_table)
    for key, note in result["notes"].items():
        print(f"{key}: {note}")
    if result["bundle"]:
        print(f"bundle: {result['bundle']} ({result['bundle'].stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
