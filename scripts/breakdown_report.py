#!/usr/bin/env python
"""Error breakdowns by race condition for every model in an experiment.

Joins each artifact's test_predictions.csv (FFR runs and the baselines run) with the
feature table rebuilt from the raw laps, then reports MAE per circuit, team, compound,
tyre-age band, wet/dry, race phase, running position and consecutive-clean-lap depth.
Every model is scored on exactly the same test laps, so the columns are comparable.

Usage:
    python scripts/breakdown_report.py --input data/processed/f1_laps_fastf1.parquet \
        --report f1_2025h2 --artifacts artifacts/f1_2025h2_baselines artifacts/f1_2025h2_ffr-m ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raceshift.features.full_context import build_full_context_table  # noqa: E402

KEYS = ["season", "round_number", "event", "session", "driver", "lap_number"]
CONTEXT = ["circuit", "team", "compound", "tyre_life", "rainfall", "position", "laps_in_segment", "track_temp_c", "data_tier"]
MIN_ROWS = 30


def load_predictions(artifact: Path) -> dict[str, pd.DataFrame]:
    """Return {model_name: frame[KEYS + actual + pred]} for one artifact directory."""
    path = artifact / "test_predictions.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    metrics = json.loads((artifact / "metrics.json").read_text())
    out: dict[str, pd.DataFrame] = {}
    if "predicted_next_lap_s" in frame.columns:
        name = metrics.get("name", artifact.name)
        out[name] = frame[KEYS + ["actual_next_lap_s"]].assign(pred=frame["predicted_next_lap_s"])
    for col in [c for c in frame.columns if c.startswith("pred_")]:
        out[col[len("pred_"):]] = frame[KEYS + ["actual_next_lap_s"]].assign(pred=frame[col])
    return out


def bands(table: pd.DataFrame) -> pd.DataFrame:
    ctx = table[KEYS + [c for c in CONTEXT if c in table.columns]].copy()
    tyre = pd.to_numeric(ctx.get("tyre_life"), errors="coerce")
    ctx["tyre_age"] = pd.cut(tyre, [0, 5, 15, 30, 999], labels=["1-5 laps", "6-15 laps", "16-30 laps", "31+ laps"]).astype(str).replace("nan", "unknown")
    rain = ctx.get("rainfall")
    ctx["conditions"] = np.where(rain.astype(str).str.lower().isin(["true", "1", "1.0"]), "wet", "dry") if rain is not None else "unknown"
    pos = pd.to_numeric(ctx.get("position"), errors="coerce")
    ctx["running_position"] = pd.cut(pos, [0, 3, 10, 99], labels=["P1-P3", "P4-P10", "P11+"]).astype(str).replace("nan", "unknown")
    depth = pd.to_numeric(ctx.get("laps_in_segment"), errors="coerce")
    ctx["clean_lap_depth"] = pd.cut(depth, [0, 1, 4, 10, 999], labels=["1 (first clean lap)", "2-4", "5-10", "11+"]).astype(str).replace("nan", "unknown")
    lap = pd.to_numeric(ctx["lap_number"], errors="coerce")
    race_len = lap.groupby([ctx["season"], ctx["event"]]).transform("max")
    ctx["race_phase"] = pd.cut(lap / race_len, [0, 0.33, 0.66, 1.0], labels=["opening third", "middle third", "final third"]).astype(str)
    return ctx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="raw lap table used for the experiment")
    p.add_argument("--report", required=True, help="report name under reports/")
    p.add_argument("--artifacts", nargs="+", required=True)
    p.add_argument("--history", type=int, default=5)
    p.add_argument("--reports-dir", default=str(ROOT / "reports"))
    args = p.parse_args()

    raw = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    table = build_full_context_table(raw, history=args.history)
    ctx = bands(table)

    models: dict[str, pd.DataFrame] = {}
    for a in args.artifacts:
        models.update(load_predictions(Path(a)))
    if not models:
        raise SystemExit("no test_predictions.csv found in the given artifacts")

    # Align every model on the same laps.
    merged = None
    for name, frame in models.items():
        f = frame.rename(columns={"pred": f"pred::{name}"})
        merged = f if merged is None else merged.merge(f.drop(columns=["actual_next_lap_s"]), on=KEYS, how="inner")
    merged = merged.merge(ctx, on=KEYS, how="left")
    names = list(models)
    for name in names:
        merged[f"ae::{name}"] = (merged[f"pred::{name}"] - merged["actual_next_lap_s"]).abs()

    dims = [("circuit", "Circuit"), ("team", "Constructor"), ("compound", "Compound"), ("tyre_age", "Tyre age"),
            ("conditions", "Conditions"), ("race_phase", "Race phase"), ("running_position", "Running position"),
            ("clean_lap_depth", "Consecutive clean laps"), ("data_tier", "Data tier")]
    lines = [f"# Error breakdowns: {args.report}", "",
             f"{len(merged)} test laps shared by {len(names)} models. MAE in seconds; groups with fewer than {MIN_ROWS} laps are omitted. Best model per row in bold.", ""]
    summary: dict[str, dict] = {}
    for col, title in dims:
        if col not in merged.columns or merged[col].nunique() < 2:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| " + title + " | laps | " + " | ".join(names) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in names) + " |")
        summary[col] = {}
        for key, group in merged.groupby(col, sort=True, dropna=False):
            if len(group) < MIN_ROWS:
                continue
            maes = {n: float(group[f"ae::{n}"].mean()) for n in names}
            best = min(maes.values())
            cells = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in maes.values()]
            lines.append(f"| {key} | {len(group)} | " + " | ".join(cells) + " |")
            summary[col][str(key)] = {"rows": int(len(group)), **{n: round(v, 4) for n, v in maes.items()}}
        lines.append("")

    report_dir = Path(args.reports_dir) / args.report
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "breakdowns.md").write_text("\n".join(lines) + "\n")
    (report_dir / "breakdowns.json").write_text(json.dumps({"models": names, "rows": int(len(merged)), "breakdowns": summary}, indent=2))
    print("\n".join(lines[:40]))
    print(f"... written to {report_dir / 'breakdowns.md'}")


if __name__ == "__main__":
    main()
