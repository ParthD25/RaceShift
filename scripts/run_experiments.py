#!/usr/bin/env python
"""Run the RaceShift experiment matrix and write a results table.

Runs the baselines and every requested FFR config as separate processes (so peak memory
is measured per run), then aggregates each metrics.json into reports/<name>/summary.json
and summary.md. Everything is chronological: the same split arguments are passed to every
run.

Example:
    python scripts/run_experiments.py --input data/processed/f1_laps.parquet \
        --name f1_2025h2 --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12 \
        --ffr configs/ffr_small.json configs/ffr_production.json configs/ffr_colab_large.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], out: Path | None = None, resume: bool = False) -> None:
    """Run one training process; with ``resume`` a finished run (metrics.json present) is skipped."""
    if resume and out is not None and (out / "metrics.json").exists():
        print(f"= reusing finished run {out}", flush=True)
        return
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    p = argparse.ArgumentParser(description="Run baselines and FFR configs, then summarise.")
    p.add_argument("--input", required=True)
    p.add_argument("--name", required=True, help="Experiment name, used for artifacts/<name>_* and reports/<name>/")
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--val-year", type=int, required=True)
    p.add_argument("--test-year", type=int, required=True)
    p.add_argument("--split-round", type=int)
    p.add_argument("--holdout-event")
    p.add_argument("--ffr", nargs="*", default=[str(ROOT / "configs" / "ffr_production.json")], help="FFR config files")
    p.add_argument("--ablate", nargs="*", default=[], help="Feature groups to ablate one at a time with the first FFR config")
    p.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    p.add_argument("--reports-dir", default=str(ROOT / "reports"))
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip runs whose metrics.json already exists (restart-safe)")
    p.add_argument("--data-source")
    args = p.parse_args()

    py = sys.executable
    split_args = ["--train-end", str(args.train_end), "--val-year", str(args.val_year), "--test-year", str(args.test_year)]
    if args.split_round is not None:
        split_args += ["--split-round", str(args.split_round)]
    if args.holdout_event:
        split_args += ["--holdout-event", args.holdout_event]
    if args.data_source:
        split_args += ["--data-source", args.data_source]

    artifacts = Path(args.artifacts_dir)
    runs: list[tuple[str, Path]] = []
    if not args.skip_baselines:
        out = artifacts / f"{args.name}_baselines"
        run([py, "scripts/train_baselines.py", "--input", args.input, "--output", str(out), *split_args], out, args.resume)
        runs.append(("baselines", out))
    for cfg in args.ffr:
        cfg_name = json.loads(Path(cfg).read_text()).get("name", Path(cfg).stem)
        out = artifacts / f"{args.name}_{cfg_name.lower().replace(' ', '_')}"
        run([py, "scripts/train_ffr.py", "--input", args.input, "--config", cfg, "--output", str(out), *split_args], out, args.resume)
        runs.append((cfg_name, out))
    for group in args.ablate:
        cfg = args.ffr[0]
        cfg_name = json.loads(Path(cfg).read_text()).get("name", Path(cfg).stem)
        label = f"{cfg_name} minus {group}"
        out = artifacts / f"{args.name}_{cfg_name.lower()}_minus_{group}"
        run([py, "scripts/train_ffr.py", "--input", args.input, "--config", cfg, "--output", str(out), "--drop-feature-group", group, "--name", label, *split_args], out, args.resume)
        runs.append((label, out))

    # Aggregate.
    rows = []
    for label, out in runs:
        metrics = json.loads((out / "metrics.json").read_text())
        if "models" in metrics:
            for name, result in metrics["models"].items():
                rows.append({
                    "model": name,
                    "family": "baseline",
                    "validation": result["validation"],
                    "test": result["test"],
                    "resources": result.get("resources", {}),
                    "artifact": out.name,
                })
        else:
            rows.append({
                "model": metrics.get("name", label),
                "family": "forward-forward",
                "architecture": metrics.get("architecture"),
                "validation": metrics["validation"],
                "test": metrics["test"],
                "resources": metrics.get("resources", {}),
                "features": metrics.get("features"),
                "test_by_event": metrics.get("test_by_event"),
                "artifact": out.name,
            })
    first = json.loads((runs[0][1] / "metrics.json").read_text())
    summary = {
        "name": args.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": Path(args.input).name,
        "data_source": first.get("data_source"),
        "is_synthetic": first.get("is_synthetic"),
        "split": first.get("split"),
        "rows": first.get("rows"),
        "results": rows,
    }
    report_dir = Path(args.reports_dir) / args.name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# RaceShift experiment: {args.name}", ""]
    lines.append(f"Generated {summary['created_utc']} from `{summary['input_file']}` (data source: {summary['data_source']}).")
    sp = summary["split"] or {}
    lines.append(f"Split: {sp.get('mode')} · train ≤ {sp.get('train_end')} · validation {sp.get('validation')} · test {sp.get('test')}"
                 + (f" · split round {sp.get('split_round')}" if sp.get("split_round") else "")
                 + (f" · holdout {sp.get('holdout_event')}" if sp.get("holdout_event") else ""))
    r = summary["rows"] or {}
    lines.append(f"Rows: train {r.get('train')} · validation {r.get('validation')} · test {r.get('test')}")
    lines.append("")
    lines.append("| Model | Val MAE (s) | Test MAE (s) | Test RMSE (s) | Test p90 (s) | 80% coverage | Interval width (s) | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        t = row["test"]; v = row["validation"]; res = row.get("resources", {}); tr = res.get("training", {})
        art = res.get("artifact_bytes")
        lines.append(
            f"| {row['model']} | {fmt(v.get('mae_s'))} | {fmt(t.get('mae_s'))} | {fmt(t.get('rmse_s'))} | {fmt(t.get('p90_ae_s'))} | "
            f"{fmt(t.get('interval80_coverage'), 3) if t.get('interval80_coverage') is not None else '—'} | {fmt(t.get('interval80_width_s'))} | "
            f"{fmt(tr.get('wall_seconds'), 1)} | {fmt(tr.get('peak_rss_mb'), 0)} | {fmt(tr.get('peak_traced_mb'), 0)} | {fmt(art / 1e6, 2) if art else '—'} |"
        )
    lines.append("")
    lines.append("MAE, RMSE and p90 are absolute errors on the true next lap time in seconds. Coverage is the share of test laps inside the 80% interval (baselines have no interval). Peak RSS is the process high-water mark, so it includes data loading; the traced peak is Python-allocated memory during the fit only (tracemalloc), the closer proxy for training-memory requirements.")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
