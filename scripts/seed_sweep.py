#!/usr/bin/env python
"""Train one FFR config under several seeds and report the spread of its test error.

    python scripts/seed_sweep.py --input data/processed/f1_laps_fastf1.parquet --name f1_2025h2 \
        --config configs/ffr_small.json --seeds 1 2 3 \
        --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12

Writes artifacts/<name>_<config>_seed<k>/ for every seed (resumable: a run with metrics.json is
skipped) and reports/<name>/seeds.{md,json} with mean and standard deviation of the test MAE,
RMSE, tolerance shares and interval coverage across seeds plus the seed-42 headline run when
artifacts/<name>_<config>/ exists. The spread is the yardstick for deciding whether a difference
between two FFR variants in the README tables is an effect or noise.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = [("mae_s", "Test MAE (s)"), ("rmse_s", "Test RMSE (s)"), ("p90_ae_s", "p90 (s)"), ("within_0_5s_share", "Within 0.5 s"), ("interval80_coverage", "80% coverage")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--name", required=True, help="experiment name (artifacts/<name>_* and reports/<name>/)")
    p.add_argument("--config", default="configs/ffr_small.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--val-year", type=int, required=True)
    p.add_argument("--test-year", type=int, required=True)
    p.add_argument("--split-round", type=int)
    p.add_argument("--holdout-event")
    p.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    p.add_argument("--reports-dir", default=str(ROOT / "reports"))
    args = p.parse_args()

    config_path = Path(args.config)
    base_cfg = json.loads(config_path.read_text())
    stem = config_path.stem.replace("ffr_", "ffr-").replace("production", "m").replace("small", "s").replace("colab_large", "l")
    split_args = ["--train-end", str(args.train_end), "--val-year", str(args.val_year), "--test-year", str(args.test_year)]
    if args.split_round is not None:
        split_args += ["--split-round", str(args.split_round)]
    if args.holdout_event:
        split_args += ["--holdout-event", args.holdout_event]

    artifacts = Path(args.artifacts_dir)
    runs: dict[str, dict] = {}
    headline = artifacts / f"{args.name}_{stem}" / "metrics.json"
    if headline.is_file():
        runs["seed 42 (headline)"] = json.loads(headline.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            out = artifacts / f"{args.name}_{stem}_seed{seed}"
            metrics_file = out / "metrics.json"
            if not metrics_file.is_file():
                cfg = dict(base_cfg)
                cfg["seed"] = int(seed)
                cfg_file = Path(tmp) / f"{config_path.stem}_seed{seed}.json"
                cfg_file.write_text(json.dumps(cfg, indent=2))
                cmd = [sys.executable, str(ROOT / "scripts" / "train_ffr.py"), "--input", args.input, "--config", str(cfg_file), "--output", str(out), "--name", f"{base_cfg.get('name', stem)}-seed{seed}", *split_args]
                print("+", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True, cwd=ROOT)
            runs[f"seed {seed}"] = json.loads(metrics_file.read_text())

    rows = []
    for label, metrics in runs.items():
        test = metrics.get("test", {})
        rows.append({"run": label, **{k: test.get(k) for k, _ in METRICS}, "train_seconds": (metrics.get("resources") or {}).get("training", {}).get("wall_seconds")})
    swept = [r for r in rows if not r["run"].startswith("seed 42")]
    spread = {}
    for key, _ in METRICS:
        values = [r[key] for r in swept if r[key] is not None]
        spread[key] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values) if len(values) > 1 else 0.0, "n": len(values)} if values else None

    report_dir = Path(args.reports_dir) / args.name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "seeds.json").write_text(json.dumps({"config": config_path.name, "runs": rows, "spread": spread}, indent=2))
    lines = [f"# Seed sensitivity: {args.name} / {config_path.name}", "", "Same data, split and hyperparameters; only the random seed differs (weight initialisation and batch order).", "", "| Run | " + " | ".join(t for _, t in METRICS) + " | Train (s) |", "| --- | " + " | ".join("---:" for _ in METRICS) + " | ---: |"]
    for r in rows:
        cells = []
        for k, _ in METRICS:
            v = r[k]
            cells.append("—" if v is None else (f"{v * 100:.1f}%" if k == "within_0_5s_share" else f"{v:.3f}"))
        lines.append(f"| {r['run']} | " + " | ".join(cells) + f" | {r['train_seconds']:.0f} |" if r["train_seconds"] else f"| {r['run']} | " + " | ".join(cells) + " | — |")
    lines += ["", "**Spread across the swept seeds** (mean ± population std):", ""]
    for k, title in METRICS:
        s = spread.get(k)
        if s:
            lines.append(f"- {title}: {s['mean']:.3f} ± {s['std']:.3f} (n = {s['n']})")
    lines += ["", "A difference between two FFR variants smaller than about two standard deviations of the test MAE here should be read as noise, not as an effect."]
    (report_dir / "seeds.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
