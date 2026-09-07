#!/usr/bin/env python
"""Inject experiment tables from reports/<name>/summary.json into README.md.

Usage:
    python scripts/update_readme_results.py f1_2025h2 holdout_monza domain_shift_2026

The README must contain the markers <!-- RESULTS:BEGIN --> and <!-- RESULTS:END -->.
Only measured numbers from summary.json are written; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def fmt(v, d=3):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))


def table(summary: dict, title: str) -> list[str]:
    sp = summary.get("split") or {}
    rows = summary.get("rows") or {}
    mode = sp.get("mode", "season_forward")
    if sp.get("holdout_event"):
        split_text = f"every season of **{sp['holdout_event']}** held out; train ≤ {sp.get('train_end')}, validation {sp.get('validation')}"
    elif mode == "season_round":
        split_text = f"train ≤ {sp.get('train_end')} · validation {sp.get('validation')} rounds ≤ {sp.get('split_round')} · test {sp.get('test')} rounds > {sp.get('split_round')}"
    else:
        split_text = f"train ≤ {sp.get('train_end')} · validation {sp.get('validation')} · test {sp.get('test')}"
    lines = [f"**{title}** — {split_text}. Rows: train {rows.get('train')}, validation {rows.get('validation')}, test {rows.get('test')}. Data: {summary.get('data_source')}.", ""]
    lines.append("| Model | Test MAE (s) | Test RMSE (s) | p90 (s) | 80% coverage | Train time (s) | Peak RSS (MB) | Traced train peak (MB) | Artifact (MB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    best = min((r["test"]["mae_s"] for r in summary["results"] if r.get("test")), default=None)
    for r in summary["results"]:
        t = r.get("test") or {}
        res = r.get("resources") or {}
        tr = res.get("training") or {}
        art = res.get("artifact_bytes")
        name = r["model"]
        if best is not None and t.get("mae_s") == best:
            name = f"**{name}**"
        cov = t.get("interval80_coverage")
        lines.append(
            f"| {name} | {fmt(t.get('mae_s'))} | {fmt(t.get('rmse_s'))} | {fmt(t.get('p90_ae_s'))} | "
            f"{fmt(cov) if cov is not None else '—'} | {fmt(tr.get('wall_seconds'), 1) if tr.get('wall_seconds') else '—'} | "
            f"{fmt(tr.get('peak_rss_mb'), 0) if tr.get('peak_rss_mb') else '—'} | {fmt(tr.get('peak_traced_mb'), 0) if tr.get('peak_traced_mb') else '—'} | {fmt(art / 1e6, 2) if art else '—'} |"
        )
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("reports", nargs="+", help="report names under reports/")
    p.add_argument("--titles", nargs="*", default=[], help="optional titles, one per report")
    args = p.parse_args()
    readme = ROOT / "README.md"
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit("README.md is missing the RESULTS markers")
    blocks: list[str] = []
    for i, name in enumerate(args.reports):
        summary = json.loads((ROOT / "reports" / name / "summary.json").read_text())
        title = args.titles[i] if i < len(args.titles) else name
        blocks += table(summary, title)
        blocks.append(f"Full table with validation metrics, interval widths and latency: `reports/{name}/summary.md`.")
        blocks.append("")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + BEGIN + "\n" + "\n".join(blocks) + END + tail)
    print(f"README results updated from {len(args.reports)} report(s)")


if __name__ == "__main__":
    main()
