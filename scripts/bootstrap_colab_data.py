#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(cmd:list[str]):
    print("+"," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True,cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description="Reproducible RaceShift Colab data bootstrap.")
    p.add_argument("--mode",choices=["smoke","core","research"],default="core")
    args=p.parse_args()
    if args.mode=="smoke":
        years=[2024]; events=["Monza"]
    else:
        years=[2022,2023,2024,2025]; events=["Bahrain","Silverstone","Monza"]
    run([sys.executable,"scripts/fetch_fastf1_seasons.py","--years",*map(str,years),"--events",*events,"--session","R"])
    run([sys.executable,"scripts/build_lap_dataset.py"])
    if args.mode=="research":
        # Large datasets are opt-in and remain outside git.
        run([sys.executable,"scripts/fetch_hf.py","FlorindoDev/f1_corner_telemetry_2024_2025","--list"])
        run([sys.executable,"scripts/fetch_hf.py","VforVitorio/f1-strategy-dataset","--list"])
        run([sys.executable,"scripts/fetch_hf.py","tobil/imsa","--list"])

if __name__=="__main__": main()
