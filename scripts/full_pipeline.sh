#!/usr/bin/env bash
# RaceShift unattended data + experiment pipeline.
#
# Two data tiers:
#   fastf1_timing   2018-today, every round, every team: laps, sectors, tyres, track status, weather
#   legacy_timing   2000-2017 (Jolpica/Ergast): lap times, positions, pit stops from 2011; no sectors/tyres/weather
#
# Stages (each is idempotent and resumable):
#   fastf1              collect every FastF1 race session (skips existing, waits out rate limits)
#   legacy              collect every Jolpica race (skips existing, stays under the hourly budget)
#   build-fastf1        concat FastF1 parquets -> $PROC/f1_laps_fastf1.parquet
#   build-all           concat both tiers      -> $PROC/f1_laps_all_tiers.parquet
#   experiments         main matrix + circuit holdout + 2026 domain shift on the FastF1 tier, README tables
#   legacy-experiments  same 2025 split, training set extended back to 2000 with the legacy tier
#
# Paths come from the environment so the same script runs locally, in Colab, or in CI:
#   RAW_FF, RAW_LEGACY, CACHE_FF, PROC, ARTIFACTS, REPORTS
set -uo pipefail
cd "$(dirname "$0")/.."

RAW_FF=${RAW_FF:-data/raw/fastf1}
RAW_LEGACY=${RAW_LEGACY:-data/raw/jolpica}
CACHE_FF=${CACHE_FF:-data/cache/fastf1}
PROC=${PROC:-data/processed}
ARTIFACTS=${ARTIFACTS:-artifacts}
REPORTS=${REPORTS:-reports}
FASTF1_YEARS=${FASTF1_YEARS:-2018-2026}
LEGACY_YEARS=${LEGACY_YEARS:-2000-2017}
PY=${PY:-python}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

concat() {  # concat <glob-dir>... -> <output>
  local out=${!#}
  local dirs=("${@:1:$#-1}")
  mkdir -p "$(dirname "$out")"
  DIRS="${dirs[*]}" OUT="$out" "$PY" - <<'EOF'
import glob, os
import pandas as pd
files = sorted(f for d in os.environ["DIRS"].split() for f in glob.glob(os.path.join(d, "*.parquet")))
frames = [pd.read_parquet(f) for f in files]
laps = pd.concat(frames, ignore_index=True)
laps.to_parquet(os.environ["OUT"], index=False)
tiers = laps.get("data_tier", pd.Series(["fastf1_timing"] * len(laps))).value_counts().to_dict()
print(f"{len(files)} sessions, {len(laps)} laps -> {os.environ['OUT']} tiers={tiers}")
print(laps.groupby("season")["event"].nunique().to_dict())
EOF
}

stage_fastf1() {
  log "FastF1 collection $FASTF1_YEARS -> $RAW_FF"
  "$PY" scripts/fetch_fastf1_seasons.py --years "$FASTF1_YEARS" --session R --skip-existing \
    --output "$RAW_FF" --cache "$CACHE_FF"
}

stage_legacy() {
  log "Legacy collection $LEGACY_YEARS -> $RAW_LEGACY"
  "$PY" scripts/fetch_jolpica_seasons.py --years "$LEGACY_YEARS" --output "$RAW_LEGACY"
}

stage_build_fastf1() { concat "$RAW_FF" "$PROC/f1_laps_fastf1.parquet"; }
stage_build_all()    { concat "$RAW_FF" "$RAW_LEGACY" "$PROC/f1_laps_all_tiers.parquet"; }

stage_experiments() {
  local input=$PROC/f1_laps_fastf1.parquet
  log "Main matrix: train <=2024, validation 2025 rounds <=12, test 2025 rounds >12"
  "$PY" scripts/run_experiments.py --input "$input" --name f1_2025h2 \
    --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12 \
    --ffr configs/ffr_production.json configs/ffr_small.json configs/ffr_colab_large.json \
          configs/ffr_m_groups_coarse.json configs/ffr_m_groups_fine.json \
    --ablate historical_numeric temporal_numeric static_categorical \
    --artifacts-dir "$ARTIFACTS" --reports-dir "$REPORTS"
  log "Circuit holdout: every Italian Grand Prix held out"
  "$PY" scripts/run_experiments.py --input "$input" --name holdout_monza \
    --train-end 2024 --val-year 2025 --test-year 2025 --holdout-event "Italian Grand Prix" \
    --ffr configs/ffr_production.json configs/ffr_small.json \
    --artifacts-dir "$ARTIFACTS" --reports-dir "$REPORTS"
  log "2026 domain shift: train <=2024, validation 2025, test 2026 (no retraining)"
  "$PY" scripts/run_experiments.py --input "$input" --name domain_shift_2026 \
    --train-end 2024 --val-year 2025 --test-year 2026 \
    --ffr configs/ffr_production.json configs/ffr_small.json \
    --artifacts-dir "$ARTIFACTS" --reports-dir "$REPORTS"
  "$PY" scripts/update_readme_results.py f1_2025h2 holdout_monza domain_shift_2026 \
    --titles "Season-round split, 2018-2025 FastF1 tier" "Circuit holdout (Monza)" "2026 domain shift, no retraining"
}

stage_legacy_experiments() {
  local input=$PROC/f1_laps_all_tiers.parquet
  log "Legacy extension: training set 2000-2024 (both tiers), same 2025 validation/test split"
  "$PY" scripts/run_experiments.py --input "$input" --name f1_2025h2_legacy_ext \
    --train-end 2024 --val-year 2025 --test-year 2025 --split-round 12 \
    --ffr configs/ffr_production.json configs/ffr_small.json \
    --artifacts-dir "$ARTIFACTS" --reports-dir "$REPORTS"
  "$PY" scripts/update_readme_results.py f1_2025h2 f1_2025h2_legacy_ext holdout_monza domain_shift_2026 \
    --titles "Season-round split, 2018-2025 FastF1 tier" "Same split, training extended to 2000 with the legacy tier" \
             "Circuit holdout (Monza)" "2026 domain shift, no retraining"
}

for stage in "${@:-fastf1 build-fastf1 experiments}"; do
  case "$stage" in
    fastf1) stage_fastf1 ;;
    legacy) stage_legacy ;;
    build-fastf1) stage_build_fastf1 ;;
    build-all) stage_build_all ;;
    experiments) stage_experiments ;;
    legacy-experiments) stage_legacy_experiments ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
  rc=$?
  if [ "$rc" -ne 0 ]; then log "stage $stage exited with $rc"; exit "$rc"; fi
  log "stage $stage done"
done
