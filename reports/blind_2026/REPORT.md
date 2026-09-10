# RaceShift independent blind test — report

Tested: https://github.com/ParthD25/RaceShift @ `626fa60` ("Public release"), cloned fresh on 2026-09-10 into
`RaceShift`.
Environment: Python 3.11.15, Node 22.22.2, 4 vCPU / 16 GB. Nothing in the product was modified.

## 1. What I did

1. Installed exactly as documented (`pip install -e ".[local]"`, `npm install`), started the API (`scripts/dev_api.py`) and the Vite UI, ran `npm run build` and the documented `npm run demo:data && npm run demo:model`.
2. **Reproduced the published 2025 numbers** from the committed artifact and the shipped `f1_2025_season.parquet`.
3. **Fed it data the model has never seen**: with the repo's own loader (`raceshift.data.fastf1_loader.export_session`) I fetched every 2026 race run so far (13 rounds, Australia -> Monza 2026-09-06, 15,149 laps), plus two 2026 sprints (China, Zandvoort) and Monza 2026 qualifying. The committed FFR-M was trained on 2018-2024 only; 2026 has new regulations, two teams (Audi, Cadillac) and four drivers (ANT, BOR, HAD, LIN) that are not in the model's one-hot vocabulary. I scored next-lap predictions against the real next laps, per event / team / driver / compound / tyre age / race phase / flag state, against the product's own "repeat the last lap" and "rolling-5 median" baselines, and against the README's own 2026 claims.
4. Simulated **live mid-race use**: forecasts at lap 5, 10, ..., 50 for every driver in every 2026 race, scored against whatever lap actually followed (including pit-in and safety-car laps a live user cannot know about).
5. Ran the same flows the UI runs (upload -> forecast -> 10-lap backtest) through the HTTP API and through the real React UI with Playwright (all routes, Compare, Datasets upload, mobile viewport).
6. Attacked edge cases: leakage/perturbation and determinism tests, malformed and adversarial files, path traversal, unseen categories, absurd values, concurrency, ONNX export equivalence on unseen data.

Scripts and raw outputs live next to this file (`reproduce_2025.py`, `eval_2026.py`, `live_sim.py`, `leakage.py`, `http_edge.py`, `ui_test*.js`, `shots/*.png`, `eval_2026_ffrm_scored.parquet`).

## 2. Headline numbers: claims vs. what I measured

| Claim (README / reports) | Measured by me | Gap |
| --- | --- | --- |
| 2025 test rounds 13-24, FFR-M MAE 0.350 s (committed `test_predictions.csv`) | Recomputed from the committed CSV: **0.3503 s**, coverage 0.862 (reproduces exactly). Re-running the artifact on the *shipped* single-season file: **0.3496 s**, coverage 0.857 | Aggregate reproduces; individual predictions differ (mean 0.049 s, max 0.74 s, 31% of rows move >0.05 s) because every circuit prior is missing in a single-season file (coverage of `hist_driver_circuit_pace` in test rows = 0.0). The model is robust to that. |
| 2025 rounds 1-12 (validation): FFR-M 0.430 vs previous-lap 0.434 | On the shipped file: FFR-M **0.443** vs previous-lap **0.434**: the model *loses* to the stopwatch on the first half of 2025 by 0.009 s | The published 0.430 needed 2018-2024 history in the file; the shipped demo file cannot reproduce it. |
| 2026 domain shift, no retraining: FFR-M MAE **0.433**, RMSE **0.768**, p90 0.957, 74.4% within 0.5 s, coverage 0.769; previous-lap 0.472 / RMSE 0.858 | Committed FFR-M on all 13 2026 races via the product's inference path: MAE **0.479**, RMSE **1.482**, p90 0.975, 74.0% within 0.5 s, 90.5% within 1 s, coverage 0.796; previous-lap **0.505** / RMSE 1.478 | MAE +0.046 s, RMSE almost doubled. Cause found and isolated (section 4.1): the product's inference path lets red-flag restart laps through that the training/report path excludes. Excluding those 9 rows: FFR-M 0.448 / RMSE 0.927, previous-lap **0.472** (matches the README's baseline exactly). |
| "predicts 8 in 10 laps within half a second and 19 in 20 within one second" | 2026: 7.4 in 10 and 18.1 in 20 (clean protocol). Live simulation, all next laps: 6.9 in 10. | Holds for 2025 H2 only. |
| 80% interval covers ~86% of test laps | 2026 clean protocol 79.6%; live use 74.7%; Chinese sprint 64.7% | Interval is a single fixed +/-0.605 s width. |
| "the 80% interval ... widened by cross-layer disagreement" (README, UI legend) | Disagreement on 11,457 unseen 2026 rows: mean 0.052 s, **max 0.333 s**, never above the 0.605 s floor -> widening fired on **0 rows**. Half-width was 0.605 s on every single row (std 0.0). | The mechanism is vacuous in practice. |
| FFR beats previous-lap baseline | 2026 clean rows: FFR wins on **52.0%** of laps (5% ties). Mean edge 0.024-0.026 s. Per-driver 10-lap UI backtests at Monza 2026: FFR better for 15 of 21 drivers; mean 0.407 vs 0.494 s. | Real but small, as the README says. On Japanese GP 2026 and Monza 2026 the stopwatch wins outright (0.385 vs 0.394; 0.758 vs 0.849). |
| Single-row inference latency 3.97 ms (metrics.json) | NumPy 3.73 ms; ONNX export 0.35 ms | OK. ONNX matches NumPy to 1e-5 on all 11,457 unseen rows; JSON preprocessor matches the joblib pickle exactly (max diff 0.0). |
| "first run takes about 10 s" | 8.7 s for the 26.7k-row season file, 0.06 s cached; 5.5 s for the 15k-row 2026 file | OK single-user. Concurrency is a different story (4.2). |

## 3. Blind results on 2026 in detail (committed FFR-M, 2025+2026 file so 2025 priors exist)

Per event (n = scored lap pairs; MAE in s; "prev" = repeat-the-last-lap):

| Event | n | FFR-M | prev | roll5 | FFR p90 | cov80 | bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hungarian GP | 1190 | 0.427 | 0.488 | 0.490 | 1.058 | 0.786 | -0.088 |
| Monaco GP | 1075 | 0.660 | 0.692 | 0.878 | 1.536 | 0.713 | +0.048 |
| Austrian GP | 1046 | 0.334 | 0.387 | 0.379 | 0.755 | 0.863 | -0.081 |
| Dutch GP (wet phases) | 1003 | 0.508 | 0.519 | 0.595 | 1.036 | 0.802 | -0.007 |
| Barcelona GP | 922 | 0.422 | 0.445 | 0.499 | 0.937 | 0.804 | -0.224 |
| Canadian GP | 902 | 0.503 | 0.601 | 0.695 | 1.164 | 0.753 | -0.001 |
| Japanese GP | 863 | 0.394 | **0.385** | 0.362 | 0.793 | 0.824 | +0.110 |
| Italian GP | 803 | 0.849 | **0.758** | 0.926 | 0.829 | 0.842 | +0.487 |
| Miami GP | 777 | 0.417 | 0.434 | 0.420 | 0.944 | 0.803 | -0.141 |
| Chinese GP | 766 | 0.412 | 0.453 | 0.440 | 0.893 | 0.800 | +0.075 |
| Australian GP | 743 | 0.462 | 0.493 | 0.556 | 1.116 | 0.775 | -0.058 |
| British GP | 743 | 0.407 | 0.426 | 0.411 | 0.826 | 0.825 | -0.044 |
| Belgian GP | 624 | 0.428 | 0.440 | 0.431 | 1.012 | 0.776 | -0.039 |

Other cuts:

- **Unseen teams**: Audi 0.495 (prev 0.535); **Cadillac 0.951** (prev 0.953, coverage 0.644). Cadillac's drivers PER (1.019) and BOT (0.862) are the two worst in the field; the model adds nothing over the stopwatch for them.
- **Unseen drivers** (ANT/BOR/HAD/LIN, all-zero one-hot): MAE 0.400 vs seen drivers 0.498, i.e. no penalty; identity features carry no signal, consistent with the README's ablation.
- Compound: HARD 0.459, MEDIUM 0.424, SOFT 0.682 (coverage 0.72). Tyre age 1-5 laps: 0.821 (prev 0.789, the stopwatch wins on fresh tyres), bias +0.23 s.
- Position: P1-3 0.304, P4-10 0.374, P11+ 0.625.
- **First clean lap of a segment** (n=992): MAE 1.093, coverage 0.627, bias +0.32. The README's 2025 breakdown reported 0.66 for this bucket.
- **Yellow-flag laps are kept as "clean"** (n=309): MAE 1.947, bias +1.46 s; the rules keep status-2 laps in, and they are the single worst systematic bucket.
- Rain-flagged laps (n=123): FFR 0.487 vs prev 0.469; the stopwatch wins in the wet.
- Without any 2025 history (2026-only file): MAE 0.485 (+0.005); per-row predictions move by mean 0.045 s, max 1.79 s.
- Legacy FFR-S (2000-2024) on the same rows: 0.487. Synthetic demo model on real 2026 laps: 0.551 (worse than the stopwatch), coverage 44%.
- Error distribution: median 0.265 s, p95 1.46 s, p99 2.82 s, **max 57.07 s**.

**Sessions the model never saw** (session one-hot has only "R"): Chinese sprint MAE 0.574 (prev 0.579, coverage 0.647, bias -0.21); Zandvoort sprint 0.525 (prev 0.569), bias -0.43 with three -9 to -12 s misses at the SC. **Qualifying** is accepted without any warning and produces MAE 23.6 s on the 7 pairs that pass the validity rules (predicts 82.7 s for a 127.0 s cool-down lap). The product should refuse or flag non-race sessions.

**Live simulation** (forecast at lap 5,10,...,50 for every driver, every 2026 race; 2,139 forecasts issued, 306 refused because lap N itself was a pit/SC/inaccurate lap):

| Next lap that actually happened | n | FFR MAE | median | p90 | cov80 | prev-lap MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All (what a live user sees) | 2135 | **1.153** | 0.291 | 1.496 | 0.747 | 1.159 |
| Clean racing lap (scored protocol) | 1985 | 0.432 | 0.268 | 0.928 | 0.796 | 0.444 |
| Not clean: pit-in / SC / VSC / inaccurate | 150 (7%) | 10.70 | 4.95 | 36.1 | 0.093 | 10.62 |

At lap 5 the model is worse than the stopwatch (0.608 vs 0.526 on clean laps); it only pulls ahead from lap 20 onward. The interval's real-world coverage is 75%, not 80-86%.

## 4. What broke or behaved unexpectedly (ranked)

### 4.1 The product's inference path lets red-flag restart laps through; training does not (data-handling bug, changes results)

`RaceShiftArtifact.inference_table()` (`src/raceshift/models/artifact.py`, used by every API forecast/backtest and the UI) filters `lap_time_s.notna()` **before** calling `build_full_context_table`. The v2 restart rule (`_red_flag_restart` in `src/raceshift/features/full_context.py`) detects a restart by counting earlier laps carrying track-status code 5, and those stoppage laps are usually **untimed**, so the filter deletes the evidence. `scripts/train_ffr.py` / `run_experiments.py` pass the raw frame without that filter.

Evidence on 2026: 59 laps carry code 5, 31 of them untimed. Training-style table: **11,445** rows with targets (exactly the README's 2026 test row count). Product path: **11,457** rows; the 12 extra are restart laps (Monza lap 6 for 7 drivers, Zandvoort lap 4 for PER, ...), flagged `is_red_flag_restart=True` when the raw frame is used and `False` on the product path. They produce 33-57 s errors: e.g. TSU Monza lap 6 (136.1 s restart lap) -> model predicts 135.4 s for lap 7, actual 88.1 s, error +47.3 s, "80% interval" 1:34.8-1:36.0. `POST /api/forecast/backtest {driver: TSU, laps: 100}` returns MAE 1.70 s, RMSE 7.8 s for TSU's race and shows that row to the user. The 9 worst rows alone move the 2026 MAE from 0.448 to 0.479 and RMSE from 0.93 to 1.48. The 2025 H2 numbers are unaffected only because no round 13-24 race in 2025 had a red flag.

### 4.2 Cold-cache thundering herd: 5 concurrent requests -> 60 s each

The feature table is cached with `lru_cache(maxsize=16)` keyed by (artifact, file, mtime, size) with no single-flight lock (`apps/api/main.py`). Warm: 0.09 s single / 0.35 s for 5 parallel. Cold (fresh file), 5 parallel backtests: **60.4-60.7 s each** versus 8 s for one. The cache key includes the artifact directory, so switching model in the UI rebuilds the same table again (8.0 s), and after 16 distinct (file, artifact) pairs the shipped season file is evicted and rebuilt (observed: 62 s for 5 parallel calls on the shipped file after my uploads). API RSS after the test session: 921 MB.

### 4.3 Unhandled 500s

- `{"file": "synthetic_fixture.csv\x00.parquet"}` -> HTTP 500 (`ValueError: embedded null byte` escapes `_safe_child`).
- Uploading a CSV whose `season` column is a string (`"twenty25"`) -> HTTP 500 on `/api/import` (`int()` in `_table_summary`); the subsequent forecast returns 400 with the internal message "Unknown datetime string format, unable to parse: __missing__-01-01".

### 4.4 Forecast refused for a driver whose race is fully backtestable, and the UI then shows nothing

`forecast_last_available` refuses when the driver's final lap is not a clean lap. In 2026 that is **45 of 279 driver-races (16%)**, e.g. NOR at Monza (final lap deleted for track limits: timed, accurate, status 1, `deleted=True`), ALO at Monza. The UI runs the 10-lap backtest only *after* a successful forecast (`apps/web/src/pages/Forecast.tsx`), so selecting NOR yields "The latest lap for this driver is not a usable completed lap (pit, deleted or inaccurate)" and **no backtest**, although `POST /api/forecast/backtest` for NOR works (MAE 0.45 vs stopwatch 0.71). The same misleading message appears when `season` is a string column (dtype mismatch between raw and table), which has nothing to do with the lap.

### 4.5 The model cannot follow a trend; it regresses to the 5-lap median

Feeding VER's Abu Dhabi laps a +3 s/lap degradation (91.2, 94.8, 97.6, 100.8, 103.5) gives a forecast of **98.3 s**, 5.2 s below the last lap and below the rolling-median trajectory. Any real tyre cliff, fuel-saving order or rain onset will be forecast late. The same anchoring is why the "first clean lap" bucket is 2.5x worse than average.

### 4.6 Validity rules leave two systematic holes

Yellow-flag laps (status 2) count as clean rows and targets: 309 of 11,457 2026 rows, MAE 1.95 s, bias +1.46 s. The first lap after a safety-car period (status "12", e.g. COL Monaco lap 71 at 99.3 s vs 79.5 s median) is treated as a clean lap and its own time becomes the rolling-5 baseline for the next prediction (19.1 s error); this row is present on *both* code paths, so it is a rule gap rather than the 4.1 bug.

### 4.7 Input validation is permissive to the point of silently producing numbers

Accepted without warning and scored/forecast "normally": negative lap times (-5 s every 7th lap), a 1e9 s lap in the history (the forecast still comes back ~88 s because the median absorbs it, but the lag inputs are 1e9 minus the median, standardized), string lap times ("fast" -> silently dropped), duplicated rows (segment length collapses to 1 for every lap, backtest still returns a summary), NaN driver codes (driver "nan" becomes selectable), a 1999 row mixed into 2025, track temperature 999 C / humidity 500 % (forecast shifts by 0.28 s, no warning), tyre life 500 / position 0, unseen compound "C6" and team "Cadillac" (all-zero one-hot, no warning), session relabelled "Q" (no warning). The UI dataset dropdown then lists every such file as a first-class dataset.

### 4.8 Documented fetch CLI fails with a raw traceback for future or ambiguous events

`scripts/fetch_fastf1.py --year 2026 --event Spanish` (FastF1 resolves it to the Madrid round on 2026-09-13, which has not happened), `--event 14`, `--event Bahrain` all crash with `fastf1.exceptions.DataNotLoadedError: The data you are trying to access has not been loaded yet` instead of "event has not taken place". The documented 2025 Abu Dhabi command works (11 s) and the UI lists the file.

### 4.9 Environment / packaging

- `pyproject.toml` pins `scikit-learn>=1.6`; a fresh install today gets 1.9.1 while `preprocessor.joblib` was pickled with 1.9.0. Every artifact load emits 14 `InconsistentVersionWarning`s ("might lead to breaking code or invalid results"). I verified numerically that the JSON preprocessor spec and the pickle agree exactly on all 2026 rows, so no harm here, but the pickle path is the one the API uses and the warning spam lands in the API log on every cold load.
- The global Playwright (1.56) expects a newer Chromium than the one installed; unrelated to the product, noted for reproducibility (I launched with an explicit `executablePath`).

### 4.10 UI nits

- Backtest copy is hard-coded: on 2026 data it still says "over the full 2025 test rounds the model is ahead by a few thousandths of a second". The interval legend always quotes the 2025 86% coverage even when the file is 2026 (79.6%) or a 3-lap CSV.
- Auto driver selection ("the winner") at Monza 2026 picked ANT; fine, but `available_drivers` includes drivers whose forecast will be refused (see 4.4) with no marker.
- Mobile (390 px) Forecast page has a 14 px horizontal overflow (scrollWidth 404).
- First Overview load was 32.8 s (Vite cold dependency optimisation; warm loads 0.9 s): not the product, but a first-run impression.
- Console: one 400 from `/api/forecast/backtest` when a driver with no clean pairs is chosen in Compare; the error box is shown correctly.

## 5. What held up

- **No leakage found.** Forecast for VER's lap 40 is bit-identical whether the file is truncated at lap 40 or contains the whole race (diff 0.00); perturbing all later laps by +30 s changes nothing (0.0); perturbing every other driver's laps in the race changes nothing (0.0); shuffling row order changes nothing; runs are deterministic. Perturbing an *earlier* event by +5 s moves the forecast by only 0.008 s: priors are close to inert, matching the README's ablation.
- Path traversal (`../`, absolute paths, `%2F`, `file/../other`, artifact `../data`, export `../`), non-csv/parquet suffixes, empty/one-column/header-only uploads, corrupt parquet bytes, overwrite-without-flag (409), `laps` bounds (422) are all rejected cleanly; upload filenames are sanitised (`../../evil.csv` -> `evil.csv`).
- Missing chronology column is refused with a clear message; a minimal 7-column CSV forecasts and backtests; a 3-lap file forecasts with the correct "only 3 completed laps / no priors" warnings; synthetic-model-on-real-data and real-model-on-synthetic are both badged and warned in the UI.
- Export endpoint builds a 4.3 MB zip in 0.7 s; the ONNX core reproduces NumPy to 1e-5 on 11,457 unseen rows.
- `npm run build` passes; `npm run demo:data && npm run demo:model` run in 0.7 s + 3.3 s and the result is badged synthetic.
- UI: all routes render, Compare works (VER vs LIN at Monza 2026: 15 pairs each), Datasets upload works with a proper duplicate-file error, no page errors.
- Committed CSV -> README metrics reproduce to the third decimal.

## 6. Bottom line

On genuinely unseen 2026 races the committed model is within 0.48 s of the real next lap on average (0.45 s once the restart-lap bug is discounted), beats the "repeat the last lap" stopwatch by 0.02-0.03 s and only on 52% of laps, is worse than the stopwatch on fresh tyres, in the wet, at lap 5, at Suzuka and at Monza, and has an "80% interval" that is a fixed +/-0.605 s band covering 75% of laps in live use. The README's characterisation of a small edge is honest; its 2026 RMSE (0.77) and coverage (0.77) are not what the shipped product produces (1.48 and 0.80) because the API's inference path and the experiment scripts do not apply the same lap-validity rules (4.1). That mismatch, the thundering-herd cache, the two 500s and the accept-anything validation are the concrete defects; the modelling weakness that matters most for a "forecast the next lap before it happens" product is that 7% of the laps a live user asks about are pit or safety-car laps where every model, this one included, is off by ~10 s with a 9% interval hit rate.


## 7. Re-run after the fixes (added by the maintainer)

The findings above were taken on commit `626fa60`. After the fixes in the following commit
(inference path applies the training lap-validity rules to the whole raw frame; single-flight
feature-table cache shared across artifacts; 400 instead of 500 for null-byte filenames and
non-numeric `season`/`lap_number`; backtest independent of the forecast in the UI; non-race
session warning; friendly fetch errors), the same 2025+2026 file through the product path gives:

| Metric | Before | After |
| --- | ---: | ---: |
| Scored 2026 lap pairs | 11,457 | 11,445 |
| FFR-M MAE / RMSE / p90 | 0.479 / 1.482 / 0.975 s | 0.435 / 0.777 / 0.969 s |
| Within 0.5 s / 80% coverage | 74.0% / 79.6% | 74.1% / 79.8% |
| Previous-lap MAE / RMSE | 0.505 / 1.478 s | 0.472 / 0.858 s |
| Largest error | 57.07 s | 19.11 s |
| `POST /api/forecast/backtest` TSU, Monza, `laps=100` RMSE | 7.8 s | 0.229 s (43 rows, MAE 0.166 s) |
| 5 parallel cold backtests on a fresh file | 60.4-60.7 s each | 6.4 s each, 0.07 s warm |
| `{"file": "synthetic_fixture.csv\u0000.parquet"}` | 500 | 400 |
| CSV with `season = "twenty25"` | 500 on import | 400 "Column 'season' must be numeric; found 'twenty25'" |
| Monza 2026 qualifying as latest session | silent 200 | 200 with `session_warning` |
| NOR Monza 2026 (final lap deleted) | forecast 400, UI shows nothing | forecast 400, backtest shown (0.45 s vs 0.71 s stopwatch) |

Not changed: yellow-flag laps still count as clean (a validity-rule change bumps
`LAP_VALIDITY_VERSION` and re-runs every experiment), the first lap after a safety-car period
still enters the rolling-5 baseline, no trend term, no value-range validation on upload. The
2026 table used here ships as `data/imports/f1_2026_races.parquet`.
