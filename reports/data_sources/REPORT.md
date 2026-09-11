# External data sources: what was added, checked and learnt

Generated 2026-09-11 on branch `data-sources`. Every number below comes from a script in
this repository run on the tables named; the commands are listed at the end.

## Sources reviewed

| Source | Verdict | What it is |
| --- | --- | --- |
| OpenF1 API (openf1.org) | **used** | Independent provider of the live-timing feed, 2023 onward, races and sprints. Loader rebuilds the full FastF1 column set (laps, sectors, stints, pit stops, positions, weather, race control as track-status codes). |
| Ergast database dump on Kaggle (`jtrotman/formula-1-race-data`) | **used** | The Ergast/Jolpica database as CSV: every race lap 1996-2026, pit stops, results. Same data the legacy tier already reads from the API; now available offline and for 1996-1999 and the current season. |
| TracingInsights archive (tracinginsights.com/data, `github.com/TracingInsights/<year>`) | **used** | FastF1's lap table per session (plus per-lap weather) for every session since 2018, published on GitHub. Not an independent measurement (FastF1 is its upstream) but a complete, rate-limit-free mirror. |
| `toUpperCase78/formula1-datasets` | evaluated, not used | Season CSVs 2019-2026 of race, qualifying and sprint results, calendars, drivers, teams. Race-level only: no per-lap timing, so nothing the next-lap model can learn from that the lap tables do not already imply. |
| Kaggle notebook `jamesd2525/feature-engineering-lgbm-regressor-f1-hackathon` | not accessible | The Kaggle token available to this environment may not read notebooks (`kernels.get` denied); its features could not be inspected. |

## Data integrity across providers

`scripts/cross_provider_check.py` matches laps on season, official round number, session,
driver and lap number (event names differ between providers) and reports agreement per
column. Lap 1 is dropped from both tables before anything is counted: every provider times
the opening lap from a different reference point. Rows with a missing key or a duplicated
key (two drivers sharing a three-letter code in the same race, in the legacy tier) are
excluded and counted, so every match is one lap to one lap.

### Ergast dump (Kaggle) against the Jolpica API, 2000-2017

| | |
| --- | --- |
| Laps from lap 2 | 355,758 (API) vs 355,748 (dump); 354,147 shared; 456 laps per side with a duplicated key excluded (Panis and Pantano both coded `PAN` in 2004) |
| Lap time within 2 ms | 99.96%; 69 laps differ by more than 1 s |
| Position | 99.97% |
| Pit-in / pit-out flags | 98.1% (the dump records pit stops before 2011, the API collector did not fetch them) |

The unmatched laps are one driver: the API loader wrote Häkkinen as `HÄK`, the dump loader
as `HAK`. Both loaders now strip accents, so the tiers agree.

### Ergast dump against FastF1, 2018-2026

| | |
| --- | --- |
| Laps from lap 2 | 199,931 (FastF1) vs 200,578 (Ergast); 199,592 shared |
| Lap time within 2 ms | 99.45%; 442 laps differ by more than 1 s |
| Position | 99.29% |
| Pit-in flag | 99.87% |
| Team name | 99.6% (the dump names teams by constructor, mapped per season: RB in 2024, Racing Bulls from 2025, Kick Sauber 2024-2025, Audi and Cadillac in 2026) |

The 442 lap-time disagreements are Ergast lap-alignment errors in a few races (2020
Austrian and 2021 Styrian Grands Prix are the worst: a lap's time is attributed to the
neighbouring lap for some drivers). Ergast has the 2018 Italian Grand Prix, which the
FastF1 archive cannot load, and no laps for the 2021 Belgian Grand Prix (run behind the
safety car); it names the 2026 Barcelona race "Barcelona-Catalunya Grand Prix" where
FastF1 says "Barcelona Grand Prix", which is why laps are matched on the round number.

### TracingInsights against FastF1, 2025-2026 races

| | |
| --- | --- |
| Laps from lap 2 | 41,083 (FastF1) vs 41,103 (TracingInsights); 41,083 shared (every FastF1 lap), 37 events on both sides |
| Lap and sector times, compound, track status, pit flags, accuracy, deleted, circuit, event date | 100% |
| Tyre life, stint, position, fresh-tyre flag | 99.7-99.9% |
| Weather columns | 85-98% (the archive joins the weather sample nearest the lap start; FastF1's table joins a neighbouring sample, so temperatures differ by a tenth of a degree) |

The archive holds every 2025 and 2026 race. An earlier pass of this check reported the 2025
São Paulo Grand Prix missing: the collector listed the season repository with `git
ls-tree`, which quotes and escapes paths with non-ASCII characters, so the "São Paulo"
folder never matched and was silently skipped. The listing is now NUL-separated. The
archive stores no circuit or round number; the collector takes both, and the event date,
from the FastF1 table (`--rounds-from`) so the rows carry the same identifiers as the
FastF1 tier, and leaves the round unset rather than numbering the events it happens to hold.

### OpenF1 against FastF1, 2025-2026 races

| | |
| --- | --- |
| Laps from lap 2 | 41,083 (FastF1) vs 40,671 (OpenF1); 40,628 shared, 37 events on both sides |
| Lap and sector times within 2 ms | 97.6% (325 laps differ by more than 1 s) |
| Position | 99.0% |
| Pit-in / pit-out flags | 99.9% |
| Compound | 98.6% |
| Track status string (rebuilt from race control) | 95.8% |
| Accuracy flag | 99.2% |
| Tyre life / fresh-tyre flag | 83.6% / 94.4% (OpenF1's stint feed reports some used sets as new, so tyre age starts at 1 instead of the true age) |
| Weather | 87-99% (OpenF1 samples once a minute; the lap-start join differs by up to a sample) |

The lap-time disagreements concentrate in the 2026 Australian Grand Prix, where OpenF1's
own lap feed is broken for several drivers: a stretch of laps is merged into one 1,168 s
entry, later lap numbers are shifted by one, and its pit records then point at the wrong
lap. No reconstruction can repair a feed that disagrees with itself; the loader's slow-lap
guard (a clear-track lap more than 1.2× the driver's median clean lap is not accurate)
removes the in-laps it would otherwise let through.

### Same model, two providers

`scripts/score_artifact.py` scores one artifact on a table built from FastF1 rows up to
2024 plus one provider's rows for 2025-2026, so the historical priors are identical and
only the scored season differs. Mean absolute error of the next-lap forecast, seconds:

| Rows scored | FastF1 | OpenF1 | TracingInsights |
| --- | --- | --- | --- |
| FFR-M on the 2025 test rounds (13-24) | 0.331 (10,519 laps) | 0.327 (10,444) | 0.322 (10,519) |
| FFR-S on the 2025 test rounds | 0.328 | 0.329 | |
| previous-lap stopwatch, same rows | 0.336 | 0.340 | 0.336 |
| FFR-M (trained to 2024) on all 2026 races | 0.413 (10,778) | 0.429 (10,675) | 0.415 (10,788) |
| previous-lap stopwatch, 2026 | 0.453 | 0.465 | 0.453 |

Per event in 2026 the two providers agree to within 0.03 s except the Australian Grand
Prix (0.457 s on FastF1 rows, 0.547 s on OpenF1's broken feed), which is most of the gap.
The TracingInsights rows are FastF1's own laps (identical timing, status, pit and tyre
columns on every scored lap) and still score 0.009 s better on the 2025 rounds: the only
columns that differ are the weather sample joined to each lap (track temperature by up to
1.7 °C, wind speed by up to 2.4 m/s, air temperature by up to 0.5 °C) and a few dozen
tyre-life and compound rows. That is the size of the model's sensitivity to the weather
join, and a reminder that a third decimal in these tables is provider noise.

The first pass of this check did not look like this. Scored on the first OpenF1 tables the
same model had an error of 0.586 s on the 2025 test rounds with a root-mean-square error
of 3.5 s, and the stopwatch itself was 0.12 s worse than on FastF1 rows. Two things were
wrong, both caught only because the same laps were scored from two sources: a race that
started behind the safety car (Belgium 2025) is announced before the start and never sends
"SAFETY CAR DEPLOYED", so its neutralised laps passed as clean; and a stoppage can arrive
as "SESSION ABORTED" with no red-flag message (Monza 2026), which left a safety car open
for the rest of the race and invalidated every lap. The loader now handles both, plus
rolling restarts and in-laps missing from the pit feed. This is the same class of leak
the blind 2026 test found in the product path earlier, and the reason the check is now
part of the repository.

## Sprints: a session type the model had never seen

The FastF1 tables hold races only; OpenF1 supplies every sprint since 2023 (23 sessions,
8,916 laps; TracingInsights holds the same sprints and agrees with OpenF1 on every one of
the 8,426 shared laps from lap 2). The 2026 domain-shift model (trained on races to 2024)
was scored on sprints, then fine-tuned on the 2023-2024 sprints with the 2025 sprints
calibrating the interval and the five 2026 sprints held out (1,611 scored laps).

| Model on 2026 sprints | MAE s | vs untouched model (cluster bootstrap 95%) | vs previous lap |
| --- | --- | --- | --- |
| previous-lap stopwatch | 0.491 | | |
| rolling-5 median | 0.517 | | |
| untouched model, zero-shot | 0.496 | | +0.005 |
| readout refit only, 20k replay rows | **0.474** | −0.022 (−0.027, −0.018) | −0.017 (−0.047, +0.007) |
| 10 local epochs per layer, no replay | 0.482 | −0.014 (−0.021, −0.007) | −0.009 |
| 10 local epochs per layer, 20k replay rows | 0.486 | −0.010 (−0.017, −0.004) | −0.005 |
| 20 epochs, learning rate 2e-4, 20k replay | 0.483 | −0.013 (−0.019, −0.009) | −0.008 |

Zero-shot the model is no better than the stopwatch on sprints: it never saw a `session`
value other than `R`, and sprint laps are short, cool-tyre, full-push laps. Every
fine-tuning variant improves on the untouched model with intervals that exclude zero;
refitting the readout on sprint rows (with replay of race rows) does most of the work, and
the extra local layer updates add nothing on 3,237 training laps. Against the stopwatch the
best variant's 0.017 s edge is not distinguishable from sampling noise at 95%.

## Adapting to a new season with forward-forward fine-tuning

Setting: the 2026 domain-shift model (FFR-M trained on 2018-2024 races, validated on 2025)
is fine-tuned on the first five 2026 races (3,591 training laps), rounds 6-7 recalibrate
the interval, and rounds 8-13 are scored (5,210 laps). Every variant reuses the base
preprocessor and contract and runs only local, layer-wise updates plus a closed-form
readout refit (`scripts/finetune_ffr.py`). The comparison rows retrain the classical
baselines and FFR-M from scratch on the same rows (2018-2024 plus 2026 rounds 1-5, the new
`--train-through-round` split). Test MAE on 2026 rounds 8-13, seconds; intervals are
event × driver cluster bootstraps of the paired |error| difference.

| Model | MAE s | vs untouched model | vs retrained tree |
| --- | --- | --- | --- |
| previous-lap stopwatch | 0.426 | | |
| rolling-5 median | 0.419 | | |
| ridge, retrained with 2026 R1-5 | 0.391 | | |
| gradient-boosted trees, retrained with 2026 R1-5 | **0.371** | | |
| FFR-M untouched (trained to 2024), zero-shot | 0.377 | | +0.005 |
| FFR-M retrained from scratch with 2026 R1-5 (70 min) | 0.3768 | +0.0001 (−0.0007, +0.0009) | +0.006 (−0.000, +0.011) |
| fine-tune: readout refit only, 20k replay rows (15 s) | 0.3765 | −0.0002 (−0.0014, +0.0009) | +0.005 (−0.001, +0.011) |
| fine-tune: 5 local epochs, lr 1e-4, 20k replay (28 s) | 0.3765 | −0.0002 (−0.0015, +0.0010) | +0.005 (−0.001, +0.011) |
| fine-tune: 20 epochs, lr 2e-4, 60k replay in layers (22 min) | 0.3771 | +0.0004 (−0.0003, +0.0011) | +0.006 (+0.000, +0.011) |
| fine-tune: 20 epochs, lr 2e-4, 20k replay | 0.3779 | +0.0012 (−0.0004, +0.0026) | +0.007 (+0.000, +0.012) |
| fine-tune: 10 epochs, 20k replay (readout and layers) | 0.3787 | +0.0020 (+0.0004, +0.0036) | +0.007 |
| fine-tune: 10 epochs, 20k replay (readout only) | 0.3794 | +0.0027 (+0.0007, +0.0047) | +0.008 |
| fine-tune: 30 epochs, 20k replay | 0.3809 | +0.0042 (+0.0020, +0.0064) | +0.010 |
| fine-tune: 10 epochs, no replay | 0.3945 | +0.0178 (+0.0129, +0.0228) | +0.023 |

What the numbers say:

- Five races of a new regulation season do not move the model. The best fine-tuning
  variants tie the untouched model (differences of 0.0002 s with intervals straddling
  zero), and so does FFR-M retrained from scratch on 2018-2024 plus those five races
  (0.3768 s, +0.0001 s against the untouched model, interval (−0.0007, +0.0009)): 70
  minutes of training buy exactly what a 15-second readout refit buys, which is nothing.
  The untouched model already beats the stopwatch by 0.049 s on these rounds.
- More local epochs at the base learning rate make things worse, and refitting the readout
  on the new rows alone without replay costs 0.018 s: the 3.6k new laps are too few to
  re-solve a 125-coefficient readout, and the layers drift towards the new distribution
  and away from the old one. Replay of old rows is the difference between harmless and
  harmful; a low learning rate (1e-4) and few epochs are equivalent to not updating the
  layers at all.
- The retrained gradient-boosted trees are the best model on these rounds by 0.005 s over
  every FFR variant, an edge whose interval just excludes zero for the fine-tuned
  variants (+0.000 to +0.011 s). This matches the season-split result: the tree is the
  strongest model, FFR sits within a few thousandths of it, and both beat the stopwatch by
  a wide margin.
- Fine-tuning does help when the new rows are a different kind of session (sprints,
  above): there the untouched model was no better than the stopwatch and a 15-second
  readout refit recovered 0.022 s.

## Commands

```bash
# providers
python scripts/fetch_openf1.py --years 2025-2026 --sessions R --output data/raw/openf1 --combine data/imports/f1_races_openf1.parquet
python scripts/fetch_openf1.py --years 2023-2026 --sessions S --output data/raw/openf1 --combine data/imports/f1_sprints_openf1.parquet
python scripts/fetch_tracinginsights.py --years 2025-2026 --sessions R --rounds-from data/processed/f1_laps_fastf1.parquet --combine data/imports/f1_races_tracinginsights.parquet
python scripts/build_ergast_kaggle.py --csv-dir data/raw/ergast_kaggle --years 1996-2026 --include-timing-era --output data/processed/f1_laps_ergast.parquet   # 2018+ rows for the checks only
python scripts/cross_provider_check.py --left data/processed/f1_laps_fastf1.parquet --right data/imports/f1_races_openf1.parquet --names fastf1 openf1

# same model, two providers (mixed table = FastF1 rows to 2024 + the provider's 2025-2026 rows)
python scripts/score_artifact.py --artifact artifacts/f1_2025h2_ffr-m --input <mixed.parquet> --season 2025 --rounds 13- --output reports/data_sources/ffr-m_openf1_2025

# 2026 fine-tuning and its comparison rows
python scripts/finetune_ffr.py --base artifacts/domain_shift_2026_ffr-m --input data/processed/f1_laps_fastf1.parquet --season 2026 --train-rounds 1-5 --val-rounds 6-7 --test-rounds 8- --epochs-per-layer 0 --replay-rows 20000 --output artifacts/ft2026_ffr-m_readout
python scripts/train_baselines.py --input data/processed/f1_laps_fastf1.parquet --output artifacts/ft2026_baselines_retrain --train-end 2024 --val-year 2026 --test-year 2026 --split-round 7 --train-through-round 5
python scripts/train_ffr.py --input data/processed/f1_laps_fastf1.parquet --config configs/ffr_production.json --output artifacts/ft2026_ffr-m_retrain --train-end 2024 --val-year 2026 --test-year 2026 --split-round 7 --train-through-round 5

# sprints (FastF1 races + OpenF1 sprints in one table)
python scripts/finetune_ffr.py --base artifacts/domain_shift_2026_ffr-m --input <races_plus_sprints.parquet> --sessions S --train-seasons 2023-2024 --val-seasons 2025 --test-seasons 2026 --epochs-per-layer 0 --replay-rows 20000 --output artifacts/ft_sprints_ffr-m_readout
```
