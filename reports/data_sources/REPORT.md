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

`scripts/cross_provider_check.py` matches laps on season, event, session, driver and lap
number and reports agreement per column (lap 1 excluded: every provider times the opening
lap from a different reference point).

### Ergast dump (Kaggle) against the Jolpica API, 2000-2017

| | |
| --- | --- |
| Laps | 363,027 (API) vs 363,017 (dump); 355,515 shared, 330 events on both sides |
| Lap time within 2 ms | 99.73% |
| Position | 99.74% |
| Pit-in / pit-out flags | 98.1% (the dump records pit stops before 2011, the API collector did not fetch them) |

The unmatched laps are one driver: the API loader wrote Häkkinen as `HÄK`, the dump loader
as `HAK`. Both loaders now strip accents, so the tiers agree.

### Ergast dump against FastF1, 2018-2026

| | |
| --- | --- |
| Laps | 203,631 (FastF1) vs 204,223 (Ergast); 198,380 shared |
| Lap time within 2 ms | 99.44%; 442 laps differ by more than 1 s |
| Position | 99.28% |
| Pit-in flag | 99.87% |
| Team name | 99.6% (the dump names teams by constructor, mapped per season: RB in 2024, Racing Bulls from 2025, Kick Sauber 2024-2025, Audi and Cadillac in 2026) |

The 442 lap-time disagreements are Ergast lap-alignment errors in a few races (2020
Austrian and 2021 Styrian Grands Prix are the worst: a lap's time is attributed to the
neighbouring lap for some drivers). Ergast has the 2018 Italian Grand Prix, which the
FastF1 archive cannot load; it names the 2026 Barcelona race "Barcelona-Catalunya Grand
Prix" where FastF1 says "Barcelona Grand Prix".

### TracingInsights against FastF1, 2025-2026 races

| | |
| --- | --- |
| Laps | 41,838 (FastF1) vs 40,610 (TracingInsights); 39,852 shared |
| Lap and sector times, compound, stint, track status, pit flags, accuracy, deleted | 100% |
| Tyre life, position, fresh-tyre flag | 99.7-99.9% |
| Weather columns | 84-98% (the archive joins the weather sample nearest the lap start; FastF1's table joins a neighbouring sample, so temperatures differ by a tenth of a degree) |

The archive is missing the 2025 São Paulo Grand Prix. Because its round numbers are derived
from lap dates, the fetch script takes official round numbers from any RaceShift table
(`--rounds-from`); without it, later rounds of an incomplete season shift by one.

### OpenF1 against FastF1, 2025-2026 races

| | |
| --- | --- |
| Laps | 41,838 (FastF1) vs 41,433 (OpenF1); 40,628 shared, 37 events on both sides |
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
| FFR-M on the 2025 test rounds (13-24) | 0.331 (10,519 laps) | 0.327 (10,444) | 0.325 (9,538; São Paulo missing) |
| FFR-S on the 2025 test rounds | 0.328 | 0.329 | |
| previous-lap stopwatch, same rows | 0.336 | 0.340 | 0.340 |
| FFR-M (trained to 2024) on all 2026 races | 0.413 (10,778) | 0.429 (10,675) | 0.416 (10,788) |
| previous-lap stopwatch, 2026 | 0.453 | 0.465 | 0.453 |

Per event in 2026 the two providers agree to within 0.03 s except the Australian Grand
Prix (0.457 s on FastF1 rows, 0.547 s on OpenF1's broken feed), which is most of the gap.

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
8,916 laps; TracingInsights agrees with OpenF1 lap for lap on the 7,086 laps it also
holds). The 2026 domain-shift model (trained on races to 2024) was scored on sprints,
then fine-tuned on the 2023-2024 sprints with the 2025 sprints calibrating the interval
and the five 2026 sprints held out (1,611 scored laps).

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
