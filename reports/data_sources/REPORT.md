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

