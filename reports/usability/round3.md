# Blind usability test, round 3

Three automated testers commissioned by the author, each given only the repository link
(commit `62bec62`, lap-validity rules v3) and a persona, with no access to the development
history. Scores are 1-10. Rounds 1 and 2 were run the same way on earlier commits.

| Persona | Criterion | Round 1 | Round 2 | Round 3 |
| --- | --- | ---: | ---: | ---: |
| F1 fan | Easy to get running | 6 | 7 | 9 |
| F1 fan | Understandable app | 5 | 8 | 6 |
| F1 fan | Understood the numbers | 5 | 6 | 7 |
| F1 fan | Did something interesting | 5 | 7 | 6 |
| F1 fan | Would show a friend | 4 | 5 | 5 |
| Data scientist | README clarity | — | — | 7 |
| Data scientist | Ease of setup | — | — | 9 |
| Data scientist | Results credible | — | — | 8 |
| Data scientist | Trust the methodology | — | 9 | 7 |
| Data scientist | Would reuse the code | — | — | 6 |
| Hiring reviewer | Engineering quality | — | — | 6 |
| Hiring reviewer | Research honesty | 7 | 8 | 7 |
| Hiring reviewer | Reproducibility | — | — | 8 |
| Hiring reviewer | Documentation | — | — | 7 |
| Hiring reviewer | Advance to interview | 6 | 7 | 7 |

## What they found, and what changed

- **The public release dropped the test suite but the docs still described it** (reviewer,
  data scientist; the largest deduction in both reports). The docs now describe the checks as
  development-time checks that the reviewers reproduced from the public code, and state that
  the public tree ships without the suite. Restoring the suite is a separate decision.
- **The blind 2026 test was called "independent" with no provenance and its scripts were
  missing** (reviewer). It is now described as an automated tester commissioned by the author,
  and the scripts are committed under `reports/blind_2026/scripts/`.
- **Only the latest race in a file could be used** (fan). Forecast and Compare Drivers now
  have a race selector listing every session in the file; `/api/imports/{file}/summary`
  returns `sessions`, and forecast/backtest accept `season`, `event`, `session`.
- **The hypothetical lap-59 forecast had more emphasis than the real backtest** (fan, data
  scientist). The forecast response carries `race_finished`; the page then titles the panel
  "Hypothetical next lap", shrinks the number and says the evidence is the backtest above.
- **"Widened by cross-layer disagreement" never fires** (data scientist, reviewer). README,
  UI legend, tooltip and model card now say the width is fixed per model and that the term
  has never exceeded its floor on a test lap.
- **The 2025 edge over the stopwatch is inside sampling noise** (data scientist).
  `scripts/paired_bootstrap.py` computes row and event x driver cluster bootstrap intervals;
  the README protocol notes report them (FFR-M vs previous lap on 2025: not distinguishable
  from zero; on 2026: clearly real; vs the tree on 2026: a tie).
- **No control for the hidden layers** (reviewer). `configs/ffr_m_random_layers.json` trains
  nothing but the ridge readout on the same architecture: 0.353 s vs 0.331 s trained.
- **Twelve scikit-learn version warnings per load** (all three). The runtime now loads the
  pickle-free JSON preprocessor spec when the artifact has one and silences the version
  warning on the joblib fallback.
- **Uploads missing required columns were stored; absurd values passed silently; no delete**
  (fan, data scientist, reviewer). Uploads missing a required column are rejected; out-of-range
  values, empty driver codes, duplicated laps and a missing chronology column are reported as
  warnings on import and on the Forecast page; `DELETE /api/imports/{file}` and a Delete
  button remove user uploads (shipped files are protected).
- **Required columns under-specified** (data scientist): `event_date`/`round_number` added to
  the documented list and flagged when absent.
- **First page load 32 s; 14 px horizontal overflow on phones; hidden Telemetry/Strategy
  routes; stale README lines; duplicate function in the API; Playwright left in devDependencies;
  CORS hard-coded to 5173; header strip not updated by Compare; forecast lost on refresh;
  "beat the stopwatch" on a 1.7 s error; gap ahead/behind always empty** (fan, reviewer). All
  fixed: Vite pre-bundles its dependencies (first load 2.8 s), the top bar shrinks at phone
  width, the routes are gone, the API reads `WEB_PORT`, Compare sets the header, the forecast
  is kept in session storage, the verdict sentence says when both numbers are unusable, and
  empty context rows are hidden.
- **Still open**: no trend-following term (the model smooths a rising lap sequence), value
  ranges are warned about but not rejected, and the research question about memory is
  answered negatively by the implementation.
