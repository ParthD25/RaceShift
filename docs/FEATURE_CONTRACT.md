# Feature Contract

The prediction target is always lap `N+1`. Every input must be known by the end of lap `N`.
The contract is enforced by code (`raceshift.features.full_context`) and by tests
(`tests/test_feature_availability.py`, `tests/test_lap_state_and_adjacency.py`).

## Target

```text
target_delta_vs_rolling5_s = lap_time(N+1) - rolling_median_5(N)
predicted_next_lap         = rolling_median_5(N) + predicted_residual
```

Training winsorizes the residual to ±6 s (`--target-clip`) so restarts, damage and traffic
outliers cannot dominate the fit. Evaluation always uses the raw next lap time.

## Which laps are training rows

A lap is a **valid racing lap** when it has a lap time, FastF1 marks it accurate, it is not
deleted, and it is not a pit-in, pit-out, safety-car, virtual-safety-car or red-flag lap
(FastF1 track status codes 4, 5, 6, 7). A row enters training only when the lap itself
**and** the adjacent next lap are valid racing laps. Yellow-flag laps (code 2) stay in and
are flagged.

The lap-state columns are kept on the table for analysis but are **not** model inputs:

```text
lap_valid  is_safety_car  is_vsc  is_yellow  is_red_flag  is_pit_in  is_pit_out  is_deleted
```

## Segments: how temporal context is scoped

Consecutive valid laps form a *segment*. A segment breaks at every invalid lap, at the first
valid lap after one, and at any gap in lap numbering. Every rolling statistic and every lag
is computed **inside the current segment only**, so "previous lap" can never silently mean
"the lap before the pit stop". `laps_in_segment` counts how deep the current run is.

## Feature taxonomy

### Static / identity (categorical, one-hot encoded with a rare-category floor)

```text
series  event  circuit  session  driver  team  driver_team  manufacturer  car_class
compound  tyre_manufacturer
```

Drivers and constructors are never ordinal integers. `driver_team` is the interaction that
survives driver moves between constructors.

### Dynamic race state (numeric, known at the end of lap N)

```text
lap_number  sector1_rel_3  sector2_rel_3  sector3_rel_3  tyre_life  fresh_tyre  stint
laps_in_segment  position  air_temp_c  track_temp_c  humidity_pct  pressure_mbar  rainfall
wind_speed_ms  wind_dir_sin  wind_dir_cos  gap_ahead_s  gap_behind_s
mean_speed_kph  max_speed_kph  mean_throttle_pct  brake_fraction  mean_rpm  max_rpm
gear_changes  drs_fraction
```

Wind direction is encoded as sine/cosine so 359° and 1° stay close. Sector features are
relative to the 3-lap sector mean.

### Temporal / recent pace (numeric, current segment only)

```text
rolling_median_5        absolute anchor, also the forecast baseline
pace_spread_3_5         rolling_median_3 - rolling_median_5
sector{1,2,3}_share_3   3-lap sector mean / rolling_median_5
pace_trend_3            mean of the last three lap-to-lap deltas
lap_delta_to_rolling5   lap_time(N) - rolling_median_5(N)
<source>_lag{1..4}      lags of 12 sources; lap-time and sector lags are stored relative to
                        the current rolling pace / sector mean (suffix _rel)
```

### Historical priors (numeric, strictly earlier events only)

```text
hist_driver_circuit_pace_rel          hist_team_circuit_pace_rel
hist_compound_circuit_pace_rel        hist_driver_circuit_compound_pace_rel
hist_team_circuit_compound_pace_rel   hist_driver_global_pace_rel
hist_team_global_pace_rel             hist_weather_compound_pace_rel
hist_driver_weather_pace_rel          hist_team_weather_pace_rel
```

Each prior is the median of per-event medians over **strictly earlier events** (ordered by
event date, then round number). It enters the model relative to the current rolling pace,
so it reads as "historically this driver is 0.3 s quicker here than the current run
suggests" and transfers across circuits. The absolute priors are kept on the table and
exposed by the API as `historical_context`.

## Why relative features

Absolute lap-time features carry circuit scale (Monza ≈ 80 s, Singapore ≈ 100 s) and season
scale. Linear and Forward-Forward models extrapolate badly on them; trees mostly cope. Every
lap-time-scale input is therefore expressed relative to the current rolling pace, and the
single absolute anchor is `rolling_median_5`.

## Sparse-feature filter

A numeric feature observed in fewer than 5% of training rows (`--min-feature-coverage`) is
dropped before preprocessing. Otherwise median imputation makes it near-constant in
training and any real value seen later becomes an enormous standardized outlier. Weather-
matched priors and telemetry summaries are the usual casualties on small datasets; the
dropped list is recorded in `metrics.json` and `feature_contract.json`.

## Availability guarantee

`tests/test_feature_availability.py` perturbs every lap after a cutoff (later laps of the
same race for every driver, and every later event) and asserts that no contract feature at
or before the cutoff changes. It also asserts that historical priors do not move when the
current event's lap times change.

## Leakage rules

Forbidden:

- any lap `N+1` timing or telemetry
- target-lap sectors
- target-lap timestamps that reveal elapsed time
- final race results as an input to an earlier lap
- historical priors that include the current event
- preprocessing fitted on validation/test data
- random lap-level headline split

Blocklisted column names are rejected by `assert_no_target_leakage` before any fit.
