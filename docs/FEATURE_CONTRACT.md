# Feature Contract

The prediction target is always lap `N+1`. Inputs must be known by the end of lap `N`.

## Identity and environment

- series
- season
- event
- circuit
- session
- driver
- team / constructor
- manufacturer
- car class
- lap number

## Tyre context

- compound
- tyre life
- fresh/used indicator
- stint
- tyre manufacturer

In Formula 1, tyre manufacturer is effectively constant, so the learned signal comes primarily from compound, age, fresh/used state and stint. The manufacturer field remains for endurance-series expansion.

## Weather

- air temperature
- track temperature
- humidity
- pressure
- rainfall
- wind speed
- wind direction

Wind direction is encoded as sine/cosine so 359 degrees and 1 degree remain close in feature space.

## Race context

- position
- gap ahead
- gap behind
- track status
- race-control state when available

## Completed-lap timing

- lap time
- sector 1
- sector 2
- sector 3
- rolling median 3
- rolling median 5
- 3-lap sector means
- 3-lap pace trend
- current delta from rolling-five pace

## Optional completed-lap telemetry summary

- mean speed
- max speed
- mean throttle
- braking-active fraction
- mean RPM
- max RPM
- gear changes
- DRS-active fraction

## Temporal context

The current production feature builder includes four lagged copies of selected fields, so the model sees a five-lap context without requiring recurrent global backpropagation.

## Historical matched-condition priors

All of these use earlier events only:

- driver + circuit pace
- team + circuit pace
- compound + circuit pace
- driver + circuit + compound pace
- team + circuit + compound pace
- driver global pace
- team global pace
- circuit + compound + weather-bin pace
- driver + matched weather pace
- team + matched weather pace

## Leakage rules

Forbidden:

- any lap `N+1` telemetry
- target-lap sectors
- target-lap timestamps that reveal elapsed time
- final race results as an input to an earlier lap
- historical priors that include the current event
- preprocessing fitted on validation/test data
- random lap-level headline split

The code enforces adjacency with:

```text
next_lap_number - lap_number == 1
```
