# Reports

Measured experiment tables produced by `scripts/run_experiments.py`. Each subdirectory holds
`summary.json` (machine-readable, every metric and resource figure per run) and `summary.md`
(the table). Every table states its data source, split and row counts. Synthetic-fixture runs
are labelled synthetic and are never Formula 1 results.

Regenerate any table with the command recorded in `docs/TRAINING_AND_RESEARCH.md`; the raw
FastF1 data is not committed (see `data/raw/` policy) but is fully reproducible from the
collection scripts and FastF1's public archive.
