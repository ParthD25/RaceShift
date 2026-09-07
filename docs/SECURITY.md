# Security and Local-First Boundaries

RaceShift is designed to run locally.

## Secrets

The default offline workflow needs no API key.

If optional authenticated live data is added later:

- credentials belong only in the Python backend
- never use `VITE_*` for a secret, because Vite exposes those variables to browser code
- keep `.env`, tokens and credential files out of Git
- use narrow-scope credentials when the source supports them

## Local file access

The forecast API only accepts filenames inside `data/imports/`. It resolves the path and rejects traversal outside that directory.

## Browser boundary

React receives prediction results. It does not train the model, store provider secrets or access arbitrary local files.

## Model artifact safety

Only load trusted local artifacts. `preprocessor.joblib` is a pickle-based object and should never be loaded from an untrusted source.

## Data privacy

Public motorsport telemetry is the intended data domain. Do not reuse the local file import mechanism for sensitive personal datasets without adding a separate threat model and access controls.
