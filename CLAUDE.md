# RaceShift: notes for coding agents

## Training policy (do not change)

Every RaceShift model is trained with **forward-forward propagation**: each layer is
fitted with its own local objective, one layer at a time, and the ridge readout is fitted
on top of the frozen layer outputs. **No global backpropagation, ever.** This applies to
initial training, to any future fine-tuning, and to any new model variant. The policy is
recorded in every artifact as
`"training_policy": "forward-forward-local-updates-no-global-backprop"` and the test suite
checks layer locality with a finite-difference gradient test.

- Regressor: `src/raceshift/models/forward_forward_regressor.py` (NumPy, CPU).
- Training entry point: `scripts/train_ffr.py` with a config from `configs/`.
- Fine-tuning on new laps means running the same local, layer-wise updates on the new
  rows (or retraining from the committed config); it never means attaching an autograd
  loss and backpropagating through the stack. Do not add PyTorch/JAX backprop paths.
- The fine-tuning entry point is `scripts/finetune_ffr.py`, which calls
  `ForwardForwardRegressor.continue_fit`: local updates from the saved weights, then a
  closed-form ridge readout refit, reusing the base artifact's preprocessor and feature
  contract. Example (adapt the 2026 model to the first five 2026 races, score rounds 8+):
  `python scripts/finetune_ffr.py --base artifacts/domain_shift_2026_ffr-m --input <laps.parquet> --season 2026 --train-rounds 1-5 --val-rounds 6-7 --test-rounds 8- --replay-rows 20000 --output artifacts/<name>`.
  Keep `--replay-rows` on: refitting the readout on the new rows alone forgets the old ones.

## Data sources

- FastF1 (2018+, races) is the primary timing tier; OpenF1 (2023+, races and sprints,
  `scripts/fetch_openf1.py`) is the second provider and the only source for sprints; the
  Ergast/Jolpica legacy tier (1996-2017, no tyres/sectors/weather) comes from the API
  (`scripts/fetch_jolpica_seasons.py`) or the Kaggle dump (`scripts/build_ergast_kaggle.py`).
  `dataset_manifest.json` records what each source is used for; `scripts/cross_provider_check.py`
  verifies that two providers describe the same laps.

## Conventions

- Commits are authored as `ParthDave <pdave7848@gmail.com>`; no co-author trailers or
  generated-by footers in commits or pull request bodies.
- Lap-validity rules are versioned (`LAP_VALIDITY_VERSION` in
  `src/raceshift/features/full_context.py`); every committed artifact must carry the
  current version, and changing the rules means rerunning `scripts/run_experiments.py`.
- Run `python -m pytest -q` and `npm run build` before pushing.
