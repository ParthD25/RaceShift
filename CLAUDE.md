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

## Conventions

- Commits are authored as `ParthDave <pdave7848@gmail.com>`; no co-author trailers or
  generated-by footers in commits or pull request bodies.
- Lap-validity rules are versioned (`LAP_VALIDITY_VERSION` in
  `src/raceshift/features/full_context.py`); every committed artifact must carry the
  current version, and changing the rules means rerunning `scripts/run_experiments.py`.
- Run `python -m pytest -q` and `npm run build` before pushing.
