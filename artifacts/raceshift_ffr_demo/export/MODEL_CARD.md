# FFR-demo model card

RaceShift Forward-Forward regressor exported from `artifacts/raceshift_ffr_demo` on 2026-09-07T22:29:06+00:00.

## What it predicts

The next lap time of a Formula 1 driver from information known at the end of the current lap. The network predicts the
residual against the driver's rolling five-lap median (`rolling_median_5`); `next_lap = rolling_median_5 + residual`. It
also returns an 80% interval (validation-residual quantile widened by cross-layer disagreement).

## Architecture

- Hidden layers: 128 → 96 → 64 → 48 nodes, ordinal groups 8 / 8 / 8 / 8; each layer trained with a local ordinal-goodness objective and an explicit local Adam update. No global backpropagation.
- Readout: closed-form ridge over all layers' goodness vectors and local predictions (alpha 2.0).
- Input: 92 numeric and 12 categorical contract columns after train-only imputation, scaling and one-hot encoding (176 model inputs).

## Training data and split

- Data source: `synthetic_fixture` (synthetic fixture, not Formula 1 results)
- Split mode: season_forward; train ≤ 2023, validation 2024, test 2025
- Rows: train 648, validation 324, test 324
- Dropped sparse features: 0; dropped groups: none

## Measured performance

| Split | MAE (s) | RMSE (s) | p90 (s) | Laps within 0.5 s | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.156 | 0.211 | 0.367 | — | 0.799 |
| test | 0.172 | 0.222 | 0.354 | — | 0.753 |

## Resources

- Training wall time 1.4 s, peak RSS 212 MB, traced training peak 2 MB, device cpu-numpy
- Inference 0.783 ms per single row, artifact 0.24 MB

## Files in this export

- `ffr-demo_ffr.onnx`: Forward-Forward core, opset 13; verified against NumPy on 1296 rows, max |Δ| 5.96e-08 s
- `preprocessor.joblib`: fitted sklearn preprocessor (joblib). skl2onnx cannot convert SimpleImputer(add_indicator=True), so there is no preprocessor ONNX; use ffr-demo_preprocessor.json with raceshift.models.export.apply_preprocessor_spec for pickle-free inference
- `ffr-demo_preprocessor.json`: fitted preprocessor as plain JSON (medians, indicators, scaling, one-hot vocabularies) for pickle-free inference via raceshift.models.export.apply_preprocessor_spec; verified against sklearn on 1296 rows, max |Δ| 0.00e+00
- `feature_contract.json`: exact input columns, history length, dropped features
- `metrics.json`: all measured numbers for this run

## Limitations

- Laps around red-flag stoppages can pass the validity rules and produce very large errors (documented gap).
- Intervals are calibrated on the validation season; under regulation change (2026) coverage drops below the nominal 80%.
- Historical priors need earlier events in the same table; a single-race file yields missing priors, which the model treats as their own category.
- The model is a research artifact for comparing local Forward-Forward learning against baselines; the gradient-boosted tree baseline is more accurate on the same data.

## Inference

```python
import json, joblib, numpy as np, onnxruntime as ort, pandas as pd
from raceshift.features.full_context import build_full_context_table
art = 'artifacts/raceshift_ffr_demo'
contract = json.load(open(f'{art}/feature_contract.json'))
table = build_full_context_table(pd.read_parquet('laps.parquet'), history=contract['history'])
x = joblib.load(f'{art}/preprocessor.joblib').transform(table[contract['numeric'] + contract['categorical']]).astype(np.float32)
sess = ort.InferenceSession(f'{art}/export/ffr-demo_ffr.onnx')
residual, lower, upper, disagreement = sess.run(None, {'features': x})
next_lap = table['rolling_median_5'].to_numpy() + residual
```
