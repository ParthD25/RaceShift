# FFR-M model card

RaceShift Forward-Forward regressor exported from `artifacts/f1_2025h2_ffr-m` on 2026-09-08T02:22:44+00:00.

## What it predicts

The next lap time of a Formula 1 driver from information known at the end of the current lap. The network predicts the
residual against the driver's rolling five-lap median (`rolling_median_5`); `next_lap = rolling_median_5 + residual`. It
also returns an 80% interval (validation-residual quantile widened by cross-layer disagreement).

## Architecture

- Hidden layers: 512 → 384 → 256 → 192 nodes, ordinal groups 8 / 16 / 32 / 64; each layer trained with a local ordinal-goodness objective and an explicit local Adam update. No global backpropagation.
- Readout: closed-form ridge over all layers' goodness vectors and local predictions (alpha 2.0).
- Input: 64 numeric and 12 categorical contract columns after train-only imputation, scaling and one-hot encoding (327 model inputs).

## Training data and split

- Data source: `fastf1_timing`
- Split mode: season_round; train ≤ 2024, validation 2025, test 2025, split round 12
- Rows: train 128071, validation 10248, test 10994
- Dropped sparse features: 28; dropped groups: none

## Measured performance

| Split | MAE (s) | RMSE (s) | p90 (s) | Laps within 0.5 s | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.430 | 0.708 | 0.910 | — | 0.800 |
| test | 0.352 | 0.595 | 0.740 | — | 0.857 |

## Resources

- Training wall time 2064.1 s, peak RSS 2973 MB, traced training peak 1423 MB, device cpu-numpy
- Inference 2.923 ms per single row, artifact 3.37 MB

## Files in this export

- `ffr-m_ffr.onnx`: Forward-Forward core, opset 13; verified against NumPy on 2000 rows, max |Δ| 1.12e-05 s
- `preprocessor.joblib`: fitted sklearn preprocessor (joblib). skl2onnx cannot convert SimpleImputer(add_indicator=True), so there is no preprocessor ONNX; use ffr-m_preprocessor.json with raceshift.models.export.apply_preprocessor_spec for pickle-free inference
- `ffr-m_preprocessor.json`: fitted preprocessor as plain JSON (medians, indicators, scaling, one-hot vocabularies) for pickle-free inference via raceshift.models.export.apply_preprocessor_spec; verified against sklearn on 2000 rows, max |Δ| 0.00e+00
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
art = 'artifacts/f1_2025h2_ffr-m'
contract = json.load(open(f'{art}/feature_contract.json'))
table = build_full_context_table(pd.read_parquet('laps.parquet'), history=contract['history'])
x = joblib.load(f'{art}/preprocessor.joblib').transform(table[contract['numeric'] + contract['categorical']]).astype(np.float32)
sess = ort.InferenceSession(f'{art}/export/ffr-m_ffr.onnx')
residual, lower, upper, disagreement = sess.run(None, {'features': x})
next_lap = table['rolling_median_5'].to_numpy() + residual
```
